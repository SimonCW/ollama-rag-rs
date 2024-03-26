use anyhow::{anyhow, Context, Result};
use arrow_array::{
    types::Float32Type, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator,
};
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding};
use rag_rs::consts::{DOCUMENTS_PATH, EMBEDDINGSIZE, MAX_TOKENS};
use rag_rs::embed::{init_model, init_splitter};
use rag_rs::embeddingsdb::{self, Client};
use rag_rs::utils::ensure_dir;
use std::env;
use std::sync::Arc;
use std::{fs, path::Path};
use tracing::{info, info_span};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// TODO: ? Add more context to the DB? E.g. which page of the book.

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();
    let span = info_span!("Main execution");
    let _enter = span.enter();
    let documents_path = Path::new(DOCUMENTS_PATH);
    let _ = ensure_dir(documents_path);
    let splitter = init_splitter()?;
    let model = init_model()?;

    // Init LanceDB
    let default_conf = embeddingsdb::Config::default();
    let db_uri = env::var("DATABASE_PATH").expect("Environment var DATABASE_PATH must be set");
    let conn = lancedb::connect(&db_uri).execute().await?;
    let tbl = conn.create_empty_table(default_conf.table_name, default_conf.schema);

    // Well ... this could be better ;)
    let content = fs::read_to_string(documents_path)?;
    let chunks: Vec<_> = splitter.chunks(&content, MAX_TOKENS).collect();
    let n_items = chunks.len();
    // Not happy with the clone. How expensive is a clone of a Vec<&str>?
    info!("Creating embeddings");
    let embeddings = model.passage_embed(chunks.clone(), None)?;
    assert_eq!(embeddings.len(), n_items);
    info!("Inserting embeddings");
    let columns = vec![
        Arc::new(Int32Array::from_iter_values(1..n_items as i32)),
        Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                embeddings
                    .into_iter()
                    .map(|inner_vec| Some(inner_vec.into_iter().map(|item| Some(item)).collect()))
                    .collect(),
                EMBEDDINGSIZE,
            ),
        ),
    ];
    //vec![RecordBatch::try_new(default_conf.schema, columns)];

    info!("Finished inserting embeddings");
    Ok(())
}
pub fn get_embedding_size(model: EmbeddingModel) -> Option<usize> {
    FlagEmbedding::list_supported_models()
        .iter()
        .find(|info| info.model == model)
        .map(|info| info.dim)
}
