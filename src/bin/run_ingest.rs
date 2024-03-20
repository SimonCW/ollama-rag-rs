use anyhow::{anyhow, Context, Result};
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding};
use lancedb::{connection::Connection, Table};
use rag_rs::consts::{DOCUMENTS_PATH, MAX_TOKENS};
use rag_rs::embed::{init_model, init_splitter};
use rag_rs::utils::ensure_dir;
use std::env;
use std::sync::Arc;
use std::{fs, path::Path};
use tracing::{info, info_span};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// TODO: ? Add more context to the DB? E.g. which page of the book.
const EMBEDDINGSIZE: i32 = 1024;

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
    let db_path = env::var("DATABASE_PATH").expect("Environment var DATABASE_PATH must be set");
    let db = lancedb::connect(&db_path).execute().await.unwrap();

    // Well ... this could be better ;)
    let content = fs::read_to_string(documents_path)?;
    let chunks: Vec<_> = splitter.chunks(&content, MAX_TOKENS).collect();
    // Not happy with the clone. How expensive is a clone of a Vec<&str>?
    info!("Creating embeddings");
    let embeddings = model.passage_embed(chunks.clone(), None)?;
    info!("Inserting embeddings");
    for (chunk, embedding) in chunks.iter().zip(embeddings) {
        let embedding = Vector::from(embedding);
        sqlx::query("INSERT INTO rippy (chunk, embedding) VALUES ($1,$2)")
            .bind(chunk)
            .bind(embedding)
            .execute(&pool)
            .await
            .with_context(|| format!("Failed for chunk: '{chunk}'"))?;
    }
    info!("Finished inserting embeddings");
    Ok(())
}

async fn create_empty_table(db: &Connection) -> Result<Table> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, true),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDINGSIZE,
            ),
            true,
        ),
    ]));
    db.create_empty_table("empty_table", schema)
        .execute()
        .await
        .map_err(|e| anyhow!("{e:#?}"))
}
pub fn get_embedding_size(model: EmbeddingModel) -> Option<usize> {
    FlagEmbedding::list_supported_models()
        .iter()
        .find(|info| info.model == model)
        .map(|info| info.dim)
}
