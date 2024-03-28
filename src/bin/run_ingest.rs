use anyhow::{anyhow, Context, Result};
use arrow_array::{
    types::Float32Type, types::Int32Type, FixedSizeListArray, Int32Array, RecordBatch,
    RecordBatchIterator,
};
use arrow_array::{ArrayRef, StringArray};
use arrow_schema::{DataType, Field, Schema};
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding};
use lancedb::{Connection, Table};
use rag_rs::consts::{EMBEDDINGSIZE, MAX_TOKENS};
use rag_rs::embed::{init_model, init_splitter};
use rag_rs::utils::ensure_dir;
use std::env;
use std::sync::Arc;
use std::{fs, path::Path};
use tracing::{info, info_span, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// TODO: ? Add more context to the DB? E.g. which page of the book.

const DOCUMENTS_PATH: &str = "./knowledge/2024-02-13_the_rust_book.txt";
const TABLE_NAME: &str = "EmbeddingsTable";

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
    //let default_conf = embeddingsdb::Config::default();
    let db_uri = env::var("DATABASE_PATH").expect("Environment var DATABASE_PATH must be set");
    let conn = lancedb::connect(&db_uri).execute().await?;

    // Well ... this could be better ;)
    let content = fs::read_to_string(documents_path)?;
    let chunks: Vec<_> = splitter.chunks(&content, MAX_TOKENS).collect();
    // Not happy with the clone. How expensive is a clone of a Vec<&str>?
    info!("Creating embeddings");
    let embeddings = model.passage_embed(chunks.clone(), None)?;
    assert_eq!(embeddings.len(), chunks.len());
    assert_eq!(embeddings[0].len() as i32, EMBEDDINGSIZE);
    let n_items = chunks.len();
    info!("Inserting embeddings");

    // Create Schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, true),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDINGSIZE,
            ),
            true,
        ),
    ]));
    let tbl = create_or_overwrite_table(&conn, TABLE_NAME, schema.clone()).await?;
    // Convert data to RecordBatch stream.
    let batches = RecordBatchIterator::new(
        vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(1..=n_items as i32)),
                Arc::new(Arc::new(StringArray::from(chunks)) as ArrayRef),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        embeddings
                            .into_iter()
                            .map(|inner_vec| Some(inner_vec.into_iter().map(|item| Some(item)))),
                        EMBEDDINGSIZE,
                    ),
                ),
            ],
        )
        .context("Creating RecordBatch failed")?]
        .into_iter()
        .map(Ok),
        schema.clone(),
    );
    // Create Table
    tbl.add(batches).execute().await?;
    info!("Finished inserting embeddings");
    Ok(())
}

async fn create_or_overwrite_table(
    conn: &Connection,
    name: &str,
    schema: Arc<Schema>,
) -> Result<Table> {
    let res = conn.open_table(name).execute().await;
    let table = match res {
        Ok(_) => {
            warn!("Dropping existing table {name}.");
            conn.drop_table(name)
                .await
                .context("Failed to drop table {name}")?;
            conn.create_empty_table(name, schema)
                .execute()
                .await
                .context("Failed to create empty table {name}")?
        }
        Err(_) => {
            info!("Creating empty table {name}.");
            conn.create_empty_table(name, schema)
                .execute()
                .await
                .context("Failed to create empty table {name}")?
        }
    };
    Ok(table)
}
async fn create_table_if_not_exists(
    conn: &Connection,
    name: &str,
    schema: Arc<Schema>,
) -> Result<Table> {
    let res = conn.open_table(name).execute().await;
    let table = match res {
        Ok(tbl) => {
            info!("Table {name} already exists.");
            tbl
        }
        Err(_) => {
            info!("Creating empty table {name}.");
            conn.create_empty_table(name, schema)
                .execute()
                .await
                .context("Failed to create empty table {name}")?
        }
    };
    Ok(table)
}

pub fn get_embedding_size(model: EmbeddingModel) -> Option<usize> {
    FlagEmbedding::list_supported_models()
        .iter()
        .find(|info| info.model == model)
        .map(|info| info.dim)
}
