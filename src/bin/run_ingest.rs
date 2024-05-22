use anyhow::{Context, Result};
use arrow_array::{
    types::Float32Type, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator,
};
use arrow_array::{ArrayRef, StringArray};
use arrow_schema::{DataType, Field, Schema};
use dotenv::dotenv;
use fastembed::{EmbeddingModel, TextEmbedding};
use lancedb::connection::CreateTableMode;
use lancedb::{Connection, Table};
use rag_rs::consts::{EMBEDDINGSIZE, MAX_TOKENS};
use rag_rs::embed::{init_model, init_splitter};
use rag_rs::utils::ensure_dir;
use std::env;
use std::sync::Arc;
use std::{fs, path::Path};
use tracing::{info, info_span};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// TODO: ? Add more context to the DB? E.g. which page of the book.
// TODO: Fix by moving to chat/completions endpoint. I changed this to use the assitants endpoint via async_openai but that isn't supported by llamafile.

const DOCUMENT_PATH: &str = "./knowledge/2024-02-13_the_rust_book.txt";
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
    let document_path = Path::new(DOCUMENT_PATH);
    let _ = ensure_dir(document_path);
    let splitter = init_splitter()?;
    let model = init_model()?;

    // Init LanceDB
    //let default_conf = embeddingsdb::Config::default();
    let db_uri = env::var("DATABASE_PATH").expect("Environment var DATABASE_PATH must be set");
    let conn = lancedb::connect(&db_uri).execute().await?;

    // Well ... this could be better ;)
    let content = fs::read_to_string(document_path).context("Failed to read documents")?;
    let chunks: Vec<_> = splitter
        .chunks(&content, MAX_TOKENS)
        .map(|text| format!("passage: {text}"))
        .collect();
    // Not happy with the clone. How expensive is a clone of a Vec<&str>?
    info!("Creating embeddings");
    let embeddings = model.embed(chunks.clone(), None)?;
    assert_eq!(embeddings.len(), chunks.len());
    assert_eq!(i32::try_from(embeddings[0].len()).unwrap(), EMBEDDINGSIZE);
    let n_items = i32::try_from(chunks.len())
        .expect("I don't expect number of vectors to be bigger than {i32::MAX}");
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
                Arc::new(Int32Array::from_iter_values(1..=n_items)),
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
    info!("Creating empty table {name}.");
    let table = conn
        .create_empty_table(name, schema)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .context("Failed to create empty table {name}")?;
    Ok(table)
}

pub fn get_embedding_size(model: EmbeddingModel) -> Option<usize> {
    TextEmbedding::list_supported_models()
        .iter()
        .find(|info| info.model == model)
        .map(|info| info.dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_or_overwrite_table;
    use std::fs;
    use std::path::Path;

    const DB_URI: &str = ".test_data/test_db";
    const TABLE_NAME: &str = "test_table";

    #[tokio::test]
    async fn should_create_table_if_not_exists() {
        let _ = fs::remove_dir_all(DB_URI);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("item", DataType::Utf8, true),
        ]));
        let conn = lancedb::connect(DB_URI).execute().await.unwrap();
        let _ = create_or_overwrite_table(&conn, TABLE_NAME, schema)
            .await
            .unwrap();
        let db_path = Path::new(DB_URI);
        assert!(db_path.exists());
        let _ = fs::remove_dir_all(DB_URI);
    }
}
