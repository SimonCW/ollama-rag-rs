use anyhow::{Context, Result};
use dotenv::dotenv;
use fastembed::EmbeddingBase;
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use rag_rs::embed::init_model;
use std::env;

const TABLE_NAME: &str = "EmbeddingsTable";

#[tokio::main]
async fn main() -> Result<()> {
    // Setup;
    dotenv().ok();
    let model = init_model()?;
    let db_uri = env::var("DATABASE_PATH").expect("Environment var DATABASE_PATH must be set");
    let conn = lancedb::connect(&db_uri).execute().await?;
    let tbl = conn.open_table(TABLE_NAME).execute().await?;

    let query = "What's the 'interior mutability' about and how to achieve it?";
    println!("Query: {query} \n");
    let query_embedding = model.query_embed(query)?;

    let nearest_neighbors = tbl
        .query()
        .nearest_to(query_embedding)
        .context("Probably cannot convert input vector")?
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    todo!("Create prompt");
    todo!("Chat with the user");
}

#[derive(Debug)]
struct NearestNeighbors {
    id: i64,
    chunk: String,
}
