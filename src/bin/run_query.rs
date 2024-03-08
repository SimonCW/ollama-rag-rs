use anyhow::Result;
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};
use futures::TryStreamExt;
use pgvector::Vector;
use rag_rs::embed::init_model;
use sqlx::{postgres::PgPoolOptions, Row};
use std::env;
use tracing::{info, info_span};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    // Setup
    dotenv().ok();
    let model = init_model()?;
    let db_url = env::var("DATABASE_URL").expect("Environment var DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&db_url)
        .await?;

    let query = "What's the 'interior mutability' about and how to achieve it?";
    println!("Query: {query} \n");
    let query_embedding = Vector::from(model.query_embed(query)?);

    todo!("Refactor neighbor retrieval");
    let mut rows = sqlx::query("SELECT id, chunk FROM rippy ORDER BY embedding <-> $1 LIMIT 2")
        .bind(query_embedding)
        .fetch(&pool);
    while let Some(row) = rows.try_next().await? {
        let neighbor = NearestNeighbors {
            id: row.get("id"),
            chunk: row.get("chunk"),
        };
        println!(
            "Neighbor {}:\n {}\n--------------------------------------------------------------",
            neighbor.id, neighbor.chunk
        );
    }
    todo!("Create prompt");
    todo!("Chat with the user");
}

#[derive(Debug)]
struct NearestNeighbors {
    id: i64,
    chunk: String,
}
