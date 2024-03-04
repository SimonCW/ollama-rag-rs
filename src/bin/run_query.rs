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
    todo!("Create prompt");
    todo!("Chat with the user");
}
