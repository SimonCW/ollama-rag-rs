use anyhow::{anyhow, Context, Result};
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions, ModelInfo};
use futures::TryStreamExt;
use pgvector::Vector;
use rag_rs::utils::{ensure_dir, write_vec_to_json};
use sqlx::{postgres::PgPoolOptions, Pool, Postgres, Row};
use std::env;
use std::{any, fs, path::Path};
use text_splitter::TextSplitter;
use tokenizers::tokenizer::Tokenizer;

const EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::MLE5Large;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let model = init_model()?;
    let embedding_size = get_embedding_size(EMBEDDING_MODEL).expect("Expect to find model info");
    let db_url = env::var("DATABASE_URL").expect("Environment var DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&db_url)
        .await?;

    let query = "What methods exist on iterators?";
    println!("Query: {query} \n");
    let query_embedding = Vector::from(model.query_embed(query)?);

    let mut rows = sqlx::query("SELECT id, chunk FROM rag_demo ORDER BY embedding <-> $1 LIMIT 2")
        .bind(query_embedding)
        .fetch(&pool);
    while let Some(row) = rows.try_next().await? {
        let neighbor = NearestNeighbors {
            id: row.get("id"),
            chunk: row.get("chunk"),
        };
        println!("Neighbor:\n {:#?}\n", neighbor.chunk);
    }

    Ok(())
}

#[derive(Debug)]
struct NearestNeighbors {
    id: i64,
    chunk: String,
}

pub fn get_embedding_size(model: EmbeddingModel) -> Option<usize> {
    FlagEmbedding::list_supported_models()
        .iter()
        .find(|info| info.model == model)
        .map(|info| info.dim)
}

pub fn init_model() -> Result<FlagEmbedding> {
    let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
        model_name: EMBEDDING_MODEL,
        show_download_message: true,
        ..Default::default()
    })?;
    Ok(model)
}
