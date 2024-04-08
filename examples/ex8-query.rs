use anyhow::Result;
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};
use futures::TryStreamExt;
use pgvector::Vector;
use sqlx::{postgres::PgPoolOptions, Row};
use std::env;

const EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::MLE5Large;

#[tokio::main]
async fn main() -> Result<()> {
    unimplemented!("You need to install sqlx, pgvector and postgres for these examples. This was written befor switching to LanceDB");
    dotenv().ok();
    let model = init_model()?;
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
        println!(
            "Neighbor {}:\n {}\n--------------------------------------------------------------",
            neighbor.id, neighbor.chunk
        );
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
