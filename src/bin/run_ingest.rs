use anyhow::{Context, Result};
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding};
use pgvector::Vector;
use rag_rs::consts::{DOCUMENTS_PATH, MAX_TOKENS};
use rag_rs::ingest::{init_model, init_splitter};
use rag_rs::utils::ensure_dir;
use sqlx::postgres::PgPoolOptions;
use std::env;
use std::{fs, path::Path};
use tracing::{info, info_span};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/* TODOs
* ? Add more context to the DB? E.g. which page of the book.
*/

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
    let db_url = env::var("DATABASE_URL").expect("Environment var DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&db_url)
        .await?;
    sqlx::migrate!().run(&pool).await?;

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

pub fn get_embedding_size(model: EmbeddingModel) -> Option<usize> {
    FlagEmbedding::list_supported_models()
        .iter()
        .find(|info| info.model == model)
        .map(|info| info.dim)
}
