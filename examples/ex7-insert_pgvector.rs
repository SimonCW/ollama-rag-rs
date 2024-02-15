use anyhow::{anyhow, Context, Result};
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};
use pgvector::Vector;
use rag_rs::utils::ensure_dir;
use sqlx::{postgres::PgPoolOptions, Pool, Postgres};
use std::env;
use std::{fs, path::Path};
use text_splitter::TextSplitter;
use tokenizers::tokenizer::Tokenizer;

const DOCUMENTS_PATH: &str = "./knowledge/2024-02-13_the_rust_book.txt";
const TOKENIZEER_MODEL: &str = "bert-base-cased";
const MAX_TOKENS: usize = 1000;
const EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::MLE5Large;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let documents_path = Path::new(DOCUMENTS_PATH);
    ensure_dir(documents_path);
    let splitter = init_splitter()?;
    let model = init_model()?;
    let embedding_size = get_embedding_size(EMBEDDING_MODEL).expect("Expect to find model info");
    let db_url = env::var("DATABASE_URL").expect("Environment var DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&db_url)
        .await?;
    create_table(&pool, embedding_size).await?;

    let content = fs::read_to_string(documents_path)?;
    let chunks: Vec<_> = splitter.chunks(&content, MAX_TOKENS).collect();
    // Not happy with the clone. How expensive is a clone of a Vec<&str>?
    let embeddings = model.passage_embed(chunks.clone(), None)?;
    for (chunk, embedding) in chunks.iter().zip(embeddings) {
        let embedding = Vector::from(embedding);
        sqlx::query("INSERT INTO rag_demo (chunk, embedding) VALUES ($1,$2)")
            .bind(chunk)
            .bind(embedding)
            .execute(&pool)
            .await
            .with_context(|| format!("Failed for chunk: '{chunk}'"))?;
    }
    Ok(())
}

pub fn get_embedding_size(model: EmbeddingModel) -> Option<usize> {
    FlagEmbedding::list_supported_models()
        .iter()
        .find(|info| info.model == model)
        .map(|info| info.dim)
}

pub async fn create_table(pool: &Pool<Postgres>, embedding_size: usize) -> Result<()> {
    sqlx::query!("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await
        .context("Failed to create extension")?;
    sqlx::query!("DROP TABLE IF EXISTS rag_demo")
        .execute(pool)
        .await
        .context("Failed to drop table")?;

    // Unprepared query b/c I want to dynamically create the table based on embedding_size for now.
    // This allows me to quickly switch embedding models. Since embedding_size is an int, risk of
    // sql injection is very low
    let create_table_query = format!("CREATE TABLE IF NOT EXISTS rag_demo (id bigserial PRIMARY KEY, chunk text, embedding vector({embedding_size}))");
    sqlx::query(&create_table_query)
        .execute(pool)
        .await
        .context("Failed to create table")?;
    Ok(())
}

pub fn init_splitter() -> Result<TextSplitter<Tokenizer>> {
    let tokenizer =
        Tokenizer::from_pretrained(TOKENIZEER_MODEL, None).map_err(|e| anyhow!("{e:#?}"))?;
    let splitter = TextSplitter::new(tokenizer).with_trim_chunks(true);
    Ok(splitter)
}

pub fn init_model() -> Result<FlagEmbedding> {
    let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
        model_name: EMBEDDING_MODEL,
        show_download_message: true,
        ..Default::default()
    })?;
    Ok(model)
}
