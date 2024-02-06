use anyhow::{anyhow, Result};
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions, ModelInfo};
use pgvector::Vector;
use rag_rs::utils::{ensure_dir, write_vec_to_json};
use sqlx::{postgres::PgPoolOptions, Pool, Postgres, Row};
use std::env;
use std::{any, fs, path::Path};
use text_splitter::TextSplitter;
use tokenizers::tokenizer::Tokenizer;

const DOCUMENTS_PATH: &str = "./examples/.data/rust_book.txt";
const EMBEDDINGS_PATH: &str = "./examples/.embeddings/";
const TOKENIZEER_MODEL: &str = "bert-base-cased";
const MAX_TOKENS: usize = 1000;
const PG_CONNECTION: &str = "postgres://simon.weiss@localhost:5432/rag";
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
    todo!("There is a database error in the following. Change to the sqlx query macro to check at compile time");
    for (chunk, embedding) in chunks.iter().zip(embeddings) {
        let embedding = Vector::from(embedding);
        sqlx::query("INSERT INTO rag_demo (chunk, embedding) VALUES ($1,$2)")
            .bind(chunk)
            .bind(embedding)
            .execute(&pool)
            .await?;
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
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await?;
    sqlx::query("DROP TABLE IF EXISTS rag_demo")
        .execute(pool)
        .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS rag_demo (id bigserial PRIMARY KEY, chunk text, embedding vector($1))",
    )
    .bind(embedding_size as i64)
    .execute(pool)
    .await?;
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
