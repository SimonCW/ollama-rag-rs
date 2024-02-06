use anyhow::{anyhow, Result};
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};
use dotenv::dotenv;
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

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let documents_path = Path::new(DOCUMENTS_PATH);
    ensure_dir(documents_path);
    let db_url = env::var("DATABASE_URL").expect("Environment var DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&db_url)
        .await?;

    init_db(&pool).await?;
    test_db(&pool).await?;

    /*
    // Handling content
    let splitter = init_splitter()?;
    let model = init_model()?;

    let content = fs::read_to_string(documents_path)?;
    let chunks = splitter.chunks(&content, MAX_TOKENS).collect();
    let embeddings = model.passage_embed(chunks, None)?;
    */
    Ok(())
}

pub async fn test_db(pool: &Pool<Postgres>) -> Result<()> {
    let embedding = Vector::from(vec![1.0, 2.0, 3.0]);
    sqlx::query("INSERT INTO item (embedding) VALUES ($1)")
        .bind(&embedding)
        .execute(pool)
        .await?;

    /*
    let rows = sqlx::query("SELECT * FROM item ORDER BY embedding <-> $1 LIMIT 1")
        .bind(&embedding)
        .fetch_all(pool)
        .await?;
    */

    let row = sqlx::query("SELECT embedding FROM item LIMIT 1")
        .fetch_one(pool)
        .await?;
    let from_table: Vector = row.try_get("embedding")?;
    println!("{:#?}", from_table);
    Ok(())
}

pub async fn init_db(pool: &Pool<Postgres>) -> Result<()> {
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await?;
    sqlx::query("CREATE TABLE IF NOT EXISTS item (id bigserial PRIMARY KEY, embedding vector(3))")
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
        model_name: EmbeddingModel::BGEBaseEN,
        show_download_message: true,
        ..Default::default()
    })?;
    Ok(model)
}
