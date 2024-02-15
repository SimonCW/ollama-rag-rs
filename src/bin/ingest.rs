use anyhow::{anyhow, Context, Result};
use dotenv::dotenv;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions, ModelInfo};
use pgvector::Vector;
use rag_rs::utils::{ensure_dir, write_vec_to_json};
use sqlx::{postgres::PgPoolOptions, Pool, Postgres, Row};
use std::env;
use std::{any, fs, path::Path};
use text_splitter::TextSplitter;
use tokenizers::tokenizer::Tokenizer;

const DOCUMENTS_PATH: &str = "./knowledge/2024-02-13_the_rust_book_short.txt";
const TOKENIZER_MODEL: &str = "bert-base-cased";
const MAX_TOKENS: usize = 1000;
const EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::MLE5Large;

/* TODOs
* Add minimial logging via the tracing crate
* ? Add more context to the DB? E.g. which page of the book.
*/

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let documents_path = Path::new(DOCUMENTS_PATH);
    ensure_dir(documents_path);
    let splitter = init_splitter()?;
    let model = init_model()?;
    let embedding_size = get_embedding_size(EMBEDDING_MODEL).expect("Model info should be there");
    let db_url = env::var("DATABASE_URL").expect("Environment var DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&db_url)
        .await?;
    println!("Running migration");
    sqlx::migrate!().run(&pool).await?;

    // Well ... this could be better ;)
    let content = fs::read_to_string(documents_path)?;
    let chunks: Vec<_> = splitter.chunks(&content, MAX_TOKENS).collect();
    // Not happy with the clone. How expensive is a clone of a Vec<&str>?
    let embeddings = model.passage_embed(chunks.clone(), None)?;
    println!("Inserting embeddings");
    for (chunk, embedding) in chunks.iter().zip(embeddings) {
        let embedding = Vector::from(embedding);
        sqlx::query("INSERT INTO rippy (chunk, embedding) VALUES ($1,$2)")
            .bind(chunk)
            .bind(embedding)
            .execute(&pool)
            .await
            .with_context(|| format!("Failed for chunk: '{chunk}'"))?;
    }
    println!("Finished inserting embeddings");
    Ok(())
}

pub fn get_embedding_size(model: EmbeddingModel) -> Option<usize> {
    FlagEmbedding::list_supported_models()
        .iter()
        .find(|info| info.model == model)
        .map(|info| info.dim)
}

pub fn init_splitter() -> Result<TextSplitter<Tokenizer>> {
    let tokenizer =
        Tokenizer::from_pretrained(TOKENIZER_MODEL, None).map_err(|e| anyhow!("{e:#?}"))?;
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
