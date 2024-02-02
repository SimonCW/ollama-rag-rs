use anyhow::{anyhow, Result};
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};
use rag_rs::utils::{ensure_dir, write_vec_to_json};
use std::{fs, path::Path};
use text_splitter::TextSplitter;
use tokenizers::tokenizer::Tokenizer;

const DOCUMENTS_PATH: &str = "./examples/.data/rust_book.txt";
const EMBEDDINGS_PATH: &str = "./examples/.embeddings/";
const EMBEDDING_MODEL: &str = "bert-base-cased";
const MAX_TOKENS: usize = 1000;

pub fn main() -> Result<()> {
    let documents_path = Path::new(DOCUMENTS_PATH);
    ensure_dir(documents_path);
    let chunks = split_document(documents_path)?;
    // Embeddings
    let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::BGEBaseEN,
        show_download_message: true,
        ..Default::default()
    })?;

    let embeddings = model.passage_embed(chunks, None)?;
    Ok(())
}

pub fn split_document(path: &Path) -> Result<Vec<&str>> {
    let tokenizer = Tokenizer::from_pretrained(EMBEDDING_MODEL, None).unwrap();
    let max_tokens = MAX_TOKENS;
    let splitter = TextSplitter::new(tokenizer).with_trim_chunks(true);
    let content = fs::read_to_string(path)?;
    let chunks = splitter.chunks(&content, max_tokens).collect();
    Ok(chunks)
}
