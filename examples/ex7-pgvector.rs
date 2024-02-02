use anyhow::{anyhow, Result};
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};
use rag_rs::utils::{ensure_dir, write_vec_to_json};
use std::{any, fs, path::Path};
use text_splitter::TextSplitter;
use tokenizers::tokenizer::Tokenizer;

const DOCUMENTS_PATH: &str = "./examples/.data/rust_book.txt";
const EMBEDDINGS_PATH: &str = "./examples/.embeddings/";
const TOKENIZEER_MODEL: &str = "bert-base-cased";
const MAX_TOKENS: usize = 1000;

pub fn main() -> Result<()> {
    let documents_path = Path::new(DOCUMENTS_PATH);
    ensure_dir(documents_path);
    let splitter = init_splitter()?;
    let model = init_model()?;

    let content = fs::read_to_string(documents_path)?;
    let chunks = splitter.chunks(&content, MAX_TOKENS).collect();
    let embeddings = model.passage_embed(chunks, None)?;
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
