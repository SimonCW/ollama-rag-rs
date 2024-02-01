use anyhow::{anyhow, Result};
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};
use rag_rs::utils::{ensure_dir, write_vec_to_json};
use std::{fs, path::Path};
use text_splitter::TextSplitter;
use tokenizers::tokenizer::Tokenizer;

const DOCUMENTS_PATH: &str = "./examples/.data/rust_book.txt";
const EMBEDDINGS_PATH: &str = "./examples/.embeddings/";

pub fn main() -> Result<()> {
    let documents_path = Path::new(DOCUMENTS_PATH);
    let embeddings_path = Path::new(EMBEDDINGS_PATH);
    ensure_dir(documents_path);
    ensure_dir(embeddings_path);
    // Splitting
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    let max_tokens = 1000;
    let splitter = TextSplitter::new(tokenizer).with_trim_chunks(true);

    let content = fs::read_to_string(documents_path)?;
    let chunks: Vec<_> = splitter.chunks(&content, max_tokens).collect();

    // Embeddings
    let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::BGEBaseEN,
        show_download_message: true,
        ..Default::default()
    })?;

    let embeddings = model.passage_embed(chunks, None)?;

    let output_path = embeddings_path.join("rust_book_embeddings.json");
    println!("Writing embeddings to {}", output_path.display());
    write_vec_to_json(&output_path, &embeddings);

    // TODO: save embeddings in DB

    Ok(())
}
