use anyhow::Result;
use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};

pub fn main() -> Result<()> {
    // With default InitOptions
    let model: FlagEmbedding = FlagEmbedding::try_new(Default::default())?;

    // With custom InitOptions
    let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::BGEBaseEN,
        show_download_message: true,
        ..Default::default()
    })?;

    let documents = vec![
        "passage: Hello, World!",
        "query: Hello, World!",
        "passage: This is an example passage.",
        // You can leave out the prefix but it's recommended
        "fastembed-rs is licensed under Apache  2.0",
    ];

    // Generate embeddings with the default batch size, 256
    let embeddings = model.embed(documents, None)?;

    println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
    println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 768
    Ok(())
}
