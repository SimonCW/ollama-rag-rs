use std::path::Path;

use crate::consts::{EMBEDDING_MODEL, TOKENIZER_MODEL};
use anyhow::{anyhow, Context, Result};
use fastembed::{InitOptions, TextEmbedding};
use text_splitter::TextSplitter;
use tokenizers::Tokenizer;
use tracing::{instrument, warn};

#[instrument]
pub fn init_splitter() -> Result<TextSplitter<Tokenizer>> {
    let tokenizer =
        Tokenizer::from_pretrained(TOKENIZER_MODEL, None).map_err(|e| anyhow!("{e:#?}"))?;
    let splitter = TextSplitter::new(tokenizer).with_trim_chunks(true);
    Ok(splitter)
}

#[instrument]
pub fn init_model() -> Result<TextEmbedding> {
    let model: TextEmbedding = TextEmbedding::try_new(InitOptions {
        model_name: EMBEDDING_MODEL,
        show_download_progress: true,
        ..Default::default()
    })
    .context("Failed to intitialize model {EMBEDDING_MODEL:#?}")?;
    Ok(model)
}
