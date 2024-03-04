use crate::consts::{EMBEDDING_MODEL, TOKENIZER_MODEL};
use anyhow::{anyhow, Result};
use fastembed::{FlagEmbedding, InitOptions};
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
pub fn init_model() -> Result<FlagEmbedding> {
    let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
        model_name: EMBEDDING_MODEL,
        show_download_message: true,
        ..Default::default()
    })?;
    Ok(model)
}
