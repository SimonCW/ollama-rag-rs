use std::io::stdout;

use futures::StreamExt;
use rag_rs::consts::{DEFAULT_SYSTEM_CLOWN, MODEL};
use rag_rs::gen::write_stream;

use anyhow::{anyhow, Context, Result};
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use tokio::io::AsyncWriteExt;

#[tokio::main]
async fn main() -> Result<()> {
    // localhost:1143
    let ollama = Ollama::default();
    let model = MODEL.to_string();

    let prompt = "What is the best programming language? (Be concise)".to_string();

    let gen_req = GenerationRequest::new(model, prompt).system(DEFAULT_SYSTEM_CLOWN.to_string());
    println!("----> Request generated");

    // Non-streaming
    // let res = ollama
    //     .generate(gen_req)
    //     .await
    //     .map_err(|e| anyhow!(e))
    //     .context("Generation failed")?;
    // println!("---> Response: {}", res.response);

    //Streamed
    write_stream(&ollama, gen_req).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
}
