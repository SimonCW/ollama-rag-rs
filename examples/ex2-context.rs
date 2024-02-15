

use futures::StreamExt;
use ollama_rs::generation::completion::GenerationContext;
use rag_rs::consts::{MODEL, SYSTEM_DEFAULT};
use rag_rs::gen::write_stream;

use anyhow::{Context, Result};
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};


#[tokio::main]
async fn main() -> Result<()> {
    // localhost:1143
    let ollama = Ollama::default();

    let prompts = &[
        "What's the capital of France",
        "What is the capital of Germany?",
        "What was my first question?",
    ];

    let mut last_ctx: Option<GenerationContext> = None;
    for prompt in prompts {
        let mut gen_req = GenerationRequest::new(MODEL.to_string(), prompt.to_string())
            .system(SYSTEM_DEFAULT.to_string());
        if let Some(ctx) = last_ctx.take() {
            gen_req = gen_req.context(ctx);
        }
        println!(">>> {prompt}");
        let final_data = write_stream(&ollama, gen_req).await?;
        if let Some(data) = final_data {
            last_ctx = Some(data.context);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
}
