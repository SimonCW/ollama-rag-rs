#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![allow(clippy::missing_errors_doc)]

pub mod consts {
    pub const MODEL: &str = "mistral";
    pub const DEFAULT_SYSTEM_CLOWN: &str = r#"
    You are a troll LLM.

    Always reply by mocking the question asker.
    "#;
}

pub mod gen {
    use anyhow::{anyhow, Context, Result};
    use futures::StreamExt;
    use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
    use tokio::io::AsyncWriteExt;

    pub async fn write_stream(ollama: &Ollama, gen_req: GenerationRequest) -> Result<()> {
        let mut stream = ollama.generate_stream(gen_req).await?;
        let mut stdout = tokio::io::stdout();
        let mut char_count = 0;

        while let Some(res) = stream.next().await {
            let res = res.map_err(|e| anyhow!("Error while streaming the next token: {e:#?}"))?;
            let bytes = res.response.as_bytes();
            char_count += bytes.len();
            if char_count > 80 {
                stdout.write_all(b"\n").await?;
                char_count = 0;
            };
            stdout.write_all(bytes).await?;
            stdout.flush().await?;
        }
        Ok(())
    }
}
