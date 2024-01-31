#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![allow(clippy::missing_errors_doc)]

pub mod consts {
    pub const MODEL: &str = "mistral";
    pub const SYSTEM_CLOWN: &str = r#"
    You are a troll LLM.

    Always reply by mocking the question asker.
    "#;
    pub const SYSTEM_DEFAULT: &str = r#"
    Be super concise!
    "#;
}

pub mod gen {
    use anyhow::{anyhow, Context, Result};
    use futures::StreamExt;
    use ollama_rs::{
        generation::completion::{request::GenerationRequest, GenerationFinalResponseData},
        Ollama,
    };
    use tokio::io::AsyncWriteExt;

    pub async fn write_stream(
        ollama: &Ollama,
        gen_req: GenerationRequest,
    ) -> Result<Option<GenerationFinalResponseData>> {
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

            if let Some(final_data) = res.final_data {
                stdout.write_all(b"\n---------------\n").await?;
                stdout.flush().await?;
                return Ok(Some(final_data));
            }
        }
        stdout.write_all(b"\n---------------\n").await?;
        stdout.flush().await?;
        Ok(None)
    }
}

pub mod utils {
    use anyhow::Result;
    use serde::Serialize;
    use std::fs;
    use std::path::Path;

    /// Check if a path exists and is a directory, create the directory otherwise.
    pub fn ensure_dir(path: &Path) -> Result<()> {
        if !path.exists() || !path.is_dir() {
            fs::create_dir_all(path)?;
            println!("Directory created: {}", path.display());
        } else {
            println!("{} already exists and is a directory.", path.display());
        }
        Ok(())
    }

    pub fn write_vec_to_json<T: Serialize>(path: &Path, vec: &Vec<T>) -> Result<()> {
        let serialized = serde_json::to_string(vec)?;
        let mut file = std::fs::File::create(path)?;
        std::io::Write::write_all(&mut file, serialized.as_bytes())?;
        Ok(())
    }
}
