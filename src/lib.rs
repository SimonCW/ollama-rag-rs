#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![allow(clippy::missing_errors_doc)]

pub mod embed;

pub mod consts {
    use fastembed::EmbeddingModel;

    pub const DOCUMENTS_PATH: &str = "./knowledge/2024-02-13_the_rust_book_short.txt";
    pub const TOKENIZER_MODEL: &str = "bert-base-cased";
    pub const MAX_TOKENS: usize = 1000;
    pub const EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::MLE5Large;

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
    use anyhow::{anyhow, Result};
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

        /*
              while let Some(Ok(res)) = stream.next().await {
             for ele in res {
                 stdout.write_all(ele.response.as_bytes()).await?;
                 stdout.flush().await?;

                 if let Some(final_data) = ele.final_data {
                     context = Some(final_data.context);
                 }

        */

        while let Some(res) = stream.next().await {
            let res = res.map_err(|e| anyhow!("Error while streaming the next token: {e:#?}"))?;
            for element in res {
                let bytes = element.response.as_bytes();
                char_count += bytes.len();
                if char_count > 80 {
                    stdout.write_all(b"\n").await?;
                    char_count = 0;
                };
                stdout.write_all(bytes).await?;
                stdout.flush().await?;

                if let Some(final_data) = element.final_data {
                    stdout.write_all(b"\n---------------\n").await?;
                    stdout.flush().await?;
                    return Ok(Some(final_data));
                }
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
    use tracing::{info, instrument, warn};

    /// Check if a path exists and is a directory, create the directory otherwise.
    #[instrument]
    pub fn ensure_dir(path: &Path) -> Result<()> {
        if path.exists() && path.is_dir() {
            info!("{} exists and is a directory", path.display());
        } else if !path.exists() || !path.is_dir() {
            fs::create_dir_all(path)?;
            info!("Directory created: {}", path.display());
        } else {
            warn!("Something weird happening at {}.", path.display());
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
