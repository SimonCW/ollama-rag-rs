use std::fs;
use std::io::{stdin, stdout};
use std::path::Path;

use futures::StreamExt;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, MessageRole};
use ollama_rs::generation::completion::GenerationContext;
use rag_rs::consts::{MODEL, SYSTEM_DEFAULT};
use rag_rs::gen::write_stream;

use anyhow::{anyhow, Context, Result};
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use walkdir::WalkDir;

const DOCUMENTS_PATH: &str = "./examples/documents/";
const EMBEDDINGS_PATH: &str = "./examples/.embeddings/";

#[tokio::main]
async fn main() -> Result<()> {
    // localhost:1143
    let ollama = Ollama::default();
    let system_msg = ChatMessage::new(MessageRole::System, SYSTEM_DEFAULT.to_string());
    let mut msg_thread: Vec<ChatMessage> = vec![system_msg];

    let documents_path = Path::new(DOCUMENTS_PATH);
    let embeddings_path = Path::new(EMBEDDINGS_PATH);
    ensure_dir(documents_path);
    ensure_dir(embeddings_path);
    create_embeddings(&ollama, documents_path, embeddings_path).await?;
    Ok(())
}

pub async fn create_embeddings(
    ollama: &Ollama,
    input_path: &Path,
    output_path: &Path,
) -> Result<()> {
    let dir = WalkDir::new(input_path);
    for entry in dir {
        let entry = entry.context("Should be able to read contents of directory")?;
        if entry.file_type().is_file() {
            println!("Found file: {}", entry.path().display());
            // Of course the reality is more tricky! E.g., what if the file is super big?
            let content = fs::read_to_string(entry.path())?;
            let embeddings = ollama
                .generate_embeddings(MODEL.to_string(), content, None)
                .await?
                .embeddings;
            let stem = entry
                .path()
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("Failed to extract file stem");
            let file_name = format!("{stem}_embeddings.json",);
            let output_path = output_path.join(file_name);
            println!("Writing embeddings to {}", output_path.display());
            write_vec_to_json(&output_path, &embeddings);
        }
    }
    Ok(())
}

fn write_vec_to_json(path: &Path, vec: &Vec<f64>) -> Result<()> {
    // Serialize the vector into a JSON string
    let serialized = serde_json::to_string(vec)?;
    // Open a file in write mode
    let mut file = std::fs::File::create(path)?;
    // Write the JSON string to the file
    std::io::Write::write_all(&mut file, serialized.as_bytes())?;
    Ok(())
}

fn ensure_dir(path: &Path) -> Result<()> {
    // Check if the path exists and is a directory
    if !path.exists() || !path.is_dir() {
        // Create the directory if it doesn't exist or isn't a directory
        fs::create_dir_all(path)?;
        println!("Directory created: {:?}", path);
    } else {
        println!("Path already exists and is a directory.");
    }

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playground() {
        let mut v: Vec<i32> = Vec::new();
        v.push(1);
        v.push(2);
        println!("{v:#?}");
    }
}
