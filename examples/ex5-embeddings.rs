use std::fs;

use std::path::Path;

use rag_rs::consts::MODEL;

use rag_rs::utils::{ensure_dir, write_vec_to_json};

use anyhow::{Context, Result};
use ollama_rs::Ollama;

use walkdir::WalkDir;

const DOCUMENTS_PATH: &str = "./examples/small_documents/";
const EMBEDDINGS_PATH: &str = "./examples/.embeddings/";

#[tokio::main]
async fn main() -> Result<()> {
    // localhost:1143
    let ollama = Ollama::default();

    let documents_path = Path::new(DOCUMENTS_PATH);
    let embeddings_path = Path::new(EMBEDDINGS_PATH);
    let _ = ensure_dir(documents_path);
    let _ = ensure_dir(embeddings_path);

    write_embeddings(&ollama, documents_path, embeddings_path).await?;
    Ok(())
}

/// Walk a directory and create embeddings for each document
pub async fn write_embeddings(
    ollama: &Ollama,
    input_path: &Path,
    output_path: &Path,
) -> Result<()> {
    let dir = WalkDir::new(input_path);
    for entry in dir {
        let entry = entry.context("Should be able to read contents of directory")?;
        // TODO: Should probably check whethter it is a text file
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
            let _ = write_vec_to_json(&output_path, &embeddings);
        }
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
