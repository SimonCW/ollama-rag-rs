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
    create_embeddings(&ollama, documents_path, embeddings_path);
    //
    // loop {
    //     let mut user_msg = String::new();
    //     println!(">> Awaiting your message");
    //     stdin().read_line(&mut user_msg);
    //     let user_msg = ChatMessage::new(MessageRole::User, user_msg.to_string());
    //     msg_thread.push(user_msg);
    //     // Clone really necessary?
    //     let req = ChatMessageRequest::new(MODEL.to_string(), msg_thread.clone());
    //     println!("----Assistant----");
    //     let assistant_msg = write_chat(&ollama, req).await?; // could be a union of response and final object.
    //     if let Some(assistant_msg) = assistant_msg {
    //         msg_thread.push(assistant_msg)
    //     }
    // }
    // println!("{msg_thread:#?}");
    let embeddings_path = Path::new(EMBEDDINGS_PATH);
    ensure_dir(documents_path);
    ensure_dir(embeddings_path);
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
            println!("File: \n {content}");
            let emb = ollama
                .generate_embeddings(MODEL.to_string(), content, None)
                .await?;
        }
    }
    Ok(())
}

pub async fn write_chat(
    ollama: &Ollama,
    chat_req: ChatMessageRequest,
) -> Result<Option<ChatMessage>> {
    let mut stream = ollama.send_chat_messages_stream(chat_req).await?;
    let mut stdout = tokio::io::stdout();
    let mut char_count = 0;
    let mut msg_stream: Vec<String> = Vec::new();

    while let Some(res) = stream.next().await {
        let res = res.map_err(|e| anyhow!("stream.next error: {:#?}", e))?;

        if let Some(msg) = res.message {
            let msg_content = msg.content;
            // Poor man's wrapping
            char_count += msg_content.len();
            if char_count > 80 {
                stdout.write_all(b"\n").await?;
                char_count = 0;
            }
            // Write output
            stdout.write_all(msg_content.as_bytes()).await?;
            stdout.flush().await?;
            msg_stream.push(msg_content);
        }

        if let Some(_final_res) = res.final_data {
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;

            let assistant_msg = msg_stream.join("");
            let assistant_msg = ChatMessage::new(MessageRole::Assistant, assistant_msg);
            return Ok(Some(assistant_msg));
        }
        // What if final result never comes, then I'm stuck in endless loop?
fn ensure_dir(path: &Path) -> Result<()> {
    // Check if the path exists and is a directory
    if !path.exists() || !path.is_dir() {
        // Create the directory if it doesn't exist or isn't a directory
        fs::create_dir_all(path)?;
        println!("Directory created: {:?}", path);
    } else {
        println!("Path already exists and is a directory.");
    }

    // new line
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;

    Ok(None)
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
