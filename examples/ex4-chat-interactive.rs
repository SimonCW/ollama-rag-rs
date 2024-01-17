use std::io::{stdin, stdout};

use futures::StreamExt;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, MessageRole};
use ollama_rs::generation::completion::GenerationContext;
use rag_rs::consts::{MODEL, SYSTEM_DEFAULT};
use rag_rs::gen::write_stream;

use anyhow::{anyhow, Context, Result};
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<()> {
    // localhost:1143
    let ollama = Ollama::default();
    let system_msg = ChatMessage::new(MessageRole::System, SYSTEM_DEFAULT.to_string());
    let mut msg_thread: Vec<ChatMessage> = vec![system_msg];

    loop {
        let mut user_msg = String::new();
        println!(">> Awaiting your message");
        stdin().read_line(&mut user_msg);
        let user_msg = ChatMessage::new(MessageRole::User, user_msg.to_string());
        msg_thread.push(user_msg);
        // Clone really necessary?
        let req = ChatMessageRequest::new(MODEL.to_string(), msg_thread.clone());
        println!("----Assistant----");
        let assistant_msg = write_chat(&ollama, req).await?; // could be a union of response and final object.
        if let Some(assistant_msg) = assistant_msg {
            msg_thread.push(assistant_msg)
        }
    }
    println!("{msg_thread:#?}");
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
