use anyhow::{Context, Result};
use arrow_array::StringArray;
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use dotenv::dotenv;
use fastembed::TextEmbedding;
use futures::TryStreamExt;
use lancedb::{query::ExecutableQuery, Table};
use rag_rs::embed::init_model;
use std::{env, io::stdin};

const TABLE_NAME: &str = "EmbeddingsTable";

#[tokio::main]
async fn main() -> Result<()> {
    // Setup
    dotenv().ok();
    let model = init_model()?;

    let db_uri = env::var("DATABASE_PATH").expect("Environment var DATABASE_PATH must be set");
    let conn = lancedb::connect(&db_uri).execute().await?;
    let tbl = conn.open_table(TABLE_NAME).execute().await?;

    let local_conf = OpenAIConfig::new()
        .with_api_key("sk-no-key-required")
        .with_api_base("http://localhost:8080/v1");
    let client = Client::with_config(local_conf);

    // Create a thread for the conversation
    let thread_request = CreateThreadRequestArgs::default().build()?;
    let thread = client.threads().create(thread_request.clone()).await?;

    // Chat
    let system = ChatCompletionRequestSystemMessageArgs::default()
                .content("Use the provided CONTEXT to answer questions. Documents in the CONTEXT are delimted with <--->. If the answer cannot be found in the CONTEXT, write 'I could not find an answer.'")
                .build()?;

    let mut msg_thread: Vec<ChatCompletionRequestMessage> = vec![system.into()];

    loop {
        // Read user message from stdin
        println!(">> Awaiting your message");
        let mut query = String::new();
        let _ = stdin().read_line(&mut query);

        // Retrieve neighbors
        let nn_chunks = get_nearest_neighbor_chunks(&query, &model, &tbl).await?;
        let context = nn_chunks[..2].join("\n<--->\n");

        let user_msg = ChatCompletionRequestUserMessageArgs::default()
            .content(format!(
                "Use the provided CONTEXT to answer the QUESTION. 
            QUESTION: {query}\n\n
            CONTEXT: {context}"
            ))
            .build()?;

        msg_thread.push(user_msg.into());

        let request = CreateChatCompletionRequestArgs::default()
            .n(1)
            .messages(msg_thread.clone())
            .build()
            .context("Failed to build ChatCompletionRequest")?;

        let response = client
            .chat()
            .create(request)
            .await
            .context("Failed to create CompletionResponse")?
            .choices
            .first()
            .unwrap()
            .message;
        msg_thread.push(response.into());
        println!("\nResponse:\n");
        let response_text = response.content.as_ref().unwrap();
        print!("{response_text}");
    }
    Ok(())
}

async fn get_nearest_neighbor_chunks(
    query: &str,
    model: &TextEmbedding,
    table: &Table,
) -> Result<Vec<String>> {
    // TODO: I might wrap LanceDB and the EmbeddingModel into one VectorStore and implement this as
    // a function on this new type.
    let query_embedding = model
        .embed(vec![query], None)?
        .pop()
        .expect("Outer Vec will contain one inner vec");
    let nearest_neighbors = table
        .query()
        .nearest_to(query_embedding)
        .context("Probably cannot convert input vector")?
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(
        1,
        nearest_neighbors.len(),
        "Pretty sure this will always be one ... but not certain"
    );
    let nn = &nearest_neighbors[0];
    let nn_chunks = nn
        .column_by_name("text")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .flatten()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();
    Ok(nn_chunks)
}
