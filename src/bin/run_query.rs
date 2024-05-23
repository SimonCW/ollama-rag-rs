use anyhow::{Context, Result};
use arrow_array::StringArray;
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client,
};
use dotenv::dotenv;
use fastembed::TextEmbedding;
use futures::TryStreamExt;
use lancedb::{query::ExecutableQuery, Table};
use rag_rs::embed::init_model;
use std::{env, io::stdin};
use tracing::debug;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

const TABLE_NAME: &str = "EmbeddingsTable";

// TODO: Fix by moving to chat/completions endpoint. I changed this to use the assitants endpoint via async_openai but that isn't supported by llamafile.
#[tokio::main]
async fn main() -> Result<()> {
    // Setup
    std::env::set_var("RUST_LOG", "ERROR");

    // Setup tracing subscriber so that library can log the errors
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    dotenv().ok();
    let model = init_model()?;

    let db_uri = env::var("DATABASE_PATH").expect("Environment var DATABASE_PATH must be set");
    let conn = lancedb::connect(&db_uri).execute().await?;
    let tbl = conn.open_table(TABLE_NAME).execute().await?;

    let local_conf = OpenAIConfig::new()
        .with_api_key("sk-no-key-required")
        .with_api_base("http://localhost:8080/v1");
    let client = Client::with_config(local_conf);

    let system = ChatCompletionRequestSystemMessageArgs::default()
                .content("Use the provided CONTEXT to answer questions. Documents in the CONTEXT are delimited with triple ~, i.e. `~~~`. If the answer cannot be found in the CONTEXT, write 'I could not find an answer.'")
                .build()?;

    let mut msg_thread: Vec<_> = vec![system.into()];

    loop {
        // Read user message from stdin
        println!(">> Awaiting your message");
        let mut query = String::new();
        let _ = stdin().read_line(&mut query);

        // Retrieve neighbors
        let nn_chunks = get_nearest_neighbor_chunks(&query, &model, &tbl).await?;
        let context = format!("```{}```", nn_chunks[..2].join("```"));

        let user_msg = ChatCompletionRequestUserMessageArgs::default()
            .content(format!(
                "QUESTION: {query}\n
            CONTEXT: {context}"
            ))
            .build()?;

        msg_thread.push(user_msg.into());

        let request = CreateChatCompletionRequestArgs::default()
            .n(1)
            .messages(msg_thread.clone())
            .build()
            .context("Failed to build ChatCompletionRequest")?;

        debug!("{}", serde_json::to_string(&request).unwrap());
        let response = &client
            .chat()
            .create(request)
            .await
            .context("Failed to create CompletionResponse")?;

        println!("\nResponse:\n");
        let response_text = response
            .choices
            .first()
            .as_ref()
            .unwrap()
            .message
            .content
            .as_ref()
            .unwrap()
            .clone();
        print!("{response_text}");
        print!("\n\n");
        let assistant_response = ChatCompletionRequestAssistantMessageArgs::default()
            .content(response_text)
            .build()?;
        msg_thread.push(assistant_response.into());
    }
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
