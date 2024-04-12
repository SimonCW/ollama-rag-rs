use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client,
};
use dotenv::dotenv;
use fastembed::EmbeddingBase;
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use rag_rs::embed::init_model;
use std::env;

const TABLE_NAME: &str = "EmbeddingsTable";

#[tokio::main]
async fn main() -> Result<()> {
    // Setup
    dotenv().ok();
    let model = init_model()?;
    let db_uri = env::var("DATABASE_PATH").expect("Environment var DATABASE_PATH must be set");
    let conn = lancedb::connect(&db_uri).execute().await?;
    let tbl = conn.open_table(TABLE_NAME).execute().await?;

    // Query
    let query = "What's the 'interior mutability' about and how to achieve it?";
    println!("Query: {query} \n");
    let query_embedding = model.query_embed(query)?;

    // Setup;
    let nearest_neighbors = tbl
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
    let nearest_neighbors = &nearest_neighbors[0];
    let local_conf = OpenAIConfig::new()
        .with_api_key("sk-no-key-required")
        .with_api_base("http://127.0.0.1:8080");
    let client = Client::with_config(local_conf);

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(512u16)
        .model("LLaMA_CPP")
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Who won the world series in 2020?")
                .build()?
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("The Los Angeles Dodgers won the World Series in 2020.")
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Where was it played?")
                .build()?
                .into(),
        ])
        .build()
        .context("Failed to build ChatCompletionRequest")?;

    println!("{}", serde_json::to_string(&request).unwrap());

    let response = client.chat().create(request).await?;

    println!("\nResponse:\n");
    for choice in response.choices {
        println!(
            "{}: Role: {}  Content: {:?}",
            choice.index, choice.message.role, choice.message.content
        );
    }

    //TODO: Create prompt
    //TODO: Chat with the user
    Ok(())
}
