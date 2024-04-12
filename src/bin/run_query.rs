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

    let local_conf = OpenAIConfig::new()
        .with_api_key("sk-no-key-required")
        .with_api_base("http://localhost:8080/v1");
    let client = Client::with_config(local_conf);

    // Query
    let query = "What's the 'interior mutability' about and how to achieve it?";
    println!("Query: {query} \n");
    // Retrieve neighbors
    let query_embedding = model.query_embed(query)?;
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
    let nn = &nearest_neighbors[0];
    let nn_chunks = nn
        .column_by_name("text")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    println!("___________");
    println!("{:#?}", nn_chunks);

    // Chat
    let system = ChatCompletionRequestSystemMessageArgs::default()
                .content("Use the provided CONTEXT to answer questions. If the answer cannot be found in the CONTEXT, write 'I could not find an answer.'")
                .build()?;
    let context = "hello";
    let user_msg = ChatCompletionRequestUserMessageArgs::default()
        .content(format!(
            "Use the provided CONTEXT to answer the QUESTION. 
            QUESTION: {query}
            CONTEXT: {context}"
        ))
        .build()?;

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(512u16)
        .messages([system.into(), user_msg.into()])
        .build()
        .context("Failed to build ChatCompletionRequest")?;

    println!("{}", serde_json::to_string(&request).unwrap());

    let response = client
        .chat()
        .create(request)
        .await
        .context("Failed to create CompletionResponse")?;

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
