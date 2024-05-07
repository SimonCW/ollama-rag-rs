use anyhow::{Context, Result};
use arrow_array::StringArray;
use async_openai::{
    config::OpenAIConfig,
    types::{
        CreateAssistantRequestArgs, CreateMessageRequestArgs, CreateRunRequestArgs,
        CreateThreadRequestArgs, MessageContent, MessageRole, RunStatus,
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

    let assistant_name = "Ferry";
    let instructions= "You are a knowledgable Rust developer that mentors and helps Rust learners. Use the provided CONTEXT to answer questions. Documents in the CONTEXT are delimted with triple backticks. If the answer cannot be found in the CONTEXT, write 'I could not find an answer.'";

    //create the assistant
    let assistant_request = CreateAssistantRequestArgs::default()
        .name(assistant_name)
        .instructions(instructions)
        .build()?;
    let assistant = client.assistants().create(assistant_request).await?;
    //get the id of the assistant
    let assistant_id = &assistant.id;

    loop {
        // Read user message from stdin
        println!("--- How can I help you?");

        let mut input = String::new();
        let _ = stdin().read_line(&mut input);
        //break out of the loop if the user enters exit()
        //
        if input.trim() == "exit()" {
            break;
        }

        // Retrieve neighbors
        let nn_chunks = get_nearest_neighbor_chunks(&input, &model, &tbl).await?;
        let context = nn_chunks[..2].join("\n<--->\n");

        let input = format!(
            "Use the provided CONTEXT to answer the QUESTION. 
            QUESTION: {input}\n\n
            CONTEXT: {context}"
        );
        //create a message for the thread
        let message = CreateMessageRequestArgs::default()
            .role(MessageRole::User)
            .content(input.clone())
            .build()?;

        //attach message to the thread
        let _message_obj = client
            .threads()
            .messages(&thread.id)
            .create(message)
            .await?;

        //create a run for the thread
        let run_request = CreateRunRequestArgs::default()
            .assistant_id(assistant_id)
            .build()?;
        let run = client
            .threads()
            .runs(&thread.id)
            .create(run_request)
            .await?;
        //wait for the run to complete
        let mut awaiting_response = true;
        while awaiting_response {
            //retrieve the run
            let run = client.threads().runs(&thread.id).retrieve(&run.id).await?;
            //check the status of the run
            match run.status {
                RunStatus::Completed => {
                    awaiting_response = false;
                    // once the run is completed we
                    // get the response from the run
                    // which will be the first message
                    // in the thread

                    //retrieve the response from the run
                    let response = client.threads().messages(&thread.id).list(&input).await?;
                    //get the message id from the response
                    let message_id = response.data.get(0).unwrap().id.clone();
                    //get the message from the response
                    let message = client
                        .threads()
                        .messages(&thread.id)
                        .retrieve(&message_id)
                        .await?;
                    //get the content from the message
                    let content = message.content.get(0).unwrap();
                    //get the text from the content
                    let text = match content {
                        MessageContent::Text(text) => text.text.value.clone(),
                        MessageContent::ImageFile(_) => {
                            panic!("imaged are not supported in the terminal")
                        }
                    };
                    //print the text
                    println!("--- Response: {}", text);
                    println!("");
                }
                RunStatus::Failed => {
                    awaiting_response = false;
                    println!("--- Run Failed: {:#?}", run);
                }
                RunStatus::Queued => {
                    println!("--- Run Queued");
                }
                RunStatus::Cancelling => {
                    println!("--- Run Cancelling");
                }
                RunStatus::Cancelled => {
                    println!("--- Run Cancelled");
                }
                RunStatus::Expired => {
                    println!("--- Run Expired");
                }
                RunStatus::RequiresAction => {
                    println!("--- Run Requires Action");
                }
                RunStatus::InProgress => {
                    println!("--- Waiting for response...");
                }
            }
            //wait for 1 second before checking the status again
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }
    //once we have broken from the main loop we can delete the assistant and thread
    client.assistants().delete(assistant_id).await?;
    client.threads().delete(&thread.id).await?;
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
