use anyhow::{anyhow, Context, Result};
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lancedb::{connection::Connection, Table};
use std::sync::Arc;

const TABLE_NAME: &str = "EmbeddingsTable";
const EMBEDDINGSIZE: i32 = 1024;
const URI: &str = ".data/embeddingsdb";

pub struct Client {
    pub uri: String,
    pub conn: Connection,
}

pub struct Config {
    pub uri: String,
    pub table_name: String,
    pub schema: Arc<Schema>,
}

impl Config {
    pub fn new(uri: String, table_name: String, schema: Arc<Schema>) -> Self {
        Config {
            uri,
            table_name,
            schema,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDINGSIZE,
                ),
                true,
            ),
        ]));
        Config {
            uri: URI.to_string(),
            table_name: TABLE_NAME.to_string(),
            schema,
        }
    }
}

impl Client {
    pub async fn new(uri: String) -> lancedb::Result<Self> {
        let conn = lancedb::connect(&uri).execute().await?;
        Ok(Client { uri, conn })
    }
    // pub async fn create_table(&self, name: &str, schema: Arc<Schema>) -> lancedb::Result<> {
    //     let table = conn
    //         .create_empty_table(table_name, schema.clone())
    //         .execute()
    //         .await?;
    // }
}
