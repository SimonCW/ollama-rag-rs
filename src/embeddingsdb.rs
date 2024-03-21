use anyhow::{anyhow, Context, Result};
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lancedb::{connection::Connection, Table};
use std::sync::Arc;

const TABLE_NAME: &str = "EmbeddingsTable";
const EMBEDDINGSIZE: i32 = 1024;
const URI: &str = ".data/embeddingsdb";

pub struct EmbeddingsDB {
    // TODO: Remove uri and schema?
    pub uri: String,
    pub conn: Connection,
    pub table: Table,
    pub schema: Arc<Schema>,
}

pub struct EmbeddingsDBConfig {
    pub uri: String,
    pub table_name: String,
    pub schema: Arc<Schema>,
}

impl Default for EmbeddingsDBConfig {
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
        EmbeddingsDBConfig {
            uri: URI.to_string(),
            table_name: TABLE_NAME.to_string(),
            schema,
        }
    }
}

impl EmbeddingsDB {
    pub async fn new(
        uri: String,
        table_name: String,
        schema: Arc<Schema>,
    ) -> lancedb::Result<Self> {
        let conn = lancedb::connect(&uri).execute().await?;
        let table = conn
            .create_empty_table(table_name, schema.clone())
            .execute()
            .await?;
        Ok(EmbeddingsDB {
            uri,
            conn,
            table,
            schema,
        })
    }
}
