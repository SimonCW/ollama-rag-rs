# Retrieval Augmented Generation with Ollama in Rust

## Useful Commands

Start Ollama in the background and log outputs.

```bash
nohup ollama serve > ~/Repos/rag-rs/logs/ollama_serve.log 2>&1 &
```

## Install Postgresql with Pgvector on MacOs

```bash
brew install postgresql
brew install pgadmin4
brew services start postgresql
psql postgres
```

```sql
CREATE ROLE username WITH LOGIN PASSWORD 'password';
ALTER ROLE username CREATEDB;
```

```bash
brew install pgvector
```

```bash
brew services stop postgresql
```

## License Notice

The Rust Book is licensed under MIT and Apache License 2.0. The Rust Book was
downloaded as pdf from here: https://doc.rust-lang.org/book/print.html and
converted to txt via:

```bash
# brew install poppler

pdftotext .data/2024-02-13_the_rust_book.pdf knowledge/2024-02-13_the_rust_book.txt
```
