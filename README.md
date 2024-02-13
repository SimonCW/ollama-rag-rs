# Retrieval Augmented Generation with Ollama in Rust

## Useful Commands

Start Ollama in the background and log outputs.

```bash
nohup ollama serve > ~/Repos/rag-rs/logs/ollama_serve.log 2>&1 &
```

## Postgresql with Pgvector

```bash
brew install postgresql
brew services start postgresql
psql postgres
```

```sql
CREATE ROLE username WITH LOGIN PASSWORD 'password';
ALTER ROLE username CREATEDB;
```

- Created DB and table via pgadmin4, TODO: do it via bash, init script, ...

```bash
brew install pgvector
```

```bash
brew services stop postgresql
```

## License Notice

The Rust Book is licensed under MIT and Apache License 2.0. The Rust Book
contents were downloaded via:

```bash
pandoc -s -r html https://doc.rust-lang.org/book/print.html -o rust_book.txt
```
