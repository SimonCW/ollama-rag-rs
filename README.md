# Retrieval Augmented Generation with Ollama in Rust

## Useful Commands

Start Ollama in the background and log outputs.

```bash
nohup ollama serve > ~/Repos/rag-rs/logs/ollama_serve.log 2>&1 &
```

## Install protobuf for LanceDB

```bash
brew install protobuf
```

## License Notice

The Rust Book is licensed under MIT and Apache License 2.0. The Rust Book was
downloaded as pdf from here: https://doc.rust-lang.org/book/print.html and
converted to txt via:

```bash
# brew install poppler

pdftotext .data/2024-02-13_the_rust_book.pdf knowledge/2024-02-13_the_rust_book.txt
```
