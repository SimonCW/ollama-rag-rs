# What?

A all-OSS, all-local, easy-install, privacy-first local RAG application, e.g.
"Chat with your filesystem"

---

# Why?

## Goals

- Local first! This should be easily installable and runnable on a laptop. I
  have a Mac so that's my primary target. We'll see about windows afterwards
- Evaluate Rust for Applications that are on the intersection of MacOS/Windows
  Software and AI
- Gain experience towards edge computing cases, e.g. can this run on a car or
  factory robot?
- Demonstrate capability to run privacy-first OSS models on own data

## Non-Goals

- Production-ready software. Although, I'll explore some Rust libraries for
  logging, testing, etc
- Fits-all RAG application. For each individual use case the problem domain
  requires additional filtering, search, ranking, etc
- Polished UI

---

# Value Proposition

- Inspire customers to think about their own use cases that would require such a
  local / onpremises approach
- Demonstrate our capabilitites
- Make a few steps towards application development

---

# Components

- **Ingestion**: Clean, chunk, and embed documents into vector db
- **LLM**: The local LLM
- **Orchestration Backend**: Chat logic, context of conversation, ...
- **CLI Interface**: For use in terminal
- **Desktop UI**

---

# Technology - LLM

## Ollama

- Nice for **experimenting** with different models locally
- Running a different model just requires: `ollama run llama2` and ollama will
  take care of everything else, see [model hub](https://ollama.com/library).
- Abstracts over prompt format (both, good and bad)
- Supports most accelerators (Nvidia, AMD, Apple), based on llama.cpp
- Has sdks for Python, JS, and unoffical one for Rust
- **NOT NICE** for production deployment. How would one of our customers run an
  app that requires ollama?

## Llamafile

- Model weights and binary for running it in **one file**
- Supports most accelerators (Nvidia, AMD, Apple), based on llama.cpp
- Nice for **production**
- **NOT SO NICE** for experimenting

---

# Technology - Vector DB

I want something that I can run locally without too much overhead. Hence,
Pinecone, and Qdrant are not an option.

## Postgres + pgvector extension

- Postgres is super robust and battle-tested
- "Hybrid search" through vanilla postgres
- Not really suited for long-term local processes (server-client architecture)

## LanceDB

- "The SQLite of Vector Databases"
- Unfortunately the Rust SDK is quite basic but is currently being refactored
  and will be the basis for all other sdks, see
  [gh issue](https://github.com/lancedb/lancedb/issues/1121)

---

# Technology - Preprocessing

- Nice ecosystem in Rust because many well-established Python libraries are just
  wrappers around a Rust core, e.g. huggingface/tokenizers
- fastembed-rs for creating embeddings locally (currently via CPU)
- text-splitter for semantic chunking
- tokenizers

---

# Technology - Rust

- Language <3. This is a topic for another talk.
- Ecosystem is amazing. Some libraries in the ML space might not be super
  mature, but the quality is much higher than in the Python ecosystem (less
  "what the fuck" moments)
- Testing, logging, ... <3

---

# Technology - UI

## Tauri

- Like Electron but more native and less heavy tbd

## egui

tbd

---

# Milestones - Where I'm at

- [x] Get acquainted with Ollama
- [x] Select and get acquainted with vector db
- [x] Write script to ingest chunks / docs into vector db
- [ ] Find interesting use case / data
- [x] Ingest data into vector db
- [ ] Switch from pgvector to LanceDB?
- [ ] Switch from Ollama to Llamafile?
- [ ] Build CLI chatbot
- [ ] Build UI
