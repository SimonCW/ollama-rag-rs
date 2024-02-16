# Design Doc

This is a design doc to outline the potential design of the application. This is
not documentation of an existing system.

## People

Just me for now, Simon Weiß.

## Glossary

- RAG = Retrieval Augmented Generation
- LLM = Large Language Model
- HF = HuggingFace

## Overview Idea

Have a LLM-RAG-Demonstrator in Rust. This could be a nice starting point for a
Blog Article, Webinar, or even Project Template for future projects.

## Context

tbd

## Goals and Non-Goals

### Goals

- Showcase capability to build LLM-RAG application on custom data with local OSS
  model (flexibility to change models)
- Local first! This should be easily installable and runnable on a laptop. I
  have a Mac so that's my primary target. We'll see about windows afterwards
- Showcase usefulness of Rust to reduce runtime errors and high performance (And
  it's not that hard!)
- Gain experience with a specific vector database

### Non-Goals

- Fits-all use cases RAG application. Most RAG applications can be compared to
  search / recommendation problems: they need some use-case-specific tuning and
  logic
- Show that RAG applications are more than just LLM on top of vector search. For
  the problem domain additional filtering, search, ranking might be necessary
  ([paper with nice graphic](https://arxiv.org/abs/2312.10997v1),
  [hn comments](https://news.ycombinator.com/item?id=39000241&utm_source=pocket_saves))

## Milestones

- [x] Get acquainted with Ollama
- [x] Select and get acquainted with vector db
- [x] Write script to ingest chunks / docs into vector db
- [ ] Find interesting use case / data
- [x] Ingest data into vector db
- [ ] Build CLI chatbot
- [ ] Build UI

## Existing Solutions

tbd

## Proposed Solution

### Components

- Ingestion pipeline to chunk and embed documents into vector db.
- Vector db, probably pgvector
- LLM abstraction, i.e., Ollama to be able to use different LLMs without much
  hassle
- CLI Application / Backend in Rust that takes plain user prompt, gets relevant
  documents from vector db, talks to Ollama LLM and returns assistant answer
  (LLM Output)

## Alternative Solutions

tbd

## Testability, Monitoring, and Alerting

tbd

## Cross-Team impact

tbd

## Open Questions

### Which use-case for the Demonstrator?

### How to embed documents and prompt locally?

#### Model

- Sbert is probably a good way to start, see
  [this X post](https://x.com/cwolferesearch/status/1747689404062126246?s=20)
- But there are better models, see
  [hf leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- `multilingual-e5-large` seems to work well for German
  [reddit-1](https://www.reddit.com/r/LocalLLaMA/comments/18fsty1/comment/kcxj4bm/?utm_source=share&utm_medium=web2x&context=3),
  [reddit-2](https://www.reddit.com/r/LocalLLaMA/comments/17p18m9/rag_embeddings/)

#### Library

I'll go with `fastembed-rs` for now. See notes below.

**Python:**

- The best option seems to be the
  [sentence transformers library](https://www.sbert.net/index.html) which is
  built on top of HF transformers and pytorch
  - Supports fine-tuning embedding models which is quite important if there is
    specific jargon, acronyms, etc

**Rust:**

- I need a way to embed the prompt in the Rust application
- Easiest would be Ollama but it doesn't support good embedding models yet
  ([GH issue for enhancement](https://github.com/jmorganca/ollama/issues/327))
- Via ONNX: https://github.com/Anush008/fastembed-rs
- Via Candle: https://github.com/huggingface/text-embeddings-inference . This
  looks super nice and supports many models. However, it is meant to run as a
  separate service and doesn't have a client library in Rust. Also, its default
  mode is running it via the huggingface model hub and servers. However, this
  could also be super cool. If the server is efficient and maybe even supports
  fine-tuned embedding models, this could be a super general solution to deploy
  for many different projects. For local "on-my-laptop" solutions I'm hesitant
  of using sth. like this though ...

**Both via ONNX:**

- Probably quite a bit of work but it might be best to create embeddings in
  Python pipeline via the sentence transformers library and then export the
  model to ONNX and use that in the Rust App
- But then I'd have to distribute the ONNX runtime with the App .... Probably,
  for real deployment it would be best to go "full Rust" and even get rid of the
  Ollama dependency and just use Candle or Burn

### Which Vector DB?

#### Update 2024-02-16

To fulfill the "local first" goal, I maybe should build use faiss or annoy on
disk or run sqlite with the
[sqlite-vss extension](https://github.com/asg017/sqlite-vss?tab=readme-ov-file)
or check out marqo.

**lancedb**

- looks promising, could be a real option and probably the better alternative to
  sqlite-vss
- https://github.com/lancedb/lancedb

**Sqlite-vss:**

- The author gives an honest overview:
  https://observablehq.com/@asg017/introducing-sqlite-vss

**Marqo**

- Wants to be the all-in-one solution, not just the vectordb part
- Seems to be built on top of vespa, onnx, sbert, etc, stiching together the
  text splitting, embedding, etc work
- The docker container is 4.7 GB! https://hub.docker.com/r/marqoai/marqo/tags.
- Their whole communication seems quite suspicious
- There are quite a few open Issues regarding compatibility problems with MacOs,
  etc

#### My initial Feeling

- For smaller projects (<100 MM Vectors): PGVector.
- For projects with high customization needs (additional search capabilities):
  OpenSearch / ElasticSearch
- If you need low latency, high throughput: Specialized VectorDB: My favorite
  would be Qdrant (pinecone if fully managed) but not chroma (who builds a DB in
  Python?!)

#### Resources

- https://news.ycombinator.com/item?id=36943318
- https://ann-benchmarks.com/glove-100-angular_10_angular.html
- https://docs.google.com/spreadsheets/d/1oAeF4Q7ILxxfInGJ8vTsBck3-2U9VV8idDf3hJOozNw/edit?pli=1#gid=0
- https://www.sicara.fr/blog-technique/how-to-choose-your-vector-database-in-2023
- https://maven.com/blog/embeddings
