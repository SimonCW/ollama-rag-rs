# Design Doc

This is a design doc to outline the potential design of the application. This is not documentation of an existing system.

## People

Just me for now, Simon Weiß.

## Glossary
* RAG = Retrieval Augmented Generation
* LLM = Large Language Model

## Overview Idea

Have a LLM-RAG-Demonstrator in Rust. This could be a nice starting point for a Blog Article, Webinar, or even Project Template for future projects.

## Context

tbd

## Goals and Non-Goals

### Goals

* Showcase capability to build LLM-RAG application on custom data with local OSS model (flexibility to change models, even OpenAI API)
* Show that RAG applications are more than just LLM on top of vector search. For the problem domain additional filtering, search, ranking might be necessary (https://news.ycombinator.com/item?id=39000241&utm_source=pocket_saves)
* Showcase usefulness of Rust to reduce runtime errors and high performance (And it's not that hard!)
* Gain experience with a specific vector database

### Non-Goals

* Fits-all use-cases RAG application. Most RAG applications can be compared to search / recommendation problems: they need some use-case-specific tuning and logic
* Everything in Rust. The ecosystem is lacking convenience libraries for many tasks

## Milestones

- [x] Get acquainted with Ollama
- [] Select and get acquainted with vector db
- [] Write script to ingest chunks / docs into vector db
- [] Find interesting use case / data
- [] Ingest data into vector db (Python Script)
- [] Build CLI chatbot

## Existing Solutions

tbd

## Proposed Solution

### Components

* Ingestion pipeline to chunk and embed documents into vector db.
* LLM abstraction, i.e., Ollama to be able to use different LLMs without much hassle
* CLI Application / Backend in Rust that takes plain user prompt, gets relevant documents from vector db, talks to Ollama LLM and returns assistant answer (LLM Output)

## Alternative Solutions

tbd

## Testability, Monitoring, and Alerting

tbd

## Cross-Team impact

tbd

## Open Questions






