# RAG_PCS

Retrieval-Augmented Generation (RAG) application built to support document search and question answering over Programa Cidade Saneada documentation.

## Overview

This project was created to make technical and institutional documentation easier to access through AI-assisted retrieval and answer generation.

Instead of relying on manual search across files, the system processes documents, prepares them for retrieval, and uses an LLM-based pipeline to answer user questions with contextual grounding in the source material.

## Objectives

- Build a RAG workflow over domain-specific documentation
- Improve access to institutional knowledge
- Reduce the effort required to search large document collections
- Explore practical AI engineering patterns for document-based Q&A systems

## What the Project Does

- Ingests source documents
- Preprocesses and organizes the content
- Splits documents into retrievable chunks
- Retrieves relevant context for a user query
- Uses an LLM to generate answers grounded in retrieved information

## Core Components

- Document ingestion and preprocessing
- Chunking and text preparation
- Retrieval pipeline
- Prompt assembly with retrieved context
- LLM-based response generation
- Supporting scripts for updates and maintenance

## Use Case

The system is designed for users who need to query Programa Cidade Saneada documentation efficiently and obtain contextual answers without manually navigating multiple files.

This is particularly useful when the document base is large, technical, or frequently consulted.

## AI / Engineering Focus

This repository showcases applied AI engineering skills in:

- Retrieval-Augmented Generation
- document processing
- context construction
- prompt design
- knowledge access over unstructured data
- domain-oriented AI applications

## Why This Project Matters

Many AI portfolio projects stop at basic prompting. This project is more relevant for hiring because it focuses on a practical system design problem:

How do you make a body of documents searchable and useful through AI while preserving context and improving answer quality?

That makes it a stronger example of applied AI than a generic chatbot demo.

## Suggested High-Level Architecture

Documents -> Preprocessing -> Chunking -> Index / Retrieval -> Context Assembly -> LLM Response

## Recommended README Additions

If these components already exist in the codebase, add them explicitly:
- embedding model used
- vector database or retrieval method used
- chunking strategy
- prompt structure
- citation or grounding strategy
- evaluation approach
- limitations and failure cases

## Example Questions

Examples you can add after validating them against the project:
- What are the main objectives of the program?
- Which documents define the project scope?
- What are the operational requirements for a given process?
- Where is a specific policy or technical rule described?

## How to Run

Add the exact setup steps used in the repository, for example:
- install dependencies
- configure API keys or environment variables
- ingest documents
- build the retrieval index
- start the application or run the query interface

## Why Recruiters Should Care

This project demonstrates the transition from traditional data work into applied AI engineering by showing:
- document-centric AI system design
- retrieval pipeline thinking
- grounding over real source material
- practical use of LLMs beyond simple chat interfaces

## Future Improvements

- Add architecture diagram
- Add sample queries and outputs
- Add citation support in responses
- Add evaluation metrics for retrieval and answer quality
- Add deployment instructions
- Add tests for ingestion and retrieval steps
