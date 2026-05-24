# Advanced Production RAG

This is a highly-performant, modular Advanced Retrieval-Augmented Generation (RAG) pipeline built entirely from scratch in Python, without relying on wrapper frameworks like LangChain or LlamaIndex.

## Features
- **Semantic Chunking**: Dynamically splits documents into chunks based on semantic cosine distance percentiles using NLTK and sentence-transformers.
- **Dual-Path Hybrid Retrieval**: 
  - **Dense**: Local vector storage using ChromaDB.
  - **Sparse**: Custom from-scratch BM25Okapi implementation with advanced Regex tokenization.
- **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense search results.
- **Cross-Encoder Re-ranking**: Uses a lightweight local cross-encoder model to determine the absolute highest-scoring contexts.
- **FastAPI Layer**: Exposes the system through clean, strongly-typed REST API endpoints.

## 🚀 Getting Started

### 1. Installation

First, ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the Server

Run the following command to start the FastAPI server. 

*Note: The first time you run this, it will take a few moments to download the lightweight AI models (under 100MB each). Wait until you see `INFO: Application startup complete.` before proceeding to step 3.*

```bash
python -m uvicorn src.main:app --reload
```

### 3. Usage

Leave the server running in your terminal. Open a **new, separate terminal window** (Command Prompt or PowerShell) to interact with the API.

#### Ingesting Data

Add a document to the knowledge base:

```cmd
curl.exe -X POST "http://127.0.0.1:8000/api/v1/ingest" -H "Content-Type: application/json" -d "{\"text\": \"Artificial intelligence is a fascinating field. It relies heavily on advanced RAG systems. RAG stands for Retrieval-Augmented Generation, which is what we are building right now!\", \"source\": \"test_document.txt\"}"
```

#### Querying the RAG

Ask a question against your indexed documents:

```cmd
curl.exe -X POST "http://127.0.0.1:8000/api/v1/query" -H "Content-Type: application/json" -d "{\"user_query\": \"What does RAG stand for?\", \"top_k\": 4}"
```
