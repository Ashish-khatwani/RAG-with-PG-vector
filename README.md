# Document Intelligence Engine

Production-grade Retrieval-Augmented Generation backend service built with FastAPI, PostgreSQL 17 + pgvector, local sentence-transformers embeddings, PostgreSQL hybrid retrieval, cross-encoder reranking, and a pluggable LLM provider configured for Mistral AI by default.

## Features

- Async FastAPI APIs with Swagger docs at `http://localhost:8000/docs`
- Batch ingestion for PDF, TXT, MD, and raw text payloads
- Cleaning, chunking, metadata enrichment, embeddings, and idempotent storage
- PostgreSQL `pgvector` similarity search plus full-text keyword search
- Reciprocal rank fusion hybrid retrieval
- Cross-encoder reranking for top chunks
- Grounded answer generation with source chunks and metadata
- Streaming query responses over Server-Sent Events for `curl -N`
- Structured JSON logging and health checks

## Project Layout

```text
app/
  api/
  core/
  db/
  models/
  prompts/
  services/
  utils/
requirements.txt
.env.example
README.md
```

## Prerequisites

- Python 3.11+
- PostgreSQL 17.x
- `pgvector` extension installed in PostgreSQL
- A Mistral AI API key for hosted answer generation

## PostgreSQL 17 + pgvector Setup

1. Install PostgreSQL 17 and ensure the server is running.
2. Install the `pgvector` extension package for your PostgreSQL distribution.
3. Create a database:

```sql
CREATE DATABASE rag_backend;
```

4. Connect to the database and enable the extension:

```sql
\c rag_backend
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

5. Update `.env` from `.env.example` with the correct database credentials.

## Mistral Setup

1. Create a Mistral AI API key in La Plateforme.
2. Put the key into `.env` as `MISTRAL_API_KEY`.
3. Choose a hosted model alias such as `mistral-small-latest`.

## Local Run Guide

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create your runtime config.

```powershell
Copy-Item .env.example .env
```

4. Start the API.

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5. Open Swagger if you want interactive inspection:

`http://localhost:8000/docs`

## API Usage With curl

### Health

```bash
curl http://localhost:8000/health
```

### Ingest files

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@./sample.pdf" \
  -F "files=@./notes.md"
```

### Ingest raw text

```bash
curl -X POST http://localhost:8000/ingest \
  -F "raw_texts=PostgreSQL with pgvector supports vector similarity search." \
  -F "raw_texts=Reranking improves retrieval precision."
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"How does hybrid retrieval work?\",\"filters\":{\"file_type\":\"md\"}}"
```

### Streaming query

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Summarize the PDF\",\"stream\":true}"
```

### Delete a document

```bash
curl -X DELETE http://localhost:8000/document \
  -H "Content-Type: application/json" \
  -d "{\"file_name\":\"notes.md\"}"
```

### Update or re-ingest a document from a file

```bash
curl -X PUT http://localhost:8000/document \
  -F "file_name=notes.md" \
  -F "file=@./notes.md"
```

### Update or re-ingest from raw text

```bash
curl -X PUT http://localhost:8000/document \
  -F "file_name=notes.md" \
  -F "raw_text=Updated document content for re-indexing."
```

## Notes

- Ingestion is idempotent for the same file name and content hash.
- Metadata stored per chunk includes `file_name`, `file_type`, `upload_timestamp`, and `chunk_index`.
- Streaming responses are sent as SSE events: `sources`, repeated `token`, then `done`.
- The default LLM provider is Mistral AI, configured through `.env`.
