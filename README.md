# NexusAI

Local-first document search and text-to-speech, sized for a 16 GB Apple Silicon Mac.

Phase 3 provides a local document knowledge base: import PDFs, text, and Markdown; search page-aware
passages with SQLite FTS5; inspect ranked snippets with source citations; and read any document aloud
with the browser's local speech engine. No account, API key, model download, or cloud service is
required.

## Stack

- React 18 and Vite
- FastAPI and Pydantic
- SQLite with FTS5 full-text ranking
- pypdf for local PDF text extraction
- Browser Web Speech API for text-to-speech
- Docker Compose as an optional run path

The production target is recorded in `.cursor/rules/production-architecture.mdc`. Local adapters
are intentionally replaceable by PostgreSQL/pgvector, S3, Redis, and neural speech services when
production requirements demand it.

## Run locally

Prerequisites: Python 3.11 or 3.12 and Node.js 18 or newer.

```bash
# Terminal 1
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 2
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. API documentation is available at
**http://localhost:8000/docs**. Local documents are stored in `backend/data/nexusai.db`.

## Run with Docker

After installing Docker Desktop:

```bash
docker compose up --build
```

The default profile starts only the API and frontend. The reserved PostgreSQL/pgvector service can
be inspected with `docker compose --profile production-data up`, but Phase 3 does not depend on it.

## Verify

```bash
cd backend
.venv/bin/python -m unittest discover -s tests

cd ../frontend
npm run build
```

Supported ingestion formats are text-based `.pdf`, `.txt`, `.md`, pasted text, and pasted
transcripts. Documents are split into overlapping passages, and PDF passages retain their page
numbers. Scanned-PDF OCR, semantic embeddings, speech-to-text, and neural TTS are subsequent
adapters.
