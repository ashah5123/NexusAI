# NexusAI

Local-first document search and text-to-speech, sized for a 16 GB Apple Silicon Mac.

Phase 5 provides a local multimodal knowledge base with hybrid retrieval: import PDFs, scanned
documents, images, text, and Markdown; search page-aware passages by exact wording or semantic
meaning; inspect ranked snippets with source citations; and read any document aloud with the
browser's local speech engine.

## Stack

- React 18 and Vite
- FastAPI and Pydantic
- SQLite with FTS5 full-text ranking
- pypdf for local PDF text extraction
- FastEmbed with quantized BGE-small embeddings for semantic retrieval
- RapidOCR and PyMuPDF for local image and scanned-PDF text recognition
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

Keyword search works immediately and offline. To enable semantic and hybrid retrieval, add at least
one document and select **Enable semantic search**. The first run downloads the approximately 67 MB
`BAAI/bge-small-en-v1.5` model into `backend/data/models`; subsequent runs use the local cache. The
model is loaded only after semantic search is enabled.

## Run with Docker

After installing Docker Desktop:

```bash
docker compose up --build
```

The default profile starts only the API and frontend. The reserved PostgreSQL/pgvector service can
be inspected with `docker compose --profile production-data up`, but Phase 5 does not depend on it.

## Verify

```bash
cd backend
.venv/bin/python -m unittest discover -s tests

cd ../frontend
npm run build
```

Supported ingestion formats are `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff`, `.bmp`,
`.txt`, `.md`, pasted text, and pasted transcripts. Documents are split into overlapping passages,
and PDF passages retain their page numbers. Pages with embedded text skip OCR; image-only pages are
rendered at 180 DPI and recognized locally. Uploads are limited to 20 MB, images to 40 megapixels,
and scanned PDFs to 50 OCR pages per upload. Search supports keyword, semantic, and hybrid modes;
hybrid results use Reciprocal Rank Fusion. Speech-to-text, grounded answers, and neural TTS are
subsequent adapters.
