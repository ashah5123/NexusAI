# NexusAI

Local-first document search and text-to-speech, sized for a 16 GB Apple Silicon Mac.

Phase 6 provides a local multimodal knowledge base with hybrid retrieval: import PDFs, scanned
documents, images, audio, video, text, and Markdown; search page-aware or timestamped passages by
exact wording or semantic meaning; inspect ranked snippets with source citations; and read any
document or transcript aloud with the browser's local speech engine.

## Stack

- React 18 and Vite
- FastAPI and Pydantic
- SQLite with FTS5 full-text ranking
- pypdf for local PDF text extraction
- FastEmbed with quantized BGE-small embeddings for semantic retrieval
- RapidOCR and PyMuPDF for local image and scanned-PDF text recognition
- faster-whisper with CTranslate2 for local timestamped transcription
- Browser Web Speech API for text-to-speech
- Docker Compose as an optional run path

The production target is recorded in `.cursor/rules/production-architecture.mdc`. Local adapters
are intentionally replaceable by PostgreSQL/pgvector, S3, Redis, and neural speech services when
production requirements demand it.

## Run locally

Prerequisites: Python 3.11 through 3.13 and Node.js 18 or newer.

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

## Grounded local answers

NexusAI can answer questions using only passages retrieved from your library. Install Ollama, then run:

```bash
ollama serve
ollama pull qwen3:1.7b
```

The **Ask AI** workspace shows the passages used for each answer. If Ollama is stopped, it falls back
to evidence-only results so retrieval remains useful.

The first audio or video upload downloads the multilingual Whisper `base` model into
`backend/data/models/whisper`. Transcription runs in an isolated CPU-int8 worker and releases model
memory when the upload finishes. This also keeps native media libraries isolated from the OCR
runtime on macOS.

## Run with Docker

After installing Docker Desktop:

```bash
docker compose up --build
```

The default profile starts only the API and frontend. The reserved PostgreSQL/pgvector service can
be inspected with `docker compose --profile production-data up`, but Phase 6 does not depend on it.

## Verify

```bash
cd backend
.venv/bin/python -m unittest discover -s tests

cd ../frontend
npm run build
```

Supported ingestion formats include `.pdf`, common images, `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`,
`.aac`, `.opus`, `.aif`, `.aiff`, `.mp4`, `.mov`, `.mkv`, `.webm`, `.m4v`, `.txt`, and `.md`.
Documents are split into overlapping passages; PDFs retain page numbers and transcripts retain
start/end timestamps. Pages with embedded text skip OCR, while image-only pages are rendered at 180
DPI and recognized locally. Document/image uploads are limited to 20 MB, media uploads to 100 MB,
images to 40 megapixels, scanned PDFs to 50 OCR pages, and recordings to two hours. Search supports
keyword, semantic, and hybrid modes using Reciprocal Rank Fusion. Grounded answers and neural TTS
are subsequent adapters.
