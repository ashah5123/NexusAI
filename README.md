# NexusAI

Local-first document search and text-to-speech, sized for a 16 GB Apple Silicon Mac.

Phase 17 provides a production-hardened local multimodal knowledge base with source-aware reading and organization: import PDFs, scanned
documents, images, audio, video, text, and Markdown; search page-aware or timestamped passages by
exact wording or semantic meaning; inspect ranked snippets with source citations; and read any
document or transcript aloud with local speech synthesis. Long-running OCR and transcription jobs
show progress, survive restarts, and can be cancelled or retried. Original sources open in the
reader at cited pages or timestamps, while collections, tags, favorites, and sorting keep the
library manageable. Reader notes and anchored highlights capture takeaways beside exact passages.
Library questions can be scoped by collection or source type, and model answers are shown only
when every cited evidence number can be verified against the retrieved passages. Smart views save
combined collection, tag, source, favorite, date, query, and sort filters. Collections can be renamed
or removed without deleting documents, and bulk organization updates selected sources together.
Grounded conversations persist locally, retain verified citations, and use recent questions to resolve
follow-ups. Conversation threads can be reopened, renamed, or deleted from the Ask workspace. Each
thread can be copied, printed, or downloaded as a Markdown research report or portable JSON record;
reports include per-answer evidence and a deduplicated source register.
Local accounts protect the API with expiring, HTTP-only sessions and PBKDF2 password hashes.
Workspaces isolate documents, imports, search, saved views, conversations, and semantic indexes;
owners can invite editors or read-only viewers with seven-day invitation tokens. The first account
created after upgrading becomes the owner of the existing local workspace and retains its data.
Explicit origin and host allowlists, CSRF checks, secure response headers, login throttling, request IDs,
structured request logs, health probes, Prometheus metrics, verified backups, and continuous integration
provide a safer operational baseline. A Caddy deployment terminates TLS and serves the frontend and API
from one origin. A PostgreSQL/pgvector schema and one-way migration utility are included for staged data
migration; the application runtime continues to use SQLite in this phase.

## Stack

- React 18 and Vite
- FastAPI and Pydantic
- SQLite with FTS5 full-text ranking
- pypdf for local PDF text extraction
- FastEmbed with quantized BGE-small embeddings for semantic retrieval
- RapidOCR and PyMuPDF for local image and scanned-PDF text recognition
- faster-whisper with CTranslate2 for local timestamped transcription
- macOS `say` for local speech audio when available
- Browser Web Speech API for text-to-speech
- SQLite-backed background ingestion worker
- Caddy, Nginx, and Docker Compose as optional deployment paths
- GitHub Actions for backend, frontend, and container verification

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
Register the first account to claim the existing local workspace.

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
be inspected with `docker compose --profile production-data up`, but the application does not depend on it.

## Production deployment

Create production settings, point the domain's DNS at the host, and start the hardened stack:

```bash
cp .env.production.example .env.production
# Replace every placeholder in .env.production before continuing.
docker compose --env-file .env.production -f compose.production.yml up -d --build
```

Caddy obtains and renews TLS certificates. The API runs as a non-root user with a read-only root
filesystem; persistent state is stored in the `nexusai-data` volume. Set an explicit domain,
metrics token, and strong database password. Do not use wildcard CORS origins. If TLS terminates at
a different proxy, preserve the original `Host` header and keep `NEXUSAI_SECURE_COOKIES=1`.

Operational endpoints are `GET /health/live`, `GET /health/ready`, and `GET /metrics`. Production
metrics require `Authorization: Bearer <NEXUSAI_METRICS_TOKEN>`. API logs are newline-delimited JSON
and every response includes an `X-Request-ID`.

## Backup and recovery

The backup command uses SQLite's online backup API and includes uploaded source files plus a
SHA-256 manifest. Verify every archive before moving or restoring it:

```bash
cd backend
.venv/bin/python -m app.maintenance backup --output ../backups/nexusai.tar.gz
.venv/bin/python -m app.maintenance verify ../backups/nexusai.tar.gz

# Stop the API before a restore. Existing data is retained with a before-restore timestamp.
.venv/bin/python -m app.maintenance restore ../backups/nexusai.tar.gz --force
```

For containers, run the same module in the `api` service and write the archive under `/app/data` so
it is stored in the persistent volume. Copy verified archives to separate encrypted storage and
periodically test restoration.

## PostgreSQL migration staging

Install the production requirements, create and verify a backup, initialize the pgvector schema,
then run the one-way copy utility:

```bash
cd backend
.venv/bin/pip install -r requirements-production.txt
.venv/bin/python -m app.migrate_postgres \
  --sqlite data/nexusai.db \
  --database-url 'postgresql://nexusai:password@localhost:5432/nexusai' \
  --force
```

`--force` confirms that the target tables may be truncated. Compare the printed per-table counts
with the source and test the target before changing any infrastructure. This prepares PostgreSQL
data but does not switch the API away from SQLite.

## Deployment settings

- `NEXUSAI_ENV`: use `production` outside local development.
- `NEXUSAI_CORS_ORIGINS`: comma-separated, explicit browser origins.
- `NEXUSAI_ALLOWED_HOSTS`: comma-separated HTTP host allowlist.
- `NEXUSAI_FORCE_HTTPS`: redirect direct HTTP requests when TLS terminates in the API.
- `NEXUSAI_SECURE_COOKIES`: require HTTPS for session cookies.
- `NEXUSAI_CSRF_HEADER`: custom mutation header; defaults to `X-NexusAI-CSRF`.
- `NEXUSAI_METRICS_TOKEN`: bearer token protecting production metrics.

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
keyword, semantic, and hybrid modes using Reciprocal Rank Fusion. Grounded answers use a local
Ollama adapter when available. Speech playback uses macOS `say` through the API when available and
falls back to the browser Web Speech API.

PDF, image, audio, and video uploads are copied to `backend/data/uploads` and processed by a single
local worker. Job state is stored in SQLite, so queued or interrupted imports resume when the API
restarts. The import activity panel reports each job's stage and supports cancellation and retry.
Completed source files belong to their documents, so import history can be cleared independently.
Deleting a document also removes its stored original file.
