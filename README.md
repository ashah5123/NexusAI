# NexusAI

Local-first multimodal search — **Phase 1: runnable foundation.**

This phase contains no search, ingestion, or model logic yet. It proves three things work
together on localhost: a React frontend, a FastAPI backend with a health check, and a
PostgreSQL database with the `pgvector` extension enabled. Later phases build on this.

Everything runs locally with no paid services, API keys, or GPU required.

## What's here

```
backend/            FastAPI service (GET /health only)
frontend/            React (Vite) home page that calls the API's /health endpoint
postgres/init/       SQL run once when the Postgres container first initializes (enables pgvector)
docker-compose.yml   Runs all three services together
.env.example         Copy to .env before running Compose
```

## Prerequisites

- Docker Desktop (or another Docker Engine + Compose v2) — for the one-command path below.
- Alternatively, to run without Docker: Python 3.11+ and Node.js 18+.

## Run everything with Docker Compose (recommended)

```bash
cp .env.example .env
docker compose up --build
```

Then open:

- Frontend: **http://localhost:5173**
- API health check: **http://localhost:8000/health**

Stop everything with `docker compose down` (add `-v` to also delete the Postgres data volume).

## Verify it's working

```bash
curl http://localhost:8000/health
# expected: {"status":"ok","service":"nexusai-api"}

docker compose exec postgres psql -U nexusai -d nexusai -c "\dx"
# expected: the "vector" extension listed in the output
```

Open http://localhost:5173 in a browser — the page should say **"API status: connected (ok)"**.
If it instead says "unreachable", the API container isn't up yet or isn't reachable; check
`docker compose logs api`.

## Running without Docker (backend + frontend only, no Postgres)

Useful for quick iteration on the API or UI. This does **not** start Postgres — the pages that
will eventually need the database don't exist yet in this phase, so this is enough to verify the
frontend/backend two work together.

```bash
# Terminal 1 — backend
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 (frontend) and http://localhost:8000/health (API), same as above.

## What was actually verified before this was committed

Run in this environment, without Docker (Docker was not available where this was built — see
below):

- `pip install -r backend/requirements.txt` succeeded.
- `uvicorn app.main:app` started and `curl http://127.0.0.1:8000/health` returned
  `200 {"status":"ok","service":"nexusai-api"}`.
- `npm install` and `npm run build` succeeded in `frontend/` with **0** `npm audit` vulnerabilities.
- `npm run dev` started and both `http://127.0.0.1:5173/` and `http://127.0.0.1:5173/src/main.jsx`
  returned `200`, with no errors in the Vite server log.
- `docker-compose.yml` was validated as syntactically correct YAML (`python3 -c "import yaml;
  yaml.safe_load(open('docker-compose.yml'))"`).

**Not verified** (no `docker`, `docker-compose`, or `podman` binary was available in the
environment this was built in): `docker compose up` was never actually run, so the Postgres
container, the `pgvector` extension actually being created, the containerized builds of the API
and frontend images, and the full three-service stack running together have not been confirmed
end to end. Run the "Docker Compose" and "Verify it's working" commands above on your machine to
confirm this — they are the exact commands to run, not a description of expected behavior only.
