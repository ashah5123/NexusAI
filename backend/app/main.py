"""NexusAI API service — Phase 1: runnable foundation only.

No database connectivity, model loading, or ingestion logic exists yet. This service exposes a
single health-check endpoint so the local stack (frontend, API, Postgres+pgvector) can be
verified end to end before any real feature is added.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NexusAI API", version="0.1.0")

# Permissive CORS for local development only — the frontend (localhost:5173) and the API
# (localhost:8000) are different origins under Vite's dev server. Revisit before any non-local
# deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/readiness check. Does not touch the database in this phase."""
    return {"status": "ok", "service": "nexusai-api"}
