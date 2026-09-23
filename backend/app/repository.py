from __future__ import annotations

import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from .models import (
    DocumentCreate,
    DocumentList,
    DocumentRead,
    EmbeddingStatus,
    IngestionJob,
    IngestionJobList,
    SearchHit,
    SearchResponse,
)

DB_PATH = Path(os.getenv("NEXUSAI_DB_PATH", "./data/nexusai.db"))
TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
CHUNK_WORDS = 180
CHUNK_OVERLAP = 30
SEMANTIC_MIN_SCORE = float(os.getenv("NEXUSAI_SEMANTIC_MIN_SCORE", "0.55"))


def split_chunks(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + CHUNK_WORDS])
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


class DocumentRepository:
    """SQLite repository with page-aware FTS5 passage retrieval."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_name TEXT,
                    status TEXT NOT NULL DEFAULT 'ready',
                    word_count INTEGER NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 1,
                    ocr_applied INTEGER NOT NULL DEFAULT 0,
                    duration_seconds REAL,
                    language TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    start_seconds REAL,
                    end_seconds REAL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    embedding_model TEXT,
                    UNIQUE(document_id, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    stage TEXT NOT NULL DEFAULT 'Waiting',
                    progress INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS ingestion_jobs_status_created
                ON ingestion_jobs(status, created_at);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    title, content, content='chunks', content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, title, content)
                    VALUES (new.id, new.title, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, title, content)
                    VALUES ('delete', old.id, old.title, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, title, content)
                    VALUES ('delete', old.id, old.title, old.content);
                    INSERT INTO chunks_fts(rowid, title, content)
                    VALUES (new.id, new.title, new.content);
                END;
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
            if "page_count" not in columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN page_count INTEGER NOT NULL DEFAULT 1"
                )
            if "ocr_applied" not in columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN ocr_applied INTEGER NOT NULL DEFAULT 0"
                )
            if "duration_seconds" not in columns:
                connection.execute("ALTER TABLE documents ADD COLUMN duration_seconds REAL")
            if "language" not in columns:
                connection.execute("ALTER TABLE documents ADD COLUMN language TEXT")
            chunk_columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
            if "embedding" not in chunk_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
            if "embedding_model" not in chunk_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN embedding_model TEXT")
            if "start_seconds" not in chunk_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN start_seconds REAL")
            if "end_seconds" not in chunk_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN end_seconds REAL")
            self._backfill_chunks(connection)
            connection.commit()

    def _backfill_chunks(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """SELECT d.id, d.title, d.content
            FROM documents d
            WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"""
        ).fetchall()
        for row in rows:
            self._insert_chunks(
                connection,
                row["id"],
                row["title"],
                [(None, None, None, row["content"])],
            )

    @staticmethod
    def _insert_chunks(
        connection: sqlite3.Connection,
        document_id: str,
        title: str,
        sections: list[tuple[int | None, float | None, float | None, str]],
    ) -> None:
        chunk_index = 0
        for page_number, start_seconds, end_seconds, text in sections:
            for content in split_chunks(text):
                connection.execute(
                    """INSERT INTO chunks
                    (document_id, chunk_index, page_number, start_seconds, end_seconds, title, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        document_id,
                        chunk_index,
                        page_number,
                        start_seconds,
                        end_seconds,
                        title,
                        content,
                    ),
                )
                chunk_index += 1

    def health(self) -> str:
        try:
            with closing(self.connect()) as connection:
                connection.execute("SELECT 1").fetchone()
            return "connected"
        except sqlite3.Error:
            return "unavailable"

    @staticmethod
    def _document(row: sqlite3.Row) -> DocumentRead:
        fields = DocumentRead.model_fields
        return DocumentRead(**{key: row[key] for key in fields})

    def create(
        self,
        payload: DocumentCreate,
        pages: list[tuple[int, str]] | None = None,
        ocr_applied: bool = False,
        page_count: int | None = None,
        timed_passages: list[tuple[float, float, str]] | None = None,
        duration_seconds: float | None = None,
        language: str | None = None,
        document_id: str | None = None,
    ) -> DocumentRead:
        now = datetime.now(UTC).isoformat()
        document_id = document_id or str(uuid4())
        resolved_page_count = page_count or (len(pages) if pages else 1)
        if timed_passages:
            sections = [(None, start, end, text) for start, end, text in timed_passages]
        elif pages:
            sections = [(page, None, None, text) for page, text in pages]
        else:
            sections = [(None, None, None, payload.content)]
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO documents
                (id, title, content, source_type, source_name, word_count, page_count, ocr_applied,
                 duration_seconds, language, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    payload.title,
                    payload.content,
                    payload.source_type,
                    payload.source_name,
                    len(TOKEN_RE.findall(payload.content)),
                    resolved_page_count,
                    ocr_applied,
                    duration_seconds,
                    language,
                    now,
                    now,
                ),
            )
            self._insert_chunks(connection, document_id, payload.title, sections)
            connection.commit()
        document = self.get(document_id)
        if document is None:
            raise RuntimeError("Document was not persisted")
        return document

    @staticmethod
    def _ingestion_job(row: sqlite3.Row) -> IngestionJob:
        fields = IngestionJob.model_fields
        return IngestionJob(**{key: row[key] for key in fields})

    def create_ingestion_job(
        self,
        job_id: str,
        title: str,
        source_type: str,
        source_name: str,
        input_path: Path,
    ) -> IngestionJob:
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO ingestion_jobs
                (id, title, source_type, source_name, input_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (job_id, title, source_type, source_name, str(input_path), now, now),
            )
            connection.commit()
        job = self.get_ingestion_job(job_id)
        if job is None:
            raise RuntimeError("Ingestion job was not persisted")
        return job

    def get_ingestion_job(self, job_id: str) -> IngestionJob | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._ingestion_job(row) if row else None

    def ingestion_job_input_path(self, job_id: str) -> Path | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT input_path FROM ingestion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return Path(row["input_path"]) if row else None

    def list_ingestion_jobs(self, limit: int = 20) -> IngestionJobList:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM ingestion_jobs").fetchone()[0]
        return IngestionJobList(items=[self._ingestion_job(row) for row in rows], total=total)

    def claim_next_ingestion_job(self) -> IngestionJob | None:
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT id FROM ingestion_jobs
                WHERE status = 'queued' ORDER BY created_at LIMIT 1"""
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            cursor = connection.execute(
                """UPDATE ingestion_jobs
                SET status = 'running', stage = 'Starting', progress = 5,
                    cancel_requested = 0, updated_at = ?
                WHERE id = ? AND status = 'queued'""",
                (now, row["id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (row["id"],)
            ).fetchone()
            connection.commit()
        return self._ingestion_job(claimed) if cursor.rowcount and claimed else None

    def update_ingestion_job(self, job_id: str, **changes) -> IngestionJob | None:
        allowed = {"status", "stage", "progress", "error", "document_id", "cancel_requested"}
        updates = {key: value for key, value in changes.items() if key in allowed}
        if not updates:
            return self.get_ingestion_job(job_id)
        updates["updated_at"] = datetime.now(UTC).isoformat()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with closing(self.connect()) as connection:
            connection.execute(
                f"UPDATE ingestion_jobs SET {assignments} WHERE id = ?",
                (*updates.values(), job_id),
            )
            connection.commit()
        return self.get_ingestion_job(job_id)

    def cancel_ingestion_job(self, job_id: str) -> IngestionJob | None:
        job = self.get_ingestion_job(job_id)
        if job is None:
            return None
        if job.status == "queued":
            return self.update_ingestion_job(
                job_id, status="cancelled", stage="Cancelled", cancel_requested=1
            )
        if job.status == "running":
            return self.update_ingestion_job(
                job_id, stage="Cancelling", cancel_requested=1
            )
        return job

    def retry_ingestion_job(self, job_id: str) -> IngestionJob | None:
        job = self.get_ingestion_job(job_id)
        if job is None:
            return None
        if job.status not in {"failed", "cancelled"}:
            return job
        return self.update_ingestion_job(
            job_id,
            status="queued",
            stage="Waiting",
            progress=0,
            error=None,
            document_id=None,
            cancel_requested=0,
        )

    def reset_interrupted_ingestion_jobs(self) -> int:
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            cancelled = connection.execute(
                """UPDATE ingestion_jobs
                SET status = 'cancelled', stage = 'Cancelled', progress = 0, updated_at = ?
                WHERE status = 'running' AND cancel_requested = 1""",
                (now,),
            ).rowcount
            cursor = connection.execute(
                """UPDATE ingestion_jobs
                SET status = 'queued', stage = 'Recovered after restart', progress = 0,
                    cancel_requested = 0, updated_at = ?
                WHERE status = 'running' AND cancel_requested = 0""",
                (now,),
            )
            connection.commit()
        return cursor.rowcount + cancelled

    def get(self, document_id: str) -> DocumentRead | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._document(row) if row else None

    def list(self, limit: int, offset: int) -> DocumentList:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return DocumentList(items=[self._document(row) for row in rows], total=total)

    def delete(self, document_id: str) -> bool:
        with closing(self.connect()) as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            connection.commit()
        return cursor.rowcount > 0

    def embedding_status(self, model: str, runtime_loaded: bool) -> EmbeddingStatus:
        with closing(self.connect()) as connection:
            total = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            indexed = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL AND embedding_model = ?",
                (model,),
            ).fetchone()[0]
        return EmbeddingStatus(
            model=model,
            total_chunks=total,
            indexed_chunks=indexed,
            pending_chunks=total - indexed,
            ready=total > 0 and indexed == total,
            loaded=runtime_loaded,
        )

    def chunks_pending_embedding(self, model: str) -> list[sqlite3.Row]:
        with closing(self.connect()) as connection:
            return connection.execute(
                """SELECT id, content FROM chunks
                WHERE embedding IS NULL OR embedding_model != ?
                ORDER BY id""",
                (model,),
            ).fetchall()

    def save_embeddings(
        self,
        embeddings: list[tuple[int, np.ndarray]],
        model: str,
    ) -> None:
        with closing(self.connect()) as connection:
            connection.executemany(
                "UPDATE chunks SET embedding = ?, embedding_model = ? WHERE id = ?",
                [(vector.astype(np.float32).tobytes(), model, chunk_id) for chunk_id, vector in embeddings],
            )
            connection.commit()

    def _keyword_candidates(self, fts_query: str, limit: int) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT d.*, c.id AS passage_id, c.chunk_index, c.page_number,
                    c.start_seconds, c.end_seconds,
                    c.content AS passage,
                    -bm25(chunks_fts, 7.0, 1.0) AS score,
                    snippet(chunks_fts, 1, '<mark>', '</mark>', ' ... ', 34) AS snippet
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def _semantic_candidates(
        self,
        query_vector: np.ndarray,
        model: str,
        limit: int,
    ) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT d.*, c.id AS passage_id, c.chunk_index, c.page_number,
                    c.start_seconds, c.end_seconds,
                    c.content AS passage, c.embedding
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL AND c.embedding_model = ?""",
                (model,),
            ).fetchall()

        query_norm = float(np.linalg.norm(query_vector)) or 1.0
        candidates: list[dict] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            vector_norm = float(np.linalg.norm(vector)) or 1.0
            item = dict(row)
            item.pop("embedding", None)
            item["score"] = float(np.dot(query_vector, vector) / (query_norm * vector_norm))
            passage = item["passage"]
            item["snippet"] = passage[:320] + (" ..." if len(passage) > 320 else "")
            if item["score"] >= SEMANTIC_MIN_SCORE:
                candidates.append(item)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    @staticmethod
    def _fuse(keyword: list[dict], semantic: list[dict], limit: int) -> list[dict]:
        fused: dict[int, dict] = {}
        scores: dict[int, float] = {}
        for results in (keyword, semantic):
            for rank, item in enumerate(results, 1):
                passage_id = item["passage_id"]
                scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (60 + rank)
                if passage_id not in fused or "<mark>" in item["snippet"]:
                    fused[passage_id] = item
        ordered = sorted(fused, key=lambda passage_id: scores[passage_id], reverse=True)[:limit]
        return [{**fused[passage_id], "score": scores[passage_id]} for passage_id in ordered]

    def retrieve(self, query: str, limit: int, embedder) -> tuple[list[dict], str | None]:
        terms = TOKEN_RE.findall(query)
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        if not fts_query:
            return [], None

        candidate_limit = max(limit * 3, 30)
        keyword = self._keyword_candidates(fts_query, candidate_limit)
        semantic: list[dict] = []
        warning = None
        status = self.embedding_status(embedder.model_name, embedder.loaded)
        if status.indexed_chunks:
            try:
                semantic = self._semantic_candidates(
                    embedder.embed_query(query), embedder.model_name, candidate_limit
                )
            except Exception:
                warning = "Semantic retrieval is unavailable; the answer uses keyword evidence."
        else:
            warning = "Semantic indexing is not enabled; the answer uses keyword evidence."

        rows = self._fuse(keyword, semantic, limit) if semantic else keyword[:limit]
        return rows, warning

    def search(self, query: str, limit: int, mode: str, embedder) -> SearchResponse:
        started = time.perf_counter()
        terms = TOKEN_RE.findall(query)
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        if not fts_query:
            return SearchResponse(query=query, items=[], total=0, elapsed_ms=0, mode=mode)

        candidate_limit = max(limit * 3, 30)
        keyword = self._keyword_candidates(fts_query, candidate_limit) if mode != "semantic" else []
        semantic: list[dict] = []
        warning = None
        if mode != "keyword":
            status = self.embedding_status(embedder.model_name, embedder.loaded)
            if status.indexed_chunks:
                try:
                    semantic = self._semantic_candidates(
                        embedder.embed_query(query), embedder.model_name, candidate_limit
                    )
                except Exception:
                    warning = "Semantic model is unavailable; showing keyword results."
            else:
                warning = "Enable semantic search to index your document passages."

        if mode == "keyword":
            rows = keyword[:limit]
        elif mode == "semantic":
            rows = semantic[:limit]
        else:
            rows = self._fuse(keyword, semantic, limit) if semantic else keyword[:limit]
        items = [SearchHit(**row) for row in rows]
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return SearchResponse(
            query=query,
            items=items,
            total=len(items),
            elapsed_ms=elapsed,
            mode=mode,
            warning=warning,
        )
