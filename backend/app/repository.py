from __future__ import annotations

import os
import json
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
    AnswerResponse,
    BulkDocumentUpdate,
    BulkUpdateResult,
    Citation,
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationMessageRead,
    ConversationRead,
    ConversationUpdate,
    DocumentList,
    DocumentNoteCreate,
    DocumentNoteList,
    DocumentNoteRead,
    DocumentNoteUpdate,
    DocumentHighlightCreate,
    DocumentHighlightList,
    DocumentHighlightRead,
    DocumentHighlightUpdate,
    DocumentRead,
    DocumentUpdate,
    EmbeddingStatus,
    IngestionJob,
    IngestionJobList,
    SearchHit,
    SearchResponse,
    SavedViewCreate,
    SavedViewList,
    SavedViewRead,
    SavedViewUpdate,
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
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_by TEXT REFERENCES users(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workspace_members (
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('owner', 'editor', 'viewer')),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workspace_invitations (
                    token_hash TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    email TEXT NOT NULL COLLATE NOCASE,
                    role TEXT NOT NULL CHECK(role IN ('editor', 'viewer')),
                    expires_at TEXT NOT NULL,
                    created_by TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    accepted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL DEFAULT 'local',
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
                    collection_name TEXT,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    favorite INTEGER NOT NULL DEFAULT 0,
                    source_path TEXT,
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
                    workspace_id TEXT NOT NULL DEFAULT 'local',
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

                CREATE TABLE IF NOT EXISTS document_notes (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS document_notes_document_updated
                ON document_notes(document_id, updated_at);

                CREATE TABLE IF NOT EXISTS document_highlights (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    selected_text TEXT NOT NULL,
                    color TEXT NOT NULL,
                    annotation TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK(start_offset >= 0 AND end_offset > start_offset),
                    CHECK(color IN ('yellow', 'green', 'blue', 'pink'))
                );

                CREATE INDEX IF NOT EXISTS document_highlights_document_position
                ON document_highlights(document_id, start_offset, end_offset);

                CREATE TABLE IF NOT EXISTS saved_views (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL,
                    query TEXT NOT NULL DEFAULT '',
                    collection_name TEXT,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    source_types_json TEXT NOT NULL DEFAULT '[]',
                    favorite INTEGER NOT NULL DEFAULT 0,
                    date_range TEXT NOT NULL DEFAULT 'all',
                    sort_mode TEXT NOT NULL DEFAULT 'recent',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL DEFAULT 'local',
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    generated INTEGER NOT NULL DEFAULT 0,
                    grounded INTEGER NOT NULL DEFAULT 0,
                    scope_description TEXT NOT NULL DEFAULT 'All documents',
                    warning TEXT,
                    created_at TEXT NOT NULL,
                    CHECK(role IN ('user', 'assistant'))
                );

                CREATE INDEX IF NOT EXISTS conversation_messages_thread_created
                ON conversation_messages(conversation_id, created_at);

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
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """INSERT OR IGNORE INTO workspaces (id, name, created_by, created_at)
                VALUES ('local', 'Local workspace', NULL, ?)""",
                (now,),
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
            if "workspace_id" not in columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'local'"
                )
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
            if "collection_name" not in columns:
                connection.execute("ALTER TABLE documents ADD COLUMN collection_name TEXT")
            if "tags_json" not in columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "favorite" not in columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0"
                )
            if "source_path" not in columns:
                connection.execute("ALTER TABLE documents ADD COLUMN source_path TEXT")
            job_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(ingestion_jobs)")
            }
            if "workspace_id" not in job_columns:
                connection.execute(
                    "ALTER TABLE ingestion_jobs ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'local'"
                )
            view_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(saved_views)")
            }
            if "workspace_id" not in view_columns:
                connection.execute(
                    "ALTER TABLE saved_views ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'local'"
                )
            conversation_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(conversations)")
            }
            if "workspace_id" not in conversation_columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'local'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS documents_workspace_created ON documents(workspace_id, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ingestion_jobs_workspace_created ON ingestion_jobs(workspace_id, created_at)"
            )
            connection.execute(
                """UPDATE documents SET source_path = (
                    SELECT input_path FROM ingestion_jobs
                    WHERE ingestion_jobs.document_id = documents.id
                )
                WHERE source_path IS NULL AND EXISTS (
                    SELECT 1 FROM ingestion_jobs
                    WHERE ingestion_jobs.document_id = documents.id
                )"""
            )
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
        data = {
            key: row[key]
            for key in DocumentRead.model_fields
            if key not in {"collection", "tags", "favorite", "source_available"}
        }
        data.update(
            collection=row["collection_name"],
            tags=json.loads(row["tags_json"] or "[]"),
            favorite=bool(row["favorite"]),
            source_available=bool(row["source_path"]),
        )
        return DocumentRead(**data)

    @staticmethod
    def _search_hit(row: dict) -> SearchHit:
        row["collection"] = row.get("collection_name")
        row["tags"] = json.loads(row.get("tags_json") or "[]")
        row["favorite"] = bool(row.get("favorite"))
        row["source_available"] = bool(row.get("source_path"))
        return SearchHit(**row)

    @staticmethod
    def _note(row: sqlite3.Row) -> DocumentNoteRead:
        return DocumentNoteRead(**{key: row[key] for key in DocumentNoteRead.model_fields})

    @staticmethod
    def _highlight(row: sqlite3.Row) -> DocumentHighlightRead:
        return DocumentHighlightRead(
            **{key: row[key] for key in DocumentHighlightRead.model_fields}
        )

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
        source_path: Path | None = None,
        workspace_id: str = "local",
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
                (id, workspace_id, title, content, source_type, source_name, word_count, page_count, ocr_applied,
                 duration_seconds, language, source_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    workspace_id,
                    payload.title,
                    payload.content,
                    payload.source_type,
                    payload.source_name,
                    len(TOKEN_RE.findall(payload.content)),
                    resolved_page_count,
                    ocr_applied,
                    duration_seconds,
                    language,
                    str(source_path) if source_path else None,
                    now,
                    now,
                ),
            )
            self._insert_chunks(connection, document_id, payload.title, sections)
            connection.commit()
        document = self.get(document_id, workspace_id)
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
        workspace_id: str = "local",
    ) -> IngestionJob:
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO ingestion_jobs
                (id, workspace_id, title, source_type, source_name, input_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, workspace_id, title, source_type, source_name, str(input_path), now, now),
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

    def list_ingestion_jobs(self, limit: int = 20, workspace_id: str = "local") -> IngestionJobList:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
                (workspace_id, limit),
            ).fetchall()
            total = connection.execute(
                "SELECT COUNT(*) FROM ingestion_jobs WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()[0]
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

    def cleanup_ingestion_jobs(self, workspace_id: str = "local") -> tuple[int, list[Path]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT input_path, status FROM ingestion_jobs
                WHERE workspace_id = ? AND status IN ('completed', 'failed', 'cancelled')""",
                (workspace_id,),
            ).fetchall()
            cursor = connection.execute(
                """DELETE FROM ingestion_jobs
                WHERE workspace_id = ? AND status IN ('completed', 'failed', 'cancelled')""",
                (workspace_id,),
            )
            connection.commit()
        disposable = [Path(row["input_path"]) for row in rows if row["status"] != "completed"]
        return cursor.rowcount, disposable

    def get(self, document_id: str, workspace_id: str = "local") -> DocumentRead | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ? AND workspace_id = ?",
                (document_id, workspace_id),
            ).fetchone()
        return self._document(row) if row else None

    def source_path(self, document_id: str, workspace_id: str = "local") -> Path | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT source_path FROM documents WHERE id = ? AND workspace_id = ?",
                (document_id, workspace_id),
            ).fetchone()
        return Path(row["source_path"]) if row and row["source_path"] else None

    def update(self, document_id: str, payload: DocumentUpdate, workspace_id: str = "local") -> DocumentRead | None:
        current = self.get(document_id, workspace_id)
        if current is None:
            return None
        fields = payload.model_fields_set
        title = payload.title if "title" in fields else current.title
        collection = payload.collection if "collection" in fields else current.collection
        tags = payload.tags if "tags" in fields else current.tags
        favorite = payload.favorite if "favorite" in fields else current.favorite
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """UPDATE documents
                SET title = ?, collection_name = ?, tags_json = ?, favorite = ?, updated_at = ?
                WHERE id = ? AND workspace_id = ?""",
                (title, collection, json.dumps(tags), favorite, now, document_id, workspace_id),
            )
            if title != current.title:
                connection.execute(
                    "UPDATE chunks SET title = ? WHERE document_id = ?", (title, document_id)
                )
            connection.commit()
        return self.get(document_id, workspace_id)

    def bulk_update(self, payload: BulkDocumentUpdate, workspace_id: str = "local") -> BulkUpdateResult:
        fields = payload.model_fields_set
        assignments: list[str] = []
        params: list[object] = []
        if "collection" in fields:
            assignments.append("collection_name = ?")
            params.append(payload.collection)
        if "tags" in fields:
            assignments.append("tags_json = ?")
            params.append(json.dumps(payload.tags))
        if "favorite" in fields:
            assignments.append("favorite = ?")
            params.append(payload.favorite)
        if not assignments:
            return BulkUpdateResult(updated=0)
        assignments.append("updated_at = ?")
        params.append(datetime.now(UTC).isoformat())
        placeholders = ",".join("?" for _ in payload.document_ids)
        params.extend(payload.document_ids)
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                f"UPDATE documents SET {', '.join(assignments)} WHERE workspace_id = ? AND id IN ({placeholders})",
                [*params[:len(params) - len(payload.document_ids)], workspace_id, *payload.document_ids],
            )
            connection.commit()
        return BulkUpdateResult(updated=cursor.rowcount)

    def rename_collection(self, old_name: str, new_name: str, workspace_id: str = "local") -> int:
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "UPDATE documents SET collection_name = ?, updated_at = ? WHERE workspace_id = ? AND collection_name = ?",
                (new_name, datetime.now(UTC).isoformat(), workspace_id, old_name),
            )
            connection.commit()
        return cursor.rowcount

    def clear_collection(self, name: str, workspace_id: str = "local") -> int:
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "UPDATE documents SET collection_name = NULL, updated_at = ? WHERE workspace_id = ? AND collection_name = ?",
                (datetime.now(UTC).isoformat(), workspace_id, name),
            )
            connection.commit()
        return cursor.rowcount

    @staticmethod
    def _saved_view(row: sqlite3.Row) -> SavedViewRead:
        return SavedViewRead(
            id=row["id"], name=row["name"], query=row["query"],
            collection=row["collection_name"], tags=json.loads(row["tags_json"]),
            source_types=json.loads(row["source_types_json"]), favorite=bool(row["favorite"]),
            date_range=row["date_range"], sort=row["sort_mode"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def list_saved_views(self, workspace_id: str = "local") -> SavedViewList:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM saved_views WHERE workspace_id = ? ORDER BY name COLLATE NOCASE",
                (workspace_id,),
            ).fetchall()
        return SavedViewList(items=[self._saved_view(row) for row in rows], total=len(rows))

    def create_saved_view(self, payload: SavedViewCreate, workspace_id: str = "local") -> SavedViewRead:
        view_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO saved_views
                (id, workspace_id, name, query, collection_name, tags_json, source_types_json, favorite, date_range, sort_mode, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (view_id, workspace_id, payload.name, payload.query, payload.collection, json.dumps(payload.tags),
                 json.dumps(payload.source_types), payload.favorite, payload.date_range, payload.sort, now, now),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM saved_views WHERE id = ?", (view_id,)).fetchone()
        return self._saved_view(row)

    def update_saved_view(self, view_id: str, payload: SavedViewUpdate, workspace_id: str = "local") -> SavedViewRead | None:
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                """UPDATE saved_views SET name = ?, query = ?, collection_name = ?, tags_json = ?,
                source_types_json = ?, favorite = ?, date_range = ?, sort_mode = ?, updated_at = ?
                WHERE id = ? AND workspace_id = ?""",
                (payload.name, payload.query, payload.collection, json.dumps(payload.tags),
                 json.dumps(payload.source_types), payload.favorite, payload.date_range, payload.sort, now, view_id, workspace_id),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM saved_views WHERE id = ?", (view_id,)).fetchone()
        return self._saved_view(row) if cursor.rowcount and row else None

    def delete_saved_view(self, view_id: str, workspace_id: str = "local") -> bool:
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM saved_views WHERE id = ? AND workspace_id = ?", (view_id, workspace_id)
            )
            connection.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _conversation(row: sqlite3.Row) -> ConversationRead:
        return ConversationRead(
            id=row["id"], title=row["title"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _conversation_message(row: sqlite3.Row) -> ConversationMessageRead:
        return ConversationMessageRead(
            id=row["id"], conversation_id=row["conversation_id"], role=row["role"],
            content=row["content"],
            citations=[Citation(**item) for item in json.loads(row["citations_json"] or "[]")],
            generated=bool(row["generated"]), grounded=bool(row["grounded"]),
            scope_description=row["scope_description"], warning=row["warning"],
            created_at=row["created_at"],
        )

    def create_conversation(self, payload: ConversationCreate, workspace_id: str = "local") -> ConversationRead:
        conversation_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                "INSERT INTO conversations (id, workspace_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, workspace_id, payload.title, now, now),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return self._conversation(row)

    def get_conversation(self, conversation_id: str, workspace_id: str = "local") -> ConversationRead | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND workspace_id = ?",
                (conversation_id, workspace_id),
            ).fetchone()
        return self._conversation(row) if row else None

    def list_conversations(self, workspace_id: str = "local") -> ConversationList:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM conversations WHERE workspace_id = ? ORDER BY updated_at DESC",
                (workspace_id,),
            ).fetchall()
        return ConversationList(items=[self._conversation(row) for row in rows], total=len(rows))

    def get_conversation_detail(self, conversation_id: str, workspace_id: str = "local") -> ConversationDetail | None:
        conversation = self.get_conversation(conversation_id, workspace_id)
        if conversation is None:
            return None
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT * FROM conversation_messages WHERE conversation_id = ?
                ORDER BY created_at ASC, rowid ASC""",
                (conversation_id,),
            ).fetchall()
        return ConversationDetail(
            conversation=conversation,
            messages=[self._conversation_message(row) for row in rows],
        )

    def update_conversation(
        self, conversation_id: str, payload: ConversationUpdate, workspace_id: str = "local"
    ) -> ConversationRead | None:
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND workspace_id = ?",
                (payload.title, now, conversation_id, workspace_id),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return self._conversation(row) if cursor.rowcount and row else None

    def delete_conversation(self, conversation_id: str, workspace_id: str = "local") -> bool:
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ? AND workspace_id = ?",
                (conversation_id, workspace_id),
            )
            connection.commit()
        return cursor.rowcount > 0

    def recent_conversation_messages(self, conversation_id: str, limit: int = 8) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT role, content FROM (
                    SELECT rowid, role, content, created_at FROM conversation_messages
                    WHERE conversation_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?
                ) ORDER BY created_at ASC, rowid ASC""",
                (conversation_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def save_conversation_exchange(
        self, conversation_id: str, question: str, answer: AnswerResponse
    ) -> None:
        now = datetime.now(UTC).isoformat()
        citations = json.dumps([citation.model_dump(mode="json") for citation in answer.citations])
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO conversation_messages
                (id, conversation_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)""",
                (str(uuid4()), conversation_id, question, now),
            )
            connection.execute(
                """INSERT INTO conversation_messages
                (id, conversation_id, role, content, citations_json, generated, grounded,
                 scope_description, warning, created_at)
                VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid4()), conversation_id, answer.answer, citations, answer.generated,
                 answer.grounded, answer.scope_description, answer.warning, now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )
            connection.commit()

    def list(self, limit: int, offset: int, workspace_id: str = "local") -> DocumentList:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (workspace_id, limit, offset),
            ).fetchall()
            total = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()[0]
        return DocumentList(items=[self._document(row) for row in rows], total=total)

    def delete(self, document_id: str, workspace_id: str = "local") -> bool:
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE id = ? AND workspace_id = ?", (document_id, workspace_id)
            )
            connection.commit()
        return cursor.rowcount > 0

    def list_notes(self, document_id: str, workspace_id: str = "local") -> DocumentNoteList | None:
        if self.get(document_id, workspace_id) is None:
            return None
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT * FROM document_notes
                WHERE document_id = ?
                ORDER BY updated_at DESC, created_at DESC""",
                (document_id,),
            ).fetchall()
        return DocumentNoteList(items=[self._note(row) for row in rows], total=len(rows))

    def create_note(self, document_id: str, payload: DocumentNoteCreate, workspace_id: str = "local") -> DocumentNoteRead | None:
        if self.get(document_id, workspace_id) is None:
            return None
        note_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO document_notes
                (id, document_id, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)""",
                (note_id, document_id, payload.content, now, now),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM document_notes WHERE id = ?", (note_id,)
            ).fetchone()
        return self._note(row)

    def update_note(
        self,
        document_id: str,
        note_id: str,
        payload: DocumentNoteUpdate,
        workspace_id: str = "local",
    ) -> DocumentNoteRead | None:
        if self.get(document_id, workspace_id) is None:
            return None
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                """UPDATE document_notes
                SET content = ?, updated_at = ?
                WHERE id = ? AND document_id = ?""",
                (payload.content, now, note_id, document_id),
            )
            connection.commit()
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM document_notes WHERE id = ?", (note_id,)
            ).fetchone()
        return self._note(row)

    def delete_note(self, document_id: str, note_id: str, workspace_id: str = "local") -> bool:
        if self.get(document_id, workspace_id) is None:
            return False
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM document_notes WHERE id = ? AND document_id = ?",
                (note_id, document_id),
            )
            connection.commit()
        return cursor.rowcount > 0

    def list_highlights(self, document_id: str, workspace_id: str = "local") -> DocumentHighlightList | None:
        if self.get(document_id, workspace_id) is None:
            return None
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT * FROM document_highlights WHERE document_id = ?
                ORDER BY start_offset ASC, created_at ASC""",
                (document_id,),
            ).fetchall()
        return DocumentHighlightList(
            items=[self._highlight(row) for row in rows], total=len(rows)
        )

    def create_highlight(
        self, document_id: str, payload: DocumentHighlightCreate, workspace_id: str = "local"
    ) -> DocumentHighlightRead | None:
        document = self.get(document_id, workspace_id)
        if document is None:
            return None
        if payload.end_offset > len(document.content):
            raise ValueError("highlight offsets exceed document length")
        actual = document.content[payload.start_offset : payload.end_offset]
        if actual != payload.selected_text:
            raise ValueError("selected text does not match document offsets")
        highlight_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            overlap = connection.execute(
                """SELECT 1 FROM document_highlights
                WHERE document_id = ? AND start_offset < ? AND end_offset > ? LIMIT 1""",
                (document_id, payload.end_offset, payload.start_offset),
            ).fetchone()
            if overlap:
                raise ValueError("highlight overlaps an existing highlight")
            connection.execute(
                """INSERT INTO document_highlights
                (id, document_id, start_offset, end_offset, selected_text, color, annotation,
                 created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (highlight_id, document_id, payload.start_offset, payload.end_offset,
                 payload.selected_text, payload.color, payload.annotation, now, now),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM document_highlights WHERE id = ?", (highlight_id,)
            ).fetchone()
        return self._highlight(row)

    def update_highlight(
        self, document_id: str, highlight_id: str, payload: DocumentHighlightUpdate,
        workspace_id: str = "local",
    ) -> DocumentHighlightRead | None:
        if self.get(document_id, workspace_id) is None:
            return None
        with closing(self.connect()) as connection:
            existing = connection.execute(
                "SELECT * FROM document_highlights WHERE id = ? AND document_id = ?",
                (highlight_id, document_id),
            ).fetchone()
            if existing is None:
                return None
            color = payload.color or existing["color"]
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """UPDATE document_highlights SET color = ?, annotation = ?, updated_at = ?
                WHERE id = ? AND document_id = ?""",
                (color, payload.annotation, now, highlight_id, document_id),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM document_highlights WHERE id = ?", (highlight_id,)
            ).fetchone()
        return self._highlight(row)

    def delete_highlight(self, document_id: str, highlight_id: str, workspace_id: str = "local") -> bool:
        if self.get(document_id, workspace_id) is None:
            return False
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM document_highlights WHERE id = ? AND document_id = ?",
                (highlight_id, document_id),
            )
            connection.commit()
        return cursor.rowcount > 0

    def embedding_status(self, model: str, runtime_loaded: bool, workspace_id: str = "local") -> EmbeddingStatus:
        with closing(self.connect()) as connection:
            total = connection.execute(
                """SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE d.workspace_id = ?""", (workspace_id,)
            ).fetchone()[0]
            indexed = connection.execute(
                """SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE d.workspace_id = ? AND c.embedding IS NOT NULL AND c.embedding_model = ?""",
                (workspace_id, model),
            ).fetchone()[0]
        return EmbeddingStatus(
            model=model,
            total_chunks=total,
            indexed_chunks=indexed,
            pending_chunks=total - indexed,
            ready=total > 0 and indexed == total,
            loaded=runtime_loaded,
        )

    def chunks_pending_embedding(self, model: str, workspace_id: str = "local") -> list[sqlite3.Row]:
        with closing(self.connect()) as connection:
            return connection.execute(
                """SELECT c.id, c.content FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.workspace_id = ? AND (c.embedding IS NULL OR c.embedding_model != ?)
                ORDER BY c.id""",
                (workspace_id, model),
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

    @staticmethod
    def _scope_sql(
        workspace_id: str,
        document_ids: list[str] | None,
        collection: str | None,
        source_types: list[str] | None,
    ) -> tuple[str, list]:
        clauses: list[str] = ["d.workspace_id = ?"]
        params: list = [workspace_id]
        if document_ids:
            clauses.append(f"d.id IN ({','.join('?' for _ in document_ids)})")
            params.extend(document_ids)
        if collection:
            clauses.append("d.collection_name = ?")
            params.append(collection)
        if source_types:
            clauses.append(f"d.source_type IN ({','.join('?' for _ in source_types)})")
            params.extend(source_types)
        return (" AND " + " AND ".join(clauses) if clauses else ""), params

    def _keyword_candidates(
        self,
        fts_query: str,
        limit: int,
        document_ids: list[str] | None = None,
        collection: str | None = None,
        source_types: list[str] | None = None,
        workspace_id: str = "local",
    ) -> list[dict]:
        scope_sql, scope_params = self._scope_sql(workspace_id, document_ids, collection, source_types)
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT d.*, c.id AS passage_id, c.chunk_index, c.page_number,
                    c.start_seconds, c.end_seconds,
                    c.content AS passage,
                    -bm25(chunks_fts, 7.0, 1.0) AS score,
                    snippet(chunks_fts, 1, '<mark>', '</mark>', ' ... ', 34) AS snippet
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ?{scope_sql}
                ORDER BY score DESC
                LIMIT ?
                """,
                (fts_query, *scope_params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def _semantic_candidates(
        self,
        query_vector: np.ndarray,
        model: str,
        limit: int,
        document_ids: list[str] | None = None,
        collection: str | None = None,
        source_types: list[str] | None = None,
        workspace_id: str = "local",
    ) -> list[dict]:
        scope_sql, scope_params = self._scope_sql(workspace_id, document_ids, collection, source_types)
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""SELECT d.*, c.id AS passage_id, c.chunk_index, c.page_number,
                    c.start_seconds, c.end_seconds,
                    c.content AS passage, c.embedding
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL AND c.embedding_model = ?{scope_sql}""",
                (model, *scope_params),
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

    def retrieve(
        self,
        query: str,
        limit: int,
        embedder,
        document_ids: list[str] | None = None,
        collection: str | None = None,
        source_types: list[str] | None = None,
        workspace_id: str = "local",
    ) -> tuple[list[dict], str | None]:
        terms = TOKEN_RE.findall(query)
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        if not fts_query:
            return [], None

        candidate_limit = max(limit * 3, 30)
        keyword = self._keyword_candidates(
            fts_query, candidate_limit, document_ids, collection, source_types, workspace_id
        )
        semantic: list[dict] = []
        warning = None
        status = self.embedding_status(embedder.model_name, embedder.loaded, workspace_id)
        if status.indexed_chunks:
            try:
                semantic = self._semantic_candidates(
                    embedder.embed_query(query), embedder.model_name, candidate_limit,
                    document_ids, collection, source_types, workspace_id
                )
            except Exception:
                warning = "Semantic retrieval is unavailable; the answer uses keyword evidence."
        else:
            warning = "Semantic indexing is not enabled; the answer uses keyword evidence."

        rows = self._fuse(keyword, semantic, limit) if semantic else keyword[:limit]
        return rows, warning

    def search(self, query: str, limit: int, mode: str, embedder, workspace_id: str = "local") -> SearchResponse:
        started = time.perf_counter()
        terms = TOKEN_RE.findall(query)
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        if not fts_query:
            return SearchResponse(query=query, items=[], total=0, elapsed_ms=0, mode=mode)

        candidate_limit = max(limit * 3, 30)
        keyword = self._keyword_candidates(
            fts_query, candidate_limit, workspace_id=workspace_id
        ) if mode != "semantic" else []
        semantic: list[dict] = []
        warning = None
        if mode != "keyword":
            status = self.embedding_status(embedder.model_name, embedder.loaded, workspace_id)
            if status.indexed_chunks:
                try:
                    semantic = self._semantic_candidates(
                        embedder.embed_query(query), embedder.model_name, candidate_limit,
                        workspace_id=workspace_id,
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
        items = [self._search_hit(row) for row in rows]
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return SearchResponse(
            query=query,
            items=items,
            total=len(items),
            elapsed_ms=elapsed,
            mode=mode,
            warning=warning,
        )
