import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .models import DocumentCreate, DocumentList, DocumentRead, SearchHit, SearchResponse

DB_PATH = Path(os.getenv("NEXUSAI_DB_PATH", "./data/nexusai.db"))
TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
CHUNK_WORDS = 180
CHUNK_OVERLAP = 30


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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    UNIQUE(document_id, chunk_index)
                );

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
            self._backfill_chunks(connection)
            connection.commit()

    def _backfill_chunks(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """SELECT d.id, d.title, d.content
            FROM documents d
            WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"""
        ).fetchall()
        for row in rows:
            self._insert_chunks(connection, row["id"], row["title"], [(None, row["content"])])

    @staticmethod
    def _insert_chunks(
        connection: sqlite3.Connection,
        document_id: str,
        title: str,
        sections: list[tuple[int | None, str]],
    ) -> None:
        chunk_index = 0
        for page_number, text in sections:
            for content in split_chunks(text):
                connection.execute(
                    """INSERT INTO chunks
                    (document_id, chunk_index, page_number, title, content)
                    VALUES (?, ?, ?, ?, ?)""",
                    (document_id, chunk_index, page_number, title, content),
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
    ) -> DocumentRead:
        now = datetime.now(UTC).isoformat()
        document_id = str(uuid4())
        page_count = len(pages) if pages else 1
        sections: list[tuple[int | None, str]] = pages or [(None, payload.content)]
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO documents
                (id, title, content, source_type, source_name, word_count, page_count,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    payload.title,
                    payload.content,
                    payload.source_type,
                    payload.source_name,
                    len(TOKEN_RE.findall(payload.content)),
                    page_count,
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

    def search(self, query: str, limit: int) -> SearchResponse:
        started = time.perf_counter()
        terms = TOKEN_RE.findall(query)
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        if not fts_query:
            return SearchResponse(query=query, items=[], total=0, elapsed_ms=0)

        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT d.*, c.chunk_index, c.page_number,
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
        items = [SearchHit(**dict(row)) for row in rows]
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return SearchResponse(query=query, items=items, total=len(items), elapsed_ms=elapsed)
