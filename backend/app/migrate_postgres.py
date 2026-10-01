from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
from contextlib import closing
from pathlib import Path


TABLES = [
    "users", "workspaces", "workspace_members", "sessions", "login_attempts",
    "workspace_invitations", "documents", "chunks", "ingestion_jobs", "document_notes",
    "document_highlights", "saved_views", "conversations", "conversation_messages",
]
JSON_COLUMNS = {"tags_json", "source_types_json", "citations_json"}
BOOLEAN_COLUMNS = {"ocr_applied", "favorite", "cancel_requested", "generated", "grounded"}


def vector_text(value: bytes | None) -> str | None:
    if value is None:
        return None
    if len(value) % 4:
        raise ValueError("Invalid embedding blob length")
    count = len(value) // 4
    numbers = struct.unpack(f"<{count}f", value)
    return "[" + ",".join(f"{number:.8g}" for number in numbers) + "]"


def convert(column: str, value):
    if value is None:
        return None
    if column in JSON_COLUMNS:
        try:
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise RuntimeError("Install requirements-production.txt before migrating") from exc
        return Jsonb(json.loads(value))
    if column in BOOLEAN_COLUMNS:
        return bool(value)
    if column == "embedding":
        return vector_text(value)
    return value


def validate_migration(sqlite_path: Path, database_url: str) -> dict[str, dict[str, int]]:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install requirements-production.txt before validating") from exc
    comparison: dict[str, dict[str, int]] = {}
    with closing(sqlite3.connect(sqlite_path)) as source, psycopg.connect(database_url) as target:
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        for table in TABLES:
            exists = source.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            source_count = (
                source.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if exists else 0
            )
            target_count = target.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            comparison[table] = {"sqlite": source_count, "postgresql": target_count}
            if source_count != target_count:
                raise ValueError(
                    f"Migration count mismatch for {table}: {source_count} != {target_count}"
                )
    return comparison


def migrate(sqlite_path: Path, database_url: str, schema_path: Path, force: bool) -> dict[str, int]:
    if not force:
        raise ValueError("Migration requires --force after you create and verify a backup")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install requirements-production.txt before migrating") from exc
    source = sqlite3.connect(sqlite_path)
    source.row_factory = sqlite3.Row
    if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("SQLite integrity check failed")
    counts: dict[str, int] = {}
    with psycopg.connect(database_url) as target:
        target.execute(schema_path.read_text(encoding="utf-8"))
        target.execute("TRUNCATE " + ", ".join(reversed(TABLES)) + " CASCADE")
        for table in TABLES:
            source_exists = source.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not source_exists:
                counts[table] = 0
                continue
            source_columns = [row[1] for row in source.execute(f"PRAGMA table_info({table})")]
            target_columns = {
                row[0] for row in target.execute(
                    """SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                      AND is_generated = 'NEVER'""", (table,)
                ).fetchall()
            }
            columns = [column for column in source_columns if column in target_columns]
            rows = source.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()
            if rows:
                placeholders = ", ".join(
                    "%s::vector" if column == "embedding" else "%s"
                    for column in columns
                )
                statement = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                target.executemany(
                    statement,
                    [tuple(convert(column, row[column]) for column in columns) for row in rows],
                )
            counts[table] = len(rows)
            target_count = target.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if target_count != len(rows):
                raise ValueError(
                    f"Migration count mismatch for {table}: {len(rows)} != {target_count}"
                )
        target.execute(
            """SELECT setval(
                pg_get_serial_sequence('chunks', 'id'),
                COALESCE((SELECT MAX(id) FROM chunks), 1),
                EXISTS (SELECT 1 FROM chunks)
            )"""
        )
        target.commit()
    source.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate NexusAI SQLite data to PostgreSQL")
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument(
        "--database-url", default=os.getenv("NEXUSAI_DATABASE_URL"),
        help="Defaults to NEXUSAI_DATABASE_URL",
    )
    parser.add_argument(
        "--schema", type=Path,
        default=Path(__file__).resolve().parents[2] / "postgres/init/02-production-schema.sql",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Compare source and target table counts without changing either database",
    )
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url or NEXUSAI_DATABASE_URL is required")
    if args.validate_only:
        result = validate_migration(args.sqlite, args.database_url)
    else:
        result = migrate(args.sqlite, args.database_url, args.schema, args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
