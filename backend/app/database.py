from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


NOCASE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s+COLLATE\s+NOCASE", re.IGNORECASE)


def _postgres_sql(statement: str) -> str:
    statement = NOCASE_RE.sub(r"LOWER(\1)", statement)
    statement = statement.replace("BEGIN IMMEDIATE", "BEGIN")
    statement = statement.replace("rowid", "id")
    statement = re.sub(r"cancel_requested\s*=\s*0", "cancel_requested = FALSE", statement)
    statement = re.sub(r"cancel_requested\s*=\s*1", "cancel_requested = TRUE", statement)
    return statement.replace("?", "%s")


class SQLiteDatabase:
    kind = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @staticmethod
    def is_error(exc: Exception) -> bool:
        return isinstance(exc, sqlite3.Error)

    @staticmethod
    def is_integrity_error(exc: Exception) -> bool:
        return isinstance(exc, sqlite3.IntegrityError)


class PostgresConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def execute(self, statement: str, params: Any = None):
        sql = _postgres_sql(statement)
        if params is None:
            return self._connection.execute(sql)
        return self._connection.execute(sql, params)

    def executemany(self, statement: str, params: Any):
        with self._connection.cursor() as cursor:
            cursor.executemany(_postgres_sql(statement), params)
            return cursor.rowcount

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


class PostgresDatabase:
    kind = "postgresql"
    path = None

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connect(self) -> PostgresConnection:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL requires backend/requirements-production.txt"
            ) from exc
        return PostgresConnection(psycopg.connect(self.database_url, row_factory=dict_row))

    @staticmethod
    def is_error(exc: Exception) -> bool:
        try:
            import psycopg
        except ImportError:
            return False
        return isinstance(exc, psycopg.Error)

    @staticmethod
    def is_integrity_error(exc: Exception) -> bool:
        try:
            import psycopg
        except ImportError:
            return False
        return isinstance(exc, psycopg.IntegrityError)
