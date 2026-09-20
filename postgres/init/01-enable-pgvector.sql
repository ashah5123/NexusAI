-- Runs once, automatically, the first time the postgres container initializes its data
-- directory (standard Postgres Docker image behavior for files under /docker-entrypoint-initdb.d).
CREATE EXTENSION IF NOT EXISTS vector;
