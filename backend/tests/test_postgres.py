import os
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from app.auth import AuthService
from app.models import (
    ConversationCreate,
    DocumentCreate,
    RegisterRequest,
    SavedViewCreate,
    WorkspaceCreate,
)
from app.migrate_postgres import migrate, validate_migration
from app.repository import DocumentRepository


DATABASE_URL = os.getenv("NEXUSAI_TEST_POSTGRES_URL")
TABLES = [
    "conversation_messages", "conversations", "saved_views", "document_highlights",
    "document_notes", "ingestion_jobs", "chunks", "documents", "workspace_invitations",
    "login_attempts", "sessions", "workspace_members", "workspaces", "users",
]


class PostgresEmbedder:
    model_name = "postgres-test-model"
    loaded = True

    def embed_query(self, _: str) -> np.ndarray:
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1.0
        return vector


@unittest.skipUnless(DATABASE_URL, "NEXUSAI_TEST_POSTGRES_URL is not configured")
class PostgresRepositoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = DocumentRepository(database_url=DATABASE_URL)
        cls.repository.initialize()

    def setUp(self) -> None:
        with closing(self.repository.connect()) as connection:
            connection.execute(
                "TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"
            )
            connection.commit()
        self.repository.initialize()
        self.auth = AuthService(self.repository.database)

    def test_authentication_and_workspace_isolation(self) -> None:
        token, session = self.auth.register(RegisterRequest(
            name="Postgres Owner",
            email="owner@example.com",
            password="secure-postgres-password",
        ))
        local_document = self.repository.create(
            DocumentCreate(title="Local", content="Local workspace evidence")
        )
        second = self.auth.create_workspace(
            self.auth.context(token), WorkspaceCreate(name="Second workspace")
        )
        self.repository.create(
            DocumentCreate(title="Second", content="Separate workspace evidence"),
            workspace_id=second.id,
        )

        _, logged_in = self.auth.login("owner@example.com", "secure-postgres-password")

        self.assertEqual(self.repository.backend, "postgresql")
        self.assertEqual(session.workspace.role, "owner")
        self.assertEqual(logged_in.user.email, "owner@example.com")
        self.assertEqual(self.repository.list(10, 0).total, 1)
        self.assertEqual(self.repository.list(10, 0, second.id).total, 1)
        self.assertIsNone(self.repository.get(local_document.id, second.id))

    def test_full_text_and_pgvector_search(self) -> None:
        document = self.repository.create(DocumentCreate(
            title="Atlas plan",
            content="Maya owns the Atlas release and the October launch checklist.",
        ))
        embedder = PostgresEmbedder()
        keyword = self.repository.search("Atlas release", 10, "keyword", embedder)
        pending = self.repository.chunks_pending_embedding(embedder.model_name)
        self.repository.save_embeddings(
            [(row["id"], embedder.embed_query(row["content"])) for row in pending],
            embedder.model_name,
        )
        semantic = self.repository.search("Who owns Atlas?", 10, "semantic", embedder)

        self.assertEqual(keyword.items[0].id, document.id)
        self.assertIn("<mark>", keyword.items[0].snippet)
        self.assertEqual(semantic.items[0].id, document.id)
        self.assertGreater(semantic.items[0].score, 0.99)

    def test_jobs_saved_views_and_conversations(self) -> None:
        self.repository.create_ingestion_job(
            "postgres-job", "Source", "text", "source.txt", Path("/tmp/source.txt")
        )
        claimed = self.repository.claim_next_ingestion_job()
        view = self.repository.create_saved_view(SavedViewCreate(
            name="Postgres sources", tags=["Database"], favorite=True
        ))
        conversation = self.repository.create_conversation(
            ConversationCreate(title="Postgres research")
        )

        self.assertEqual(claimed.id, "postgres-job")
        self.assertEqual(claimed.status, "running")
        self.assertEqual(view.tags, ["Database"])
        self.assertTrue(view.favorite)
        self.assertEqual(
            self.repository.get_conversation(conversation.id).title,
            "Postgres research",
        )

    def test_sqlite_migration_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sqlite_path = Path(temporary) / "source.db"
            source = DocumentRepository(sqlite_path)
            source.initialize()
            auth = AuthService(source.database)
            auth.register(RegisterRequest(
                name="Migration Owner",
                email="migration@example.com",
                password="secure-migration-password",
            ))
            source.create(DocumentCreate(
                title="Migrated Atlas",
                content="Atlas migration evidence survives the database cutover.",
            ))
            source.create_saved_view(SavedViewCreate(
                name="Migration view", tags=["Cutover"]
            ))
            embedder = PostgresEmbedder()
            source.save_embeddings(
                [
                    (row["id"], embedder.embed_query(row["content"]))
                    for row in source.chunks_pending_embedding(embedder.model_name)
                ],
                embedder.model_name,
            )
            schema = Path(__file__).resolve().parents[2] / "postgres/init/02-production-schema.sql"

            counts = migrate(sqlite_path, DATABASE_URL, schema, force=True)
            comparison = validate_migration(sqlite_path, DATABASE_URL)

        migrated = DocumentRepository(database_url=DATABASE_URL)
        result = migrated.search("Atlas migration", 10, "hybrid", PostgresEmbedder())
        self.assertEqual(counts["documents"], 1)
        self.assertEqual(comparison["chunks"], {"sqlite": 1, "postgresql": 1})
        self.assertEqual(result.items[0].title, "Migrated Atlas")


if __name__ == "__main__":
    unittest.main()
