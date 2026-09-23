import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main
from app.embedding import EmbeddingService
from app.jobs import IngestionWorker
from app.models import DocumentCreate
from app.ocr import OCRService
from app.repository import DocumentRepository
from app.transcription import TranscriptionService


class DocumentApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = DocumentRepository(Path(self.temp_dir.name) / "api.db")
        self.worker = IngestionWorker(
            self.repository,
            OCRService(),
            TranscriptionService(),
            EmbeddingService(),
            Path(self.temp_dir.name) / "uploads",
        )
        self.original_repository = main.repository
        self.original_worker = main.ingestion_worker
        main.repository = self.repository
        main.ingestion_worker = self.worker
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        main.repository = self.original_repository
        main.ingestion_worker = self.original_worker
        self.temp_dir.cleanup()

    def test_source_metadata_and_delete_lifecycle(self) -> None:
        source = Path(self.temp_dir.name) / "source.png"
        source.write_bytes(b"stored-source")
        document = self.repository.create(
            DocumentCreate(
                title="Source preview",
                content="Extracted source content",
                source_type="image",
                source_name="source.png",
            ),
            source_path=source,
        )

        source_response = self.client.get(f"/api/documents/{document.id}/source")
        update_response = self.client.patch(
            f"/api/documents/{document.id}",
            json={"collection": "Research", "tags": ["source"], "favorite": True},
        )
        delete_response = self.client.delete(f"/api/documents/{document.id}")

        self.assertEqual(source_response.status_code, 200)
        self.assertEqual(source_response.content, b"stored-source")
        self.assertEqual(update_response.json()["collection"], "Research")
        self.assertEqual(update_response.json()["tags"], ["source"])
        self.assertTrue(update_response.json()["favorite"])
        self.assertEqual(delete_response.status_code, 204)
        self.assertFalse(source.exists())

    def test_cleanup_endpoint_removes_finished_job_files(self) -> None:
        source = Path(self.temp_dir.name) / "failed.png"
        source.write_bytes(b"failed-source")
        self.repository.create_ingestion_job(
            "failed-job", "Failed", "image", "failed.png", source
        )
        self.repository.update_ingestion_job("failed-job", status="failed")

        response = self.client.delete("/api/ingestion-jobs")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"removed": 1})
        self.assertFalse(source.exists())


if __name__ == "__main__":
    unittest.main()
