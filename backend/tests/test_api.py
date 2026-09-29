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

    def test_document_note_endpoints(self) -> None:
        document = self.repository.create(
            DocumentCreate(title="Research source", content="Evidence worth annotating.")
        )

        create_response = self.client.post(
            f"/api/documents/{document.id}/notes",
            json={"content": "Check this against the launch memo."},
        )
        note_id = create_response.json()["id"]
        update_response = self.client.patch(
            f"/api/documents/{document.id}/notes/{note_id}",
            json={"content": "Confirmed by the launch memo."},
        )
        list_response = self.client.get(f"/api/documents/{document.id}/notes")
        delete_response = self.client.delete(f"/api/documents/{document.id}/notes/{note_id}")

        self.assertEqual(create_response.status_code, 201)
        self.assertEqual(update_response.json()["content"], "Confirmed by the launch memo.")
        self.assertEqual(list_response.json()["total"], 1)
        self.assertEqual(delete_response.status_code, 204)

    def test_document_highlight_endpoints(self) -> None:
        content = "Evidence worth annotating."
        document = self.repository.create(DocumentCreate(title="Research source", content=content))
        create_response = self.client.post(
            f"/api/documents/{document.id}/highlights",
            json={"start_offset": 0, "end_offset": 8, "selected_text": "Evidence", "color": "blue", "annotation": "Important"},
        )
        highlight_id = create_response.json()["id"]
        update_response = self.client.patch(
            f"/api/documents/{document.id}/highlights/{highlight_id}",
            json={"color": "pink", "annotation": "Review this"},
        )
        list_response = self.client.get(f"/api/documents/{document.id}/highlights")
        delete_response = self.client.delete(
            f"/api/documents/{document.id}/highlights/{highlight_id}"
        )

        self.assertEqual(create_response.status_code, 201)
        self.assertEqual(update_response.json()["color"], "pink")
        self.assertEqual(list_response.json()["total"], 1)
        self.assertEqual(delete_response.status_code, 204)

    def test_highlight_rejects_stale_anchor(self) -> None:
        document = self.repository.create(DocumentCreate(title="Research", content="Evidence"))
        response = self.client.post(
            f"/api/documents/{document.id}/highlights",
            json={"start_offset": 0, "end_offset": 8, "selected_text": "Different"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
