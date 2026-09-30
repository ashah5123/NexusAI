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
        self.client.headers.update({"X-NexusAI-CSRF": "1"})
        register = self.client.post("/api/auth/register", json={
            "name": "Test Owner", "email": "owner@example.com", "password": "secure-test-password"
        })
        self.assertEqual(register.status_code, 201)

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

    def test_smart_view_and_bulk_organization_endpoints(self) -> None:
        first = self.repository.create(DocumentCreate(title="First", content="First source"))
        second = self.repository.create(DocumentCreate(title="Second", content="Second source"))
        bulk_response = self.client.patch("/api/documents", json={
            "document_ids": [first.id, second.id], "collection": "Research", "tags": ["Review"]
        })
        view_response = self.client.post("/api/saved-views", json={
            "name": "Research review", "collection": "Research", "tags": ["Review"]
        })
        list_response = self.client.get("/api/saved-views")
        rename_response = self.client.patch("/api/collections/Research", json={"name": "Evidence"})

        self.assertEqual(bulk_response.json(), {"updated": 2})
        self.assertEqual(view_response.status_code, 201)
        self.assertEqual(list_response.json()["total"], 1)
        self.assertEqual(rename_response.json(), {"updated": 2})
        self.assertEqual(self.repository.get(first.id).collection, "Evidence")

    def test_conversation_endpoints(self) -> None:
        create_response = self.client.post(
            "/api/conversations", json={"title": "Launch research"}
        )
        conversation_id = create_response.json()["id"]
        rename_response = self.client.patch(
            f"/api/conversations/{conversation_id}", json={"title": "Launch plan"}
        )
        detail_response = self.client.get(f"/api/conversations/{conversation_id}")
        list_response = self.client.get("/api/conversations")
        markdown_response = self.client.get(
            f"/api/conversations/{conversation_id}/report?format=markdown"
        )
        json_response = self.client.get(
            f"/api/conversations/{conversation_id}/report?format=json"
        )
        delete_response = self.client.delete(f"/api/conversations/{conversation_id}")

        self.assertEqual(create_response.status_code, 201)
        self.assertEqual(rename_response.json()["title"], "Launch plan")
        self.assertEqual(detail_response.json()["messages"], [])
        self.assertEqual(list_response.json()["total"], 1)
        self.assertIn("# Launch plan", markdown_response.text)
        self.assertIn("launch-plan.md", markdown_response.headers["content-disposition"])
        self.assertEqual(json_response.json()["format"], "nexusai-research-report-v1")
        self.assertEqual(delete_response.status_code, 204)

    def test_workspaces_isolate_documents(self) -> None:
        local_document = self.client.post(
            "/api/documents", json={"title": "Local source", "content": "Local evidence"}
        )
        created_workspace = self.client.post(
            "/api/auth/workspaces", json={"name": "Second workspace"}
        ).json()
        switched = self.client.post(
            "/api/auth/workspaces/switch", json={"workspace_id": created_workspace["id"]}
        )
        empty_list = self.client.get("/api/documents")
        second_document = self.client.post(
            "/api/documents", json={"title": "Second source", "content": "Separate evidence"}
        )
        isolated_search = self.client.get("/api/search?q=Local&mode=keyword")

        self.assertEqual(local_document.status_code, 201)
        self.assertEqual(switched.status_code, 200)
        self.assertEqual(empty_list.json()["total"], 0)
        self.assertEqual(second_document.status_code, 201)
        self.assertEqual(self.client.get("/api/documents").json()["total"], 1)
        self.assertEqual(isolated_search.json()["total"], 0)
        self.assertEqual(self.client.get(f"/api/documents/{local_document.json()['id']}").status_code, 404)

    def test_invited_viewer_has_read_only_access(self) -> None:
        invitation = self.client.post(
            "/api/auth/invitations", json={"email": "viewer@example.com", "role": "viewer"}
        )
        self.client.post("/api/auth/logout")
        registered = self.client.post("/api/auth/register", json={
            "name": "Test Viewer", "email": "viewer@example.com", "password": "viewer-test-password"
        })
        accepted = self.client.post(
            "/api/auth/invitations/accept", json={"token": invitation.json()["token"]}
        )
        denied = self.client.post(
            "/api/documents", json={"title": "Denied", "content": "Viewer cannot write"}
        )
        logout = self.client.post("/api/auth/logout")
        unauthenticated = self.client.get("/api/documents")

        self.assertEqual(registered.status_code, 201)
        self.assertEqual(accepted.json()["workspace"]["role"], "viewer")
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(logout.status_code, 204)
        self.assertEqual(unauthenticated.status_code, 401)

    def test_mutations_require_csrf_header_and_allowed_origin(self) -> None:
        missing_header = self.client.post(
            "/api/auth/logout", headers={"X-NexusAI-CSRF": ""}
        )
        untrusted_origin = self.client.post(
            "/api/auth/logout", headers={"Origin": "https://attacker.example"}
        )

        self.assertEqual(missing_header.status_code, 403)
        self.assertEqual(missing_header.json()["detail"], "CSRF header required")
        self.assertEqual(untrusted_origin.status_code, 403)
        self.assertEqual(untrusted_origin.json()["detail"], "Origin not allowed")

    def test_health_metrics_and_security_headers(self) -> None:
        response = self.client.get(
            "/health/ready", headers={"X-Request-ID": "health-check-1"}
        )
        metrics_response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(response.headers["x-request-id"], "health-check-1")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(metrics_response.status_code, 200)
        self.assertIn("nexusai_http_requests_total", metrics_response.text)

    def test_login_is_temporarily_locked_after_repeated_failures(self) -> None:
        self.client.post("/api/auth/logout")
        payload = {"email": "owner@example.com", "password": "incorrect-password"}

        responses = [self.client.post("/api/auth/login", json=payload) for _ in range(6)]

        self.assertTrue(all(response.status_code == 401 for response in responses[:5]))
        self.assertEqual(responses[-1].status_code, 429)
        self.assertEqual(responses[-1].headers["retry-after"], "900")


if __name__ == "__main__":
    unittest.main()
