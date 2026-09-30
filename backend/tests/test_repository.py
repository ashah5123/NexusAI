import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from app.models import (
    AnswerResponse,
    BulkDocumentUpdate,
    Citation,
    ConversationCreate,
    ConversationUpdate,
    DocumentCreate,
    DocumentHighlightCreate,
    DocumentHighlightUpdate,
    DocumentNoteCreate,
    DocumentNoteUpdate,
    DocumentUpdate,
    SavedViewCreate,
    SavedViewUpdate,
)
import app.maintenance as maintenance
from app.embedding import EmbeddingService
from app.generation import OllamaAnswerService
from app.ingestion import IngestionError, extract_image, extract_pdf
from app.jobs import IngestionWorker
from app.ocr import OCRService
from app.repository import CHUNK_OVERLAP, CHUNK_WORDS, DocumentRepository, split_chunks
from app.reports import build_json_report, build_markdown_report, report_filename
from app.speech import LocalSpeechService, SpeechError
from app.transcription import TranscriptionError, TranscriptionService, group_segments


class FakeEmbedder:
    model_name = "test-model"
    loaded = True

    def embed_query(self, _: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeOCREngine:
    def __call__(self, _):
        return SimpleNamespace(txts=("Scanned contract", "Total due 2026"))


def image_bytes(text: str = "Scanned contract") -> bytes:
    image = Image.new("RGB", (700, 160), "white")
    ImageDraw.Draw(image).text((30, 50), text, fill="black", font_size=36)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class DocumentRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = DocumentRepository(Path(self.temp_dir.name) / "test.db")
        self.repository.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_create_search_and_delete_document(self) -> None:
        document = self.repository.create(
            DocumentCreate(title="Vector search", content="Hybrid retrieval combines lexical and semantic ranking.")
        )

        results = self.repository.search("semantic", 10, "keyword", FakeEmbedder())

        self.assertEqual(results.total, 1)
        self.assertEqual(results.items[0].id, document.id)
        self.assertIn("<mark>semantic</mark>", results.items[0].snippet)
        self.assertIsNone(results.items[0].page_number)
        self.assertFalse(document.ocr_applied)
        self.assertTrue(self.repository.delete(document.id))
        self.assertEqual(self.repository.list(10, 0).total, 0)

    def test_backup_verify_and_restore_round_trip(self) -> None:
        self.repository.create(
            DocumentCreate(title="Recovery source", content="Durable recovery evidence")
        )
        upload_dir = Path(self.temp_dir.name) / "uploads"
        upload_dir.mkdir()
        (upload_dir / "source.txt").write_text("stored source", encoding="utf-8")
        archive = Path(self.temp_dir.name) / "backup.tar.gz"
        restored_db = Path(self.temp_dir.name) / "restored" / "nexusai.db"
        restored_uploads = Path(self.temp_dir.name) / "restored-uploads"

        with patch.object(maintenance, "DB_PATH", self.repository.path), patch.object(
            maintenance, "UPLOAD_DIR", upload_dir
        ):
            maintenance.create_backup(archive)
            manifest = maintenance.verify_backup(archive)
        with patch.object(maintenance, "DB_PATH", restored_db), patch.object(
            maintenance, "UPLOAD_DIR", restored_uploads
        ):
            maintenance.restore_backup(archive, force=True)

        restored = DocumentRepository(restored_db)
        self.assertEqual(manifest["format"], "nexusai-backup-v1")
        self.assertEqual(restored.list(10, 0).total, 1)
        self.assertEqual((restored_uploads / "source.txt").read_text(), "stored source")

    def test_retrieval_keeps_full_passage_for_grounded_answers(self) -> None:
        self.repository.create(
            DocumentCreate(title="Atlas notes", content="Maya owns the Atlas release on October 14.")
        )

        passages, warning = self.repository.retrieve("Atlas release", 6, FakeEmbedder())

        self.assertEqual(passages[0]["passage"], "Maya owns the Atlas release on October 14.")
        self.assertIn("keyword evidence", warning)

    def test_retrieval_respects_collection_and_source_scope(self) -> None:
        research = self.repository.create(
            DocumentCreate(title="Research", content="Atlas launch evidence", source_type="pdf")
        )
        self.repository.update(research.id, DocumentUpdate(collection="Work"))
        personal = self.repository.create(
            DocumentCreate(title="Personal", content="Atlas garden evidence", source_type="text")
        )
        self.repository.update(personal.id, DocumentUpdate(collection="Home"))

        passages, _ = self.repository.retrieve(
            "Atlas evidence", 6, FakeEmbedder(), collection="Work", source_types=["pdf"]
        )

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["id"], research.id)

    def test_answer_service_falls_back_to_retrieved_evidence(self) -> None:
        service = OllamaAnswerService()
        passages = [{
            "id": "doc-1", "title": "Atlas notes", "source_type": "text",
            "page_number": None, "start_seconds": None, "end_seconds": None,
            "passage": "Maya owns the Atlas release.",
        }]

        with patch("app.generation.urlopen", side_effect=URLError("offline")):
            result = service.answer("Who owns Atlas?", passages)

        self.assertFalse(result.generated)
        self.assertEqual(result.citations[0].document_id, "doc-1")
        self.assertIn("unavailable", result.answer)

    def test_answer_service_withholds_invalid_model_citations(self) -> None:
        service = OllamaAnswerService()
        passages = [{
            "id": "doc-1", "title": "Atlas notes", "source_type": "text",
            "page_number": None, "start_seconds": None, "end_seconds": None,
            "passage": "Maya owns the Atlas release.", "content": "Maya owns the Atlas release.",
            "chunk_index": 0, "score": 1.0,
        }]
        response = BytesIO(b'{"message":{"content":"Maya owns it [9]."}}')

        with patch("app.generation.urlopen", return_value=response):
            result = service.answer("Who owns Atlas?", passages)

        self.assertFalse(result.grounded)
        self.assertIn("withheld", result.answer)
        self.assertEqual(result.citations[0].start_offset, 0)

    def test_page_aware_chunks_preserve_pdf_citation(self) -> None:
        payload = DocumentCreate(
            title="Research paper",
            content="First page introduction. Second page contains the decisive evidence.",
            source_type="pdf",
            source_name="paper.pdf",
        )
        document = self.repository.create(
            payload,
            pages=[(1, "First page introduction."), (2, "Second page contains the decisive evidence.")],
        )

        results = self.repository.search("decisive evidence", 10, "keyword", FakeEmbedder())

        self.assertEqual(document.page_count, 2)
        self.assertEqual(results.items[0].page_number, 2)
        self.assertEqual(results.items[0].source_name, "paper.pdf")

    def test_document_metadata_and_source_are_persisted(self) -> None:
        source_path = Path(self.temp_dir.name) / "paper.pdf"
        source_path.write_bytes(b"%PDF-source")
        document = self.repository.create(
            DocumentCreate(title="Draft paper", content="A searchable source document."),
            source_path=source_path,
        )

        updated = self.repository.update(
            document.id,
            DocumentUpdate(
                title="Published paper",
                collection="Research",
                tags=["Reference", "reference", "2026"],
                favorite=True,
            ),
        )

        self.assertEqual(updated.title, "Published paper")
        self.assertEqual(updated.collection, "Research")
        self.assertEqual(updated.tags, ["Reference", "2026"])
        self.assertTrue(updated.favorite)
        self.assertTrue(updated.source_available)
        self.assertEqual(self.repository.source_path(document.id), source_path)
        self.assertEqual(
            self.repository.search("Published", 10, "keyword", FakeEmbedder()).items[0].id,
            document.id,
        )

    def test_bulk_organization_and_collection_management(self) -> None:
        first = self.repository.create(DocumentCreate(title="First", content="First source"))
        second = self.repository.create(DocumentCreate(title="Second", content="Second source"))

        result = self.repository.bulk_update(BulkDocumentUpdate(
            document_ids=[first.id, second.id], collection="Research", tags=["Review"], favorite=True
        ))
        renamed = self.repository.rename_collection("Research", "Evidence")

        self.assertEqual(result.updated, 2)
        self.assertEqual(renamed, 2)
        self.assertEqual(self.repository.get(first.id).collection, "Evidence")
        self.assertEqual(self.repository.get(second.id).tags, ["Review"])
        self.assertTrue(self.repository.get(second.id).favorite)
        self.assertEqual(self.repository.clear_collection("Evidence"), 2)
        self.assertIsNone(self.repository.get(first.id).collection)

    def test_saved_views_are_persisted_updated_and_deleted(self) -> None:
        created = self.repository.create_saved_view(SavedViewCreate(
            name="Recent research", query="atlas", collection="Research",
            tags=["Review"], source_types=["pdf"], favorite=True, date_range="30d", sort="title",
        ))
        updated = self.repository.update_saved_view(created.id, SavedViewUpdate(
            name="Atlas research", query="atlas", collection="Research",
            tags=["Review"], source_types=["pdf"], favorite=True, date_range="7d", sort="recent",
        ))

        self.assertEqual(self.repository.list_saved_views().total, 1)
        self.assertEqual(updated.name, "Atlas research")
        self.assertEqual(updated.date_range, "7d")
        self.assertTrue(self.repository.delete_saved_view(created.id))
        self.assertEqual(self.repository.list_saved_views().total, 0)

    def test_conversation_history_lifecycle(self) -> None:
        conversation = self.repository.create_conversation(
            ConversationCreate(title="Atlas ownership")
        )
        answer = AnswerResponse(
            question="Who owns Atlas?", answer="Maya owns Atlas [1].", citations=[Citation(
                number=1, document_id="doc-1", title="Atlas memo", source_type="pdf",
                page_number=2, start_seconds=None, end_seconds=None,
                passage="Maya owns the Atlas launch.", chunk_index=0, score=0.9,
                start_offset=0, end_offset=28,
            )],
            model="test", generated=True, elapsed_ms=2, grounded=True,
            scope_description="All documents", conversation_id=conversation.id,
        )
        self.repository.save_conversation_exchange(
            conversation.id, "Who owns Atlas?", answer
        )
        detail = self.repository.get_conversation_detail(conversation.id)
        renamed = self.repository.update_conversation(
            conversation.id, ConversationUpdate(title="Atlas launch")
        )

        self.assertEqual(len(detail.messages), 2)
        self.assertEqual(detail.messages[0].role, "user")
        self.assertEqual(detail.messages[1].content, "Maya owns Atlas [1].")
        self.assertEqual(detail.messages[1].citations[0].page_number, 2)
        markdown = build_markdown_report(detail)
        portable = build_json_report(detail)
        self.assertIn("# Atlas ownership", markdown)
        self.assertIn("## Evidence register", markdown)
        self.assertIn("Atlas memo", markdown)
        self.assertEqual(len(portable["evidence_register"]), 1)
        self.assertEqual(report_filename("Atlas: Ownership?", "md"), "atlas-ownership.md")
        self.assertEqual(len(self.repository.recent_conversation_messages(conversation.id)), 2)
        self.assertEqual(renamed.title, "Atlas launch")
        self.assertTrue(self.repository.delete_conversation(conversation.id))
        self.assertIsNone(self.repository.get_conversation_detail(conversation.id))

    def test_document_notes_can_be_created_updated_and_deleted(self) -> None:
        document = self.repository.create(
            DocumentCreate(title="Annotated memo", content="Local note taking matters.")
        )

        created = self.repository.create_note(
            document.id,
            DocumentNoteCreate(content=" Summarize the decision section. "),
        )
        updated = self.repository.update_note(
            document.id,
            created.id,
            DocumentNoteUpdate(content="Decision: keep notes local."),
        )
        notes = self.repository.list_notes(document.id)

        self.assertEqual(created.content, "Summarize the decision section.")
        self.assertEqual(updated.content, "Decision: keep notes local.")
        self.assertEqual(notes.total, 1)
        self.assertEqual(notes.items[0].id, created.id)
        self.assertTrue(self.repository.delete_note(document.id, created.id))
        self.assertEqual(self.repository.list_notes(document.id).total, 0)

    def test_document_delete_cascades_notes(self) -> None:
        document = self.repository.create(
            DocumentCreate(title="Temporary memo", content="Remove this source.")
        )
        self.repository.create_note(document.id, DocumentNoteCreate(content="No longer needed."))

        self.assertTrue(self.repository.delete(document.id))

        self.assertIsNone(self.repository.list_notes(document.id))

    def test_document_highlights_are_anchored_updated_and_deleted(self) -> None:
        content = "The launch decision is documented here."
        document = self.repository.create(DocumentCreate(title="Launch memo", content=content))
        start = content.index("launch decision")
        end = start + len("launch decision")

        created = self.repository.create_highlight(
            document.id,
            DocumentHighlightCreate(
                start_offset=start,
                end_offset=end,
                selected_text="launch decision",
                color="yellow",
                annotation="Key decision",
            ),
        )
        updated = self.repository.update_highlight(
            document.id,
            created.id,
            DocumentHighlightUpdate(color="green", annotation="Confirmed decision"),
        )

        self.assertEqual(self.repository.list_highlights(document.id).total, 1)
        self.assertEqual(updated.color, "green")
        self.assertEqual(updated.annotation, "Confirmed decision")
        self.assertTrue(self.repository.delete_highlight(document.id, created.id))

    def test_highlight_rejects_mismatched_text_and_overlap(self) -> None:
        document = self.repository.create(DocumentCreate(title="Memo", content="abcdef"))
        self.repository.create_highlight(
            document.id,
            DocumentHighlightCreate(start_offset=1, end_offset=3, selected_text="bc"),
        )
        with self.assertRaises(ValueError):
            self.repository.create_highlight(
                document.id,
                DocumentHighlightCreate(start_offset=2, end_offset=4, selected_text="cd"),
            )
        with self.assertRaises(ValueError):
            self.repository.create_highlight(
                document.id,
                DocumentHighlightCreate(start_offset=3, end_offset=5, selected_text="wrong"),
            )

    def test_chunking_has_bounded_size_and_overlap(self) -> None:
        words = [f"word{index}" for index in range(CHUNK_WORDS + 20)]
        chunks = split_chunks(" ".join(words))

        self.assertEqual(len(chunks), 2)
        self.assertLessEqual(len(chunks[0].split()), CHUNK_WORDS)
        self.assertEqual(
            chunks[0].split()[-CHUNK_OVERLAP:],
            chunks[1].split()[:CHUNK_OVERLAP],
        )

    def test_rejects_non_pdf_bytes(self) -> None:
        with self.assertRaisesRegex(IngestionError, "not a valid PDF"):
            extract_pdf(b"plain text")

    def test_image_ocr_extracts_searchable_text(self) -> None:
        ocr = OCRService()
        ocr._engine = FakeOCREngine()

        extraction = extract_image(image_bytes(), ocr)

        self.assertEqual(extraction.pages, [(1, "Scanned contract\nTotal due 2026")])
        self.assertEqual(extraction.ocr_pages, 1)

    def test_scanned_pdf_routes_page_through_ocr(self) -> None:
        pdf = pymupdf.open()
        page = pdf.new_page(width=700, height=160)
        page.insert_image(page.rect, stream=image_bytes())
        data = pdf.tobytes()
        pdf.close()
        ocr = OCRService()
        ocr._engine = FakeOCREngine()

        extraction = extract_pdf(data, ocr)

        self.assertEqual(extraction.pages, [(1, "Scanned contract\nTotal due 2026")])
        self.assertEqual(extraction.ocr_pages, 1)

    def test_semantic_search_finds_concept_without_keyword_overlap(self) -> None:
        self.repository.create(
            DocumentCreate(title="Operations", content="Reducing cloud costs improves the annual budget.")
        )
        self.repository.create(
            DocumentCreate(title="Garden", content="Tomatoes need sunlight and regular watering.")
        )
        chunks = self.repository.chunks_pending_embedding("test-model")
        self.repository.save_embeddings(
            [
                (chunks[0]["id"], np.array([1.0, 0.0], dtype=np.float32)),
                (chunks[1]["id"], np.array([0.0, 1.0], dtype=np.float32)),
            ],
            "test-model",
        )

        results = self.repository.search("infrastructure expenses", 10, "semantic", FakeEmbedder())

        self.assertEqual(results.total, 1)
        self.assertEqual(results.items[0].title, "Operations")
        self.assertEqual(results.mode, "semantic")
        self.assertIsNone(results.warning)

    def test_transcript_passages_preserve_timestamps_in_search(self) -> None:
        payload = DocumentCreate(
            title="Team meeting",
            content="The launch date is October. The budget review follows.",
            source_type="audio",
            source_name="meeting.m4a",
        )
        document = self.repository.create(
            payload,
            timed_passages=[
                (12.5, 18.0, "The launch date is October."),
                (18.0, 24.25, "The budget review follows."),
            ],
            duration_seconds=24.25,
            language="en",
        )

        results = self.repository.search("budget", 10, "keyword", FakeEmbedder())

        self.assertEqual(document.duration_seconds, 24.25)
        self.assertEqual(document.language, "en")
        self.assertEqual(results.items[0].start_seconds, 18.0)
        self.assertEqual(results.items[0].end_seconds, 24.25)

    def test_groups_whisper_segments_into_bounded_passages(self) -> None:
        segments = [
            SimpleNamespace(start=0.0, end=4.0, text="one two three"),
            SimpleNamespace(start=4.0, end=8.5, text="four five"),
        ]

        passages = group_segments(segments)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].start, 0.0)
        self.assertEqual(passages[0].end, 8.5)
        self.assertEqual(passages[0].text, "one two three four five")

    def test_rejects_undecodable_media_before_model_load(self) -> None:
        service = TranscriptionService()

        with self.assertRaisesRegex(TranscriptionError, "could not be decoded"):
            service.transcribe(b"not media", ".mp3")

        self.assertFalse(service.loaded)

    def test_transcription_cancellation_terminates_worker_process(self) -> None:
        service = TranscriptionService()
        process = SimpleNamespace(
            terminate=lambda: None,
            wait=lambda timeout: 0,
            kill=lambda: None,
        )

        with patch("app.transcription.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(TranscriptionError, "cancelled"):
                service.transcribe(b"media", ".wav", should_cancel=lambda: True)

    def test_speech_service_reports_unavailable_without_say(self) -> None:
        with patch("app.speech.which", return_value=None):
            service = LocalSpeechService()

        self.assertFalse(service.available())
        self.assertEqual(service.voices(), [])
        with self.assertRaisesRegex(SpeechError, "unavailable"):
            service.synthesize("Hello", None, 180)

    def test_speech_service_parses_macos_voice_names(self) -> None:
        completed = SimpleNamespace(stdout="Alex                en_US    # Most people recognize me\nSamantha            en_US\n")
        with patch("app.speech.which", return_value="/usr/bin/say"), patch(
            "app.speech.subprocess.run",
            return_value=completed,
        ):
            service = LocalSpeechService()

            self.assertEqual(service.voices(), ["Alex", "Samantha"])

    def test_background_image_ingestion_completes_and_creates_document(self) -> None:
        ocr = OCRService()
        ocr._engine = FakeOCREngine()
        worker = IngestionWorker(
            self.repository,
            ocr,
            TranscriptionService(),
            EmbeddingService(),
            Path(self.temp_dir.name) / "uploads",
        )

        queued = worker.submit(
            image_bytes(), "Scanned invoice", "image", "invoice.png"
        )

        self.assertEqual(queued.status, "queued")
        self.assertTrue(worker.process_next())
        completed = self.repository.get_ingestion_job(queued.id)
        document = self.repository.get(queued.id)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.progress, 100)
        self.assertEqual(completed.document_id, queued.id)
        self.assertEqual(document.title, "Scanned invoice")
        self.assertTrue(document.ocr_applied)
        self.assertTrue(document.source_available)

    def test_job_cleanup_keeps_sources_owned_by_completed_documents(self) -> None:
        uploads = Path(self.temp_dir.name) / "uploads"
        uploads.mkdir()
        completed_path = uploads / "completed.png"
        failed_path = uploads / "failed.png"
        completed_path.write_bytes(image_bytes())
        failed_path.write_bytes(b"invalid")
        self.repository.create_ingestion_job(
            "complete-job", "Complete", "image", "complete.png", completed_path
        )
        self.repository.update_ingestion_job("complete-job", status="completed")
        self.repository.create_ingestion_job(
            "failed-job", "Failed", "image", "failed.png", failed_path
        )
        self.repository.update_ingestion_job("failed-job", status="failed")

        removed, disposable = self.repository.cleanup_ingestion_jobs()

        self.assertEqual(removed, 2)
        self.assertEqual(disposable, [failed_path])
        self.assertNotIn(completed_path, disposable)

    def test_ingestion_job_can_be_cancelled_and_retried(self) -> None:
        ocr = OCRService()
        ocr._engine = FakeOCREngine()
        worker = IngestionWorker(
            self.repository,
            ocr,
            TranscriptionService(),
            EmbeddingService(),
            Path(self.temp_dir.name) / "uploads",
        )
        queued = worker.submit(image_bytes(), "Receipt", "image", "receipt.png")

        cancelled = worker.cancel(queued.id)
        self.assertEqual(cancelled.status, "cancelled")
        retried = worker.retry(queued.id)
        self.assertEqual(retried.status, "queued")
        self.assertTrue(worker.process_next())
        self.assertEqual(self.repository.get_ingestion_job(queued.id).status, "completed")

    def test_interrupted_ingestion_jobs_return_to_queue(self) -> None:
        input_path = Path(self.temp_dir.name) / "source.png"
        input_path.write_bytes(image_bytes())
        job = self.repository.create_ingestion_job(
            "job-1", "Recovered scan", "image", "scan.png", input_path
        )

        claimed = self.repository.claim_next_ingestion_job()
        self.assertEqual(claimed.id, job.id)
        self.assertEqual(claimed.status, "running")
        self.assertEqual(self.repository.reset_interrupted_ingestion_jobs(), 1)
        self.assertEqual(self.repository.get_ingestion_job(job.id).status, "queued")


if __name__ == "__main__":
    unittest.main()
