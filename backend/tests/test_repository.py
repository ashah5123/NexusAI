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

from app.models import DocumentCreate
from app.embedding import EmbeddingService
from app.generation import OllamaAnswerService
from app.ingestion import IngestionError, extract_image, extract_pdf
from app.jobs import IngestionWorker
from app.ocr import OCRService
from app.repository import CHUNK_OVERLAP, CHUNK_WORDS, DocumentRepository, split_chunks
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

    def test_retrieval_keeps_full_passage_for_grounded_answers(self) -> None:
        self.repository.create(
            DocumentCreate(title="Atlas notes", content="Maya owns the Atlas release on October 14.")
        )

        passages, warning = self.repository.retrieve("Atlas release", 6, FakeEmbedder())

        self.assertEqual(passages[0]["passage"], "Maya owns the Atlas release on October 14.")
        self.assertIn("keyword evidence", warning)

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
