import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from app.models import DocumentCreate
from app.ingestion import IngestionError, extract_image, extract_pdf
from app.ocr import OCRService
from app.repository import CHUNK_OVERLAP, CHUNK_WORDS, DocumentRepository, split_chunks


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


if __name__ == "__main__":
    unittest.main()
