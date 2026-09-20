import tempfile
import unittest
from pathlib import Path

from app.models import DocumentCreate
from app.ingestion import IngestionError, extract_pdf
from app.repository import CHUNK_OVERLAP, CHUNK_WORDS, DocumentRepository, split_chunks


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

        results = self.repository.search("semantic", 10)

        self.assertEqual(results.total, 1)
        self.assertEqual(results.items[0].id, document.id)
        self.assertIn("<mark>semantic</mark>", results.items[0].snippet)
        self.assertIsNone(results.items[0].page_number)
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

        results = self.repository.search("decisive evidence", 10)

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


if __name__ == "__main__":
    unittest.main()
