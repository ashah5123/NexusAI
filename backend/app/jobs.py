from __future__ import annotations

import os
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

from .embedding import EmbeddingService
from .ingestion import extract_image, extract_pdf
from .models import DocumentCreate, IngestionJob
from .ocr import OCRService
from .repository import DocumentRepository
from .transcription import TranscriptionService

UPLOAD_DIR = Path(os.getenv("NEXUSAI_UPLOAD_DIR", "./data/uploads"))


class IngestionWorker:
    """Single local worker backed by durable SQLite job state."""

    def __init__(
        self,
        repository: DocumentRepository,
        ocr: OCRService,
        transcription: TranscriptionService,
        embeddings: EmbeddingService,
        upload_dir: Path = UPLOAD_DIR,
    ) -> None:
        self.repository = repository
        self.ocr = ocr
        self.transcription = transcription
        self.embeddings = embeddings
        self.upload_dir = upload_dir
        self._wake = Event()
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.repository.reset_interrupted_ingestion_jobs()
        self._stop.clear()
        self._thread = Thread(target=self._run, name="nexusai-ingestion", daemon=True)
        self._thread.start()
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2)

    def submit(self, data: bytes, title: str, source_type: str, source_name: str) -> IngestionJob:
        job_id = str(uuid4())
        suffix = Path(source_name).suffix.lower()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.upload_dir / f"{job_id}{suffix}"
        temporary_path = self.upload_dir / f"{job_id}.part"
        temporary_path.write_bytes(data)
        temporary_path.replace(input_path)
        try:
            job = self.repository.create_ingestion_job(
                job_id, title, source_type, source_name, input_path
            )
        except Exception:
            input_path.unlink(missing_ok=True)
            raise
        self._wake.set()
        return job

    def cancel(self, job_id: str) -> IngestionJob | None:
        job = self.repository.cancel_ingestion_job(job_id)
        self._wake.set()
        return job

    def retry(self, job_id: str) -> IngestionJob | None:
        path = self.repository.ingestion_job_input_path(job_id)
        if path is None or not path.exists():
            return None
        job = self.repository.retry_ingestion_job(job_id)
        self._wake.set()
        return job

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.process_next():
                self._wake.wait(timeout=2)
                self._wake.clear()

    def _cancelled(self, job_id: str) -> bool:
        current = self.repository.get_ingestion_job(job_id)
        return current is None or current.cancel_requested

    def _finish_cancelled(self, job_id: str, document_id: str | None = None) -> None:
        if document_id:
            self.repository.delete(document_id)
        self.repository.update_ingestion_job(
            job_id,
            status="cancelled",
            stage="Cancelled",
            progress=0,
            error=None,
            document_id=None,
            cancel_requested=1,
        )

    def process_next(self) -> bool:
        job = self.repository.claim_next_ingestion_job()
        if job is None:
            return False
        try:
            existing = self.repository.get(job.id)
            if existing is not None:
                self.repository.update_ingestion_job(
                    job.id,
                    status="completed",
                    stage="Complete",
                    progress=100,
                    document_id=existing.id,
                )
                return True

            input_path = self.repository.ingestion_job_input_path(job.id)
            if input_path is None or not input_path.exists():
                raise ValueError("The queued source file is missing")
            data = input_path.read_bytes()
            if self._cancelled(job.id):
                self._finish_cancelled(job.id)
                return True

            if job.source_type == "pdf":
                self.repository.update_ingestion_job(
                    job.id, stage="Extracting pages and OCR", progress=20
                )
                extraction = extract_pdf(
                    data,
                    self.ocr,
                    should_cancel=lambda: self._cancelled(job.id),
                    on_ocr_progress=lambda current, total: self.repository.update_ingestion_job(
                        job.id,
                        stage=f"OCR page {current} of {total}",
                        progress=min(70, 20 + round((current / total) * 50)),
                    ),
                )
                payload = DocumentCreate(
                    title=job.title,
                    content="\n\n".join(text for _, text in extraction.pages),
                    source_type="pdf",
                    source_name=job.source_name,
                )
                create_options = {
                    "pages": extraction.pages,
                    "ocr_applied": extraction.ocr_pages > 0,
                    "page_count": extraction.total_pages,
                }
            elif job.source_type == "image":
                self.repository.update_ingestion_job(job.id, stage="Reading image", progress=20)
                extraction = extract_image(data, self.ocr)
                payload = DocumentCreate(
                    title=job.title,
                    content=extraction.pages[0][1],
                    source_type="image",
                    source_name=job.source_name,
                )
                create_options = {
                    "pages": extraction.pages,
                    "ocr_applied": True,
                    "page_count": extraction.total_pages,
                }
            else:
                self.repository.update_ingestion_job(
                    job.id, stage="Transcribing media", progress=20
                )
                transcription = self.transcription.transcribe(
                    data,
                    input_path.suffix.lower(),
                    should_cancel=lambda: self._cancelled(job.id),
                )
                payload = DocumentCreate(
                    title=job.title,
                    content=transcription.text,
                    source_type=job.source_type,
                    source_name=job.source_name,
                )
                create_options = {
                    "timed_passages": [
                        (passage.start, passage.end, passage.text)
                        for passage in transcription.passages
                    ],
                    "duration_seconds": transcription.duration_seconds,
                    "language": transcription.language,
                }

            if self._cancelled(job.id):
                self._finish_cancelled(job.id)
                return True
            self.repository.update_ingestion_job(job.id, stage="Saving passages", progress=80)
            document = self.repository.create(
                payload, document_id=job.id, source_path=input_path, **create_options
            )
            if self._cancelled(job.id):
                self._finish_cancelled(job.id, document.id)
                return True

            if self.embeddings.loaded:
                self.repository.update_ingestion_job(job.id, stage="Indexing semantics", progress=90)
                try:
                    self.embeddings.index_pending(self.repository)
                except Exception:
                    pass
            self.repository.update_ingestion_job(
                job.id,
                status="completed",
                stage="Complete",
                progress=100,
                error=None,
                document_id=document.id,
            )
        except Exception as exc:
            if self._cancelled(job.id):
                self._finish_cancelled(job.id)
            else:
                self.repository.update_ingestion_job(
                    job.id,
                    status="failed",
                    stage="Failed",
                    error=str(exc) or "Import failed",
                )
        return True
