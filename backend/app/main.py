"""NexusAI API for local document ingestion and retrieval."""

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Body, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from starlette.background import BackgroundTask

from .embedding import EmbeddingService
from .generation import OllamaAnswerService
from .ingestion import MAX_UPLOAD_BYTES
from .jobs import IngestionWorker
from .models import (
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
    DocumentCreate,
    DocumentList,
    DocumentRead,
    EmbeddingStatus,
    HealthResponse,
    IngestionJob,
    IngestionJobList,
    ReindexResult,
    SearchResponse,
    SpeechRequest,
    SpeechStatus,
    TranscriptionStatus,
)
from .repository import DocumentRepository
from .ocr import OCRService
from .speech import LocalSpeechService, SpeechError
from .transcription import MAX_MEDIA_BYTES, TranscriptionService

repository = DocumentRepository()
embedding_service = EmbeddingService()
ocr_service = OCRService()
transcription_service = TranscriptionService()
answer_service = OllamaAnswerService()
speech_service = LocalSpeechService()
ingestion_worker = IngestionWorker(
    repository, ocr_service, transcription_service, embedding_service
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    repository.initialize()
    ingestion_worker.start()
    try:
        yield
    finally:
        ingestion_worker.stop()


app = FastAPI(title="NexusAI API", version="0.9.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def api_root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="nexusai-api", database=repository.health())


@app.post("/api/documents", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
def create_document(payload: DocumentCreate) -> DocumentRead:
    return repository.create(payload)


@app.post(
    "/api/documents/pdf",
    response_model=IngestionJob,
    status_code=status.HTTP_202_ACCEPTED,
)
def upload_pdf(
    data: Annotated[bytes, Body(media_type="application/pdf")],
    filename: Annotated[str, Query(min_length=1, max_length=500)],
    title: Annotated[str | None, Query(min_length=1, max_length=240)] = None,
) -> IngestionJob:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB local upload limit")
    return ingestion_worker.submit(
        data, title or filename.rsplit(".", 1)[0], "pdf", filename
    )


@app.post(
    "/api/documents/image",
    response_model=IngestionJob,
    status_code=status.HTTP_202_ACCEPTED,
)
def upload_image(
    data: Annotated[bytes, Body(media_type="application/octet-stream")],
    filename: Annotated[str, Query(min_length=1, max_length=500)],
    title: Annotated[str | None, Query(min_length=1, max_length=240)] = None,
) -> IngestionJob:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB local upload limit")
    return ingestion_worker.submit(
        data, title or filename.rsplit(".", 1)[0], "image", filename
    )


@app.post(
    "/api/documents/media",
    response_model=IngestionJob,
    status_code=status.HTTP_202_ACCEPTED,
)
def upload_media(
    data: Annotated[bytes, Body(media_type="application/octet-stream")],
    filename: Annotated[str, Query(min_length=1, max_length=500)],
    title: Annotated[str | None, Query(min_length=1, max_length=240)] = None,
) -> IngestionJob:
    if len(data) > MAX_MEDIA_BYTES:
        raise HTTPException(status_code=413, detail="Media exceeds the 100 MB local limit")
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    video_extensions = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    audio_extensions = {
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".aif", ".aiff"
    }
    if suffix not in video_extensions | audio_extensions:
        raise HTTPException(status_code=422, detail="Unsupported audio or video format")
    source_type = "video" if suffix in video_extensions else "audio"
    return ingestion_worker.submit(
        data, title or filename.rsplit(".", 1)[0], source_type, filename
    )


@app.get("/api/ingestion-jobs", response_model=IngestionJobList)
def list_ingestion_jobs(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> IngestionJobList:
    return repository.list_ingestion_jobs(limit)


@app.post("/api/ingestion-jobs/{job_id}/cancel", response_model=IngestionJob)
def cancel_ingestion_job(job_id: str) -> IngestionJob:
    job = ingestion_worker.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Import job not found")
    return job


@app.post("/api/ingestion-jobs/{job_id}/retry", response_model=IngestionJob)
def retry_ingestion_job(job_id: str) -> IngestionJob:
    job = ingestion_worker.retry(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Import job or source file not found")
    return job


@app.get("/api/documents", response_model=DocumentList)
def list_documents(
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    return repository.list(limit=limit, offset=offset)


@app.get("/api/documents/{document_id}", response_model=DocumentRead)
def get_document(document_id: str) -> DocumentRead:
    document = repository.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str) -> Response:
    if not repository.delete(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/search", response_model=SearchResponse)
def search_documents(
    q: Annotated[str, Query(min_length=1, max_length=300)],
    limit: Annotated[int, Query(ge=1, le=50)] = 12,
    mode: Annotated[Literal["keyword", "semantic", "hybrid"], Query()] = "hybrid",
) -> SearchResponse:
    return repository.search(q.strip(), limit, mode, embedding_service)


@app.get("/api/answers/status", response_model=AnswerStatus)
def answer_status() -> AnswerStatus:
    return AnswerStatus(model=answer_service.model_name, available=answer_service.available())


@app.post("/api/answers", response_model=AnswerResponse)
def answer_question(payload: AnswerRequest) -> AnswerResponse:
    passages, warning = repository.retrieve(payload.question, 6, embedding_service)
    return answer_service.answer(payload.question, passages, warning)


@app.get("/api/speech/status", response_model=SpeechStatus)
def speech_status() -> SpeechStatus:
    return SpeechStatus(
        engine=speech_service.engine,
        available=speech_service.available(),
        voices=speech_service.voices(),
    )


@app.post("/api/speech")
def synthesize_speech(payload: SpeechRequest) -> FileResponse:
    try:
        path = speech_service.synthesize(payload.text, payload.voice, payload.rate)
    except SpeechError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="audio/aiff",
        filename="nexusai-speech.aiff",
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


@app.get("/api/embeddings/status", response_model=EmbeddingStatus)
def embedding_status() -> EmbeddingStatus:
    return repository.embedding_status(embedding_service.model_name, embedding_service.loaded)


@app.post("/api/embeddings/reindex", response_model=ReindexResult)
def reindex_embeddings() -> ReindexResult:
    try:
        return embedding_service.index_pending(repository)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The embedding model could not be downloaded or loaded. Keyword search remains available.",
        ) from exc


@app.get("/api/transcription/status", response_model=TranscriptionStatus)
def transcription_status() -> TranscriptionStatus:
    return TranscriptionStatus(
        model=transcription_service.model_name,
        loaded=transcription_service.loaded,
    )
