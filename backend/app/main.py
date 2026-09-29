"""NexusAI API for local document ingestion and retrieval."""

import json
import mimetypes
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
    BulkDocumentUpdate,
    BulkUpdateResult,
    CleanupResult,
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationRead,
    ConversationUpdate,
    DocumentCreate,
    DocumentList,
    DocumentNoteCreate,
    DocumentNoteList,
    DocumentNoteRead,
    DocumentNoteUpdate,
    DocumentHighlightCreate,
    DocumentHighlightList,
    DocumentHighlightRead,
    DocumentHighlightUpdate,
    DocumentRead,
    DocumentUpdate,
    CollectionRename,
    EmbeddingStatus,
    HealthResponse,
    IngestionJob,
    IngestionJobList,
    ReindexResult,
    SearchResponse,
    SavedViewCreate,
    SavedViewList,
    SavedViewRead,
    SavedViewUpdate,
    SpeechRequest,
    SpeechStatus,
    TranscriptionStatus,
)
from .repository import DocumentRepository
from .reports import build_json_report, build_markdown_report, report_filename
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


app = FastAPI(title="NexusAI API", version="0.16.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
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


@app.delete("/api/ingestion-jobs", response_model=CleanupResult)
def cleanup_ingestion_jobs() -> CleanupResult:
    removed, paths = repository.cleanup_ingestion_jobs()
    for path in paths:
        path.unlink(missing_ok=True)
    return CleanupResult(removed=removed)


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


@app.patch("/api/documents/{document_id}", response_model=DocumentRead)
def update_document(document_id: str, payload: DocumentUpdate) -> DocumentRead:
    document = repository.update(document_id, payload)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.patch("/api/documents", response_model=BulkUpdateResult)
def bulk_update_documents(payload: BulkDocumentUpdate) -> BulkUpdateResult:
    return repository.bulk_update(payload)


@app.patch("/api/collections/{collection_name}", response_model=BulkUpdateResult)
def rename_collection(collection_name: str, payload: CollectionRename) -> BulkUpdateResult:
    return BulkUpdateResult(updated=repository.rename_collection(collection_name, payload.name))


@app.delete("/api/collections/{collection_name}", response_model=BulkUpdateResult)
def clear_collection(collection_name: str) -> BulkUpdateResult:
    return BulkUpdateResult(updated=repository.clear_collection(collection_name))


@app.get("/api/saved-views", response_model=SavedViewList)
def list_saved_views() -> SavedViewList:
    return repository.list_saved_views()


@app.post("/api/saved-views", response_model=SavedViewRead, status_code=status.HTTP_201_CREATED)
def create_saved_view(payload: SavedViewCreate) -> SavedViewRead:
    return repository.create_saved_view(payload)


@app.put("/api/saved-views/{view_id}", response_model=SavedViewRead)
def update_saved_view(view_id: str, payload: SavedViewUpdate) -> SavedViewRead:
    view = repository.update_saved_view(view_id, payload)
    if view is None:
        raise HTTPException(status_code=404, detail="Saved view not found")
    return view


@app.delete("/api/saved-views/{view_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved_view(view_id: str) -> Response:
    if not repository.delete_saved_view(view_id):
        raise HTTPException(status_code=404, detail="Saved view not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/documents/{document_id}/source")
def get_document_source(document_id: str) -> FileResponse:
    document = repository.get(document_id)
    path = repository.source_path(document_id)
    if document is None or path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Original source file not found")
    media_type = mimetypes.guess_type(document.source_name or path.name)[0]
    return FileResponse(path, media_type=media_type or "application/octet-stream")


@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str) -> Response:
    source_path = repository.source_path(document_id)
    if not repository.delete(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    if source_path:
        source_path.unlink(missing_ok=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/documents/{document_id}/notes", response_model=DocumentNoteList)
def list_document_notes(document_id: str) -> DocumentNoteList:
    notes = repository.list_notes(document_id)
    if notes is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return notes


@app.post(
    "/api/documents/{document_id}/notes",
    response_model=DocumentNoteRead,
    status_code=status.HTTP_201_CREATED,
)
def create_document_note(
    document_id: str,
    payload: DocumentNoteCreate,
) -> DocumentNoteRead:
    note = repository.create_note(document_id, payload)
    if note is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return note


@app.patch(
    "/api/documents/{document_id}/notes/{note_id}",
    response_model=DocumentNoteRead,
)
def update_document_note(
    document_id: str,
    note_id: str,
    payload: DocumentNoteUpdate,
) -> DocumentNoteRead:
    note = repository.update_note(document_id, note_id, payload)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@app.delete(
    "/api/documents/{document_id}/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document_note(document_id: str, note_id: str) -> Response:
    if not repository.delete_note(document_id, note_id):
        raise HTTPException(status_code=404, detail="Note not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/documents/{document_id}/highlights", response_model=DocumentHighlightList)
def list_document_highlights(document_id: str) -> DocumentHighlightList:
    highlights = repository.list_highlights(document_id)
    if highlights is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return highlights


@app.post(
    "/api/documents/{document_id}/highlights",
    response_model=DocumentHighlightRead,
    status_code=status.HTTP_201_CREATED,
)
def create_document_highlight(
    document_id: str, payload: DocumentHighlightCreate
) -> DocumentHighlightRead:
    try:
        highlight = repository.create_highlight(document_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if highlight is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return highlight


@app.patch(
    "/api/documents/{document_id}/highlights/{highlight_id}",
    response_model=DocumentHighlightRead,
)
def update_document_highlight(
    document_id: str, highlight_id: str, payload: DocumentHighlightUpdate
) -> DocumentHighlightRead:
    highlight = repository.update_highlight(document_id, highlight_id, payload)
    if highlight is None:
        raise HTTPException(status_code=404, detail="Highlight not found")
    return highlight


@app.delete(
    "/api/documents/{document_id}/highlights/{highlight_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document_highlight(document_id: str, highlight_id: str) -> Response:
    if not repository.delete_highlight(document_id, highlight_id):
        raise HTTPException(status_code=404, detail="Highlight not found")
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


@app.get("/api/conversations", response_model=ConversationList)
def list_conversations() -> ConversationList:
    return repository.list_conversations()


@app.post(
    "/api/conversations", response_model=ConversationRead, status_code=status.HTTP_201_CREATED
)
def create_conversation(payload: ConversationCreate) -> ConversationRead:
    return repository.create_conversation(payload)


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str) -> ConversationDetail:
    conversation = repository.get_conversation_detail(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/api/conversations/{conversation_id}/report")
def export_conversation_report(
    conversation_id: str,
    format: Annotated[Literal["markdown", "json"], Query()] = "markdown",
) -> Response:
    detail = repository.get_conversation_detail(conversation_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if format == "json":
        content = json.dumps(build_json_report(detail), ensure_ascii=False, indent=2)
        filename = report_filename(detail.conversation.title, "json")
        media_type = "application/json"
    else:
        content = build_markdown_report(detail)
        filename = report_filename(detail.conversation.title, "md")
        media_type = "text/markdown"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.patch("/api/conversations/{conversation_id}", response_model=ConversationRead)
def update_conversation(
    conversation_id: str, payload: ConversationUpdate
) -> ConversationRead:
    conversation = repository.update_conversation(conversation_id, payload)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/api/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str) -> Response:
    if not repository.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/answers", response_model=AnswerResponse)
def answer_question(payload: AnswerRequest) -> AnswerResponse:
    if payload.conversation_id:
        conversation = repository.get_conversation(payload.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        title = payload.question[:80].rstrip()
        conversation = repository.create_conversation(ConversationCreate(title=title))
    history = repository.recent_conversation_messages(conversation.id)
    prior_questions = [message["content"] for message in history if message["role"] == "user"][-2:]
    retrieval_query = " ".join([*prior_questions, payload.question])
    passages, warning = repository.retrieve(
        retrieval_query,
        6,
        embedding_service,
        payload.document_ids,
        payload.collection,
        payload.source_types,
    )
    scope_parts = []
    if payload.collection:
        scope_parts.append(f'collection "{payload.collection}"')
    if payload.source_types:
        scope_parts.append("sources: " + ", ".join(payload.source_types))
    if payload.document_ids:
        scope_parts.append(f"{len(payload.document_ids)} selected document(s)")
    scope = " · ".join(scope_parts) if scope_parts else "All documents"
    answer = answer_service.answer(payload.question, passages, warning, scope, history)
    answer = answer.model_copy(update={"conversation_id": conversation.id})
    repository.save_conversation_exchange(conversation.id, payload.question, answer)
    return answer


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
