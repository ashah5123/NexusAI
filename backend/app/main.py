"""NexusAI API for local document ingestion and retrieval."""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .ingestion import IngestionError, MAX_UPLOAD_BYTES, extract_pdf
from .models import DocumentCreate, DocumentList, DocumentRead, HealthResponse, SearchResponse
from .repository import DocumentRepository

repository = DocumentRepository()


@asynccontextmanager
async def lifespan(_: FastAPI):
    repository.initialize()
    yield


app = FastAPI(title="NexusAI API", version="0.3.0", lifespan=lifespan)

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


@app.post("/api/documents/pdf", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
def upload_pdf(
    data: Annotated[bytes, Body(media_type="application/pdf")],
    filename: Annotated[str, Query(min_length=1, max_length=500)],
    title: Annotated[str | None, Query(min_length=1, max_length=240)] = None,
) -> DocumentRead:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB local upload limit")
    try:
        pages = extract_pdf(data)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    document_title = title or filename.rsplit(".", 1)[0]
    content = "\n\n".join(text for _, text in pages)
    payload = DocumentCreate(
        title=document_title,
        content=content,
        source_type="pdf",
        source_name=filename,
    )
    return repository.create(payload, pages=pages)


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
) -> SearchResponse:
    return repository.search(q.strip(), limit)
