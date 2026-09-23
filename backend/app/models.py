from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    database: str


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=2_000_000)
    source_type: Literal[
        "text", "markdown", "transcript", "pdf", "image", "audio", "video"
    ] = "text"
    source_name: str | None = Field(default=None, max_length=500)

    @field_validator("title", "content")
    @classmethod
    def reject_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class DocumentRead(BaseModel):
    id: str
    title: str
    content: str
    source_type: str
    source_name: str | None
    status: str
    word_count: int
    page_count: int
    ocr_applied: bool
    duration_seconds: float | None
    language: str | None
    collection: str | None
    tags: list[str]
    favorite: bool
    source_available: bool
    created_at: datetime
    updated_at: datetime


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    collection: str | None = Field(default=None, max_length=120)
    tags: list[str] | None = Field(default=None, max_length=12)
    favorite: bool | None = None

    @field_validator("title")
    @classmethod
    def reject_blank_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("collection")
    @classmethod
    def normalize_collection(cls, value: str | None) -> str | None:
        value = value.strip() if value else None
        return value or None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        tags: list[str] = []
        for item in value:
            tag = item.strip()[:40]
            if tag and tag.lower() not in {existing.lower() for existing in tags}:
                tags.append(tag)
        return tags


class DocumentList(BaseModel):
    items: list[DocumentRead]
    total: int


class IngestionJob(BaseModel):
    id: str
    title: str
    source_type: Literal["pdf", "image", "audio", "video"]
    source_name: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    stage: str
    progress: int = Field(ge=0, le=100)
    error: str | None
    document_id: str | None
    cancel_requested: bool
    created_at: datetime
    updated_at: datetime


class IngestionJobList(BaseModel):
    items: list[IngestionJob]
    total: int


class CleanupResult(BaseModel):
    removed: int


class SearchHit(DocumentRead):
    score: float
    snippet: str
    chunk_index: int
    page_number: int | None
    start_seconds: float | None
    end_seconds: float | None


class SearchResponse(BaseModel):
    query: str
    items: list[SearchHit]
    total: int
    elapsed_ms: float
    mode: Literal["keyword", "semantic", "hybrid"] = "keyword"
    warning: str | None = None


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class Citation(BaseModel):
    number: int
    document_id: str
    title: str
    source_type: str
    page_number: int | None
    start_seconds: float | None
    end_seconds: float | None
    passage: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    model: str
    generated: bool
    elapsed_ms: float
    warning: str | None = None


class AnswerStatus(BaseModel):
    model: str
    available: bool


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    voice: str | None = Field(default=None, max_length=120)
    rate: int = Field(default=180, ge=80, le=320)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class SpeechStatus(BaseModel):
    engine: str
    available: bool
    voices: list[str] = []


class EmbeddingStatus(BaseModel):
    model: str
    total_chunks: int
    indexed_chunks: int
    pending_chunks: int
    ready: bool
    loaded: bool


class ReindexResult(EmbeddingStatus):
    indexed_now: int
    elapsed_ms: float


class TranscriptionStatus(BaseModel):
    model: str
    loaded: bool
    device: str = "cpu-int8"
