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
    source_type: Literal["text", "markdown", "transcript", "pdf"] = "text"
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
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentRead]
    total: int


class SearchHit(DocumentRead):
    score: float
    snippet: str
    chunk_index: int
    page_number: int | None


class SearchResponse(BaseModel):
    query: str
    items: list[SearchHit]
    total: int
    elapsed_ms: float
    mode: Literal["keyword", "semantic", "hybrid"] = "keyword"
    warning: str | None = None


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
