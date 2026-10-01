from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    database: str
    database_backend: Literal["sqlite", "postgresql"]


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=10, max_length=200)

    @field_validator("name", "email")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=200)


class UserRead(BaseModel):
    id: str
    name: str
    email: str


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class WorkspaceRead(BaseModel):
    id: str
    name: str
    role: Literal["owner", "editor", "viewer"]


class AuthSessionRead(BaseModel):
    user: UserRead
    workspace: WorkspaceRead
    workspaces: list[WorkspaceRead]


class WorkspaceSwitch(BaseModel):
    workspace_id: str


class InvitationCreate(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    role: Literal["editor", "viewer"] = "viewer"


class InvitationRead(BaseModel):
    token: str
    email: str
    workspace_name: str
    role: Literal["editor", "viewer"]
    expires_at: datetime


class InvitationAccept(BaseModel):
    token: str = Field(min_length=20, max_length=200)


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


class BulkDocumentUpdate(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=100)
    collection: str | None = Field(default=None, max_length=120)
    tags: list[str] | None = Field(default=None, max_length=12)
    favorite: bool | None = None

    @field_validator("document_ids")
    @classmethod
    def unique_document_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @field_validator("collection")
    @classmethod
    def normalize_collection(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str] | None) -> list[str] | None:
        return DocumentUpdate.normalize_tags(value)


class BulkUpdateResult(BaseModel):
    updated: int


class CollectionRename(BaseModel):
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class SavedViewCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    query: str = Field(default="", max_length=300)
    collection: str | None = Field(default=None, max_length=120)
    tags: list[str] = Field(default_factory=list, max_length=12)
    source_types: list[Literal["text", "markdown", "pdf", "image", "audio", "video", "transcript"]] = Field(default_factory=list, max_length=7)
    favorite: bool = False
    date_range: Literal["all", "7d", "30d", "year"] = "all"
    sort: Literal["recent", "title", "favorite"] = "recent"

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return value.strip()

    @field_validator("collection")
    @classmethod
    def normalize_collection(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        return DocumentUpdate.normalize_tags(value) or []


class SavedViewUpdate(SavedViewCreate):
    pass


class SavedViewRead(SavedViewCreate):
    id: str
    created_at: datetime
    updated_at: datetime


class SavedViewList(BaseModel):
    items: list[SavedViewRead]
    total: int


class DocumentNoteCreate(BaseModel):
    content: str = Field(min_length=1, max_length=10_000)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class DocumentNoteUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=10_000)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class DocumentNoteRead(BaseModel):
    id: str
    document_id: str
    content: str
    created_at: datetime
    updated_at: datetime


class DocumentNoteList(BaseModel):
    items: list[DocumentNoteRead]
    total: int


class DocumentHighlightCreate(BaseModel):
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    selected_text: str = Field(min_length=1, max_length=20_000)
    color: Literal["yellow", "green", "blue", "pink"] = "yellow"
    annotation: str | None = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def validate_offsets(self) -> "DocumentHighlightCreate":
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if self.end_offset - self.start_offset != len(self.selected_text):
            raise ValueError("offset span must equal selected_text length")
        return self

    @field_validator("selected_text")
    @classmethod
    def reject_blank_selection(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("annotation")
    @classmethod
    def normalize_annotation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class DocumentHighlightUpdate(BaseModel):
    color: Literal["yellow", "green", "blue", "pink"] | None = None
    annotation: str | None = Field(default=None, max_length=10_000)

    @field_validator("annotation")
    @classmethod
    def normalize_annotation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class DocumentHighlightRead(BaseModel):
    id: str
    document_id: str
    start_offset: int
    end_offset: int
    selected_text: str
    color: str
    annotation: str | None
    created_at: datetime
    updated_at: datetime


class DocumentHighlightList(BaseModel):
    items: list[DocumentHighlightRead]
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
    workspace_id: str = "local"


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
    conversation_id: str | None = None
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    collection: str | None = Field(default=None, max_length=120)
    source_types: list[Literal["text", "markdown", "pdf", "image", "audio", "video", "transcript"]] = Field(
        default_factory=list, max_length=7
    )

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("collection")
    @classmethod
    def normalize_collection(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None


class Citation(BaseModel):
    number: int
    document_id: str
    title: str
    source_type: str
    page_number: int | None
    start_seconds: float | None
    end_seconds: float | None
    passage: str
    chunk_index: int
    score: float
    start_offset: int | None
    end_offset: int | None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    model: str
    generated: bool
    elapsed_ms: float
    warning: str | None = None
    grounded: bool = False
    scope_description: str = "All documents"
    conversation_id: str | None = None


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ConversationRead(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationList(BaseModel):
    items: list[ConversationRead]
    total: int


class ConversationMessageRead(BaseModel):
    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] = Field(default_factory=list)
    generated: bool = False
    grounded: bool = False
    scope_description: str = "All documents"
    warning: str | None = None
    created_at: datetime


class ConversationDetail(BaseModel):
    conversation: ConversationRead
    messages: list[ConversationMessageRead]


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
