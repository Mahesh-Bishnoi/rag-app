"""HTTP wire shapes. OpenAI-compatible chat plus KB upload views."""

from typing import Literal

from pydantic import BaseModel, Field

from rag_lab.core.models import Citation


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    citations: list[Citation] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    db: bool
    blob: bool


class ModelEntry(BaseModel):
    id: str
    object: str


class ModelsResponse(BaseModel):
    object: str
    data: list[ModelEntry]


class InitiateUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(gt=0)


class InitiateUploadResponse(BaseModel):
    document_id: str
    upload_url: str
    expires_in: int
