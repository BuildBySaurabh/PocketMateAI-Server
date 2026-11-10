from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


MAX_CONTEXT_ITEMS = 6
MAX_CONTEXT_CHARS = 1500


class ImagePayload(BaseModel):
    """One picture the user chose to send (a photo they took or a screenshot they picked)."""

    media_type: Literal["image/jpeg", "image/png", "image/webp"]
    # base64 text; about 2 MB of picture
    data: str = Field(min_length=1, max_length=2_800_000)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(min_length=1, max_length=40)
    # Facts the user explicitly asked the app to remember. Optional.
    memory: Optional[List[str]] = Field(default=None, max_length=50)
    # A short style hint chosen in the app ("be gentle and brief"). Optional.
    persona: Optional[str] = Field(default=None, max_length=400)
    # Passages from the user's own imported documents, picked on the phone. Optional.
    context: Optional[List[str]] = Field(default=None, max_length=MAX_CONTEXT_ITEMS)
    # True when the user allowed web search for this question. The server decides if it can.
    web_search: bool = False
    image: Optional[ImagePayload] = None

    @field_validator("context")
    @classmethod
    def _context_items_are_short(cls, items):
        if items and any(len(i) > MAX_CONTEXT_CHARS for i in items):
            raise ValueError("context passage too long")
        return items


class ChatResponse(BaseModel):
    reply: str
    # True only if web results were actually added to this answer.
    web_search_used: bool = False


class ErrorResponse(BaseModel):
    # Stable code the app maps to a friendly message.
    code: Literal["unauthorized", "rate_limited", "model_unavailable", "bad_request", "server_error"]
    message: str
