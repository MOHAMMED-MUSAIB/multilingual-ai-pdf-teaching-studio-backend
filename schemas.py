from __future__ import annotations

from pydantic import BaseModel, Field

from config import OLLAMA_MODEL

class ChatRequest(BaseModel):
    document_id: str | None = None
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
    current_page: int | None = Field(default=None, ge=1)
    selected_text: str | None = Field(default=None, max_length=4000)
    include_current_page: bool = True
    model: str | None = None

class ReadingPosition(BaseModel):
    page: int = Field(ge=1)

class TimerSettings(BaseModel):
    focus_minutes: int = Field(default=25, ge=1, le=180)
    break_minutes: int = Field(default=5, ge=1, le=60)

class TimerSession(BaseModel):
    mode: str = Field(pattern="^(focus|break)$")
    duration_seconds: int = Field(ge=1, le=10800)
    completed: bool = False
    started_at: str | None = None

class Lesson(BaseModel):
    id: str
    createdAt: str
    updatedAt: str
    pdfName: str = ""
    language: str = ""
    level: str = ""
    page: int = Field(default=1, ge=1)
    paragraph: str = ""
    lesson: dict = Field(default_factory=dict)
    completed: bool = False
    score: float | None = None

class GenerateRequest(BaseModel):
    paragraph: str = Field(min_length=1, max_length=12000)
    language: str
    level: str
    model: str = OLLAMA_MODEL
