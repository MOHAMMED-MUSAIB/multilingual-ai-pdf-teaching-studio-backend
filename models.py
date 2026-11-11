from __future__ import annotations

from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class LessonRecord(Base):
    __tablename__ = "lessons"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    pdf_name: Mapped[str | None] = mapped_column(String)
    language: Mapped[str | None] = mapped_column(String)
    level: Mapped[str | None] = mapped_column(String)
    page: Mapped[int | None] = mapped_column(Integer)
    paragraph: Mapped[str | None] = mapped_column(Text)
    lesson_json: Mapped[str | None] = mapped_column(Text)
    completed: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float | None] = mapped_column(Float)

class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_at: Mapped[str] = mapped_column(String, nullable=False)
    last_read_page: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

class PDFPageRecord(Base):
    __tablename__ = "pdf_pages"
    __table_args__ = (
        Index("idx_pdf_pages_document_page", "document_id", "page_number", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

class ConversationRecord(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_conversations_document_updated", "document_id", "updated_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)

class MessageRecord(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
        Index("idx_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String, nullable=False)

class TimerSettingsRecord(Base):
    __tablename__ = "timer_settings"
    __table_args__ = (CheckConstraint("id = 1", name="ck_timer_settings_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    focus_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=25)
    break_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=5)

class FocusSessionRecord(Base):
    __tablename__ = "focus_sessions"
    __table_args__ = (
        CheckConstraint("mode IN ('focus', 'break')", name="ck_focus_sessions_mode"),
        Index("idx_focus_sessions_completed_at", "completed_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    completed: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[str] = mapped_column(String, nullable=False)
    completed_at: Mapped[str] = mapped_column(String, nullable=False)
