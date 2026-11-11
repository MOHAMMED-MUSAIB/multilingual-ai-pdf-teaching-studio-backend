import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError

from config import OLLAMA_MODEL, logger
from database import DatabaseConnection, DatabaseResult, database, init_db, utc_now
from ai_service import ollama_chat, ollama_chat_stream, ollama_get, prepare_chat, retrieve_pages, save_chat_response
from pdf_processing import (
    _dominant_ocr_language, _extract_pdf_page_ocr, _extract_preprocessed_ocr,
    _has_sufficient_arabic_text, _is_urdu_pdf, _needs_pdf_ocr, document_or_404, page_payload,
)
from schemas import ChatRequest, GenerateRequest, Lesson, ReadingPosition, TimerSession, TimerSettings
from voice_service import transcribe_audio_bytes
from config import (
    BASE_DIR, DB_PATH, MAX_PDF_BYTES, MAX_PDF_PAGES, OLLAMA_TIMEOUT, OLLAMA_URL,
    PDF_DIR, PDF_OCR_LANGUAGES, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE,
    WHISPER_MAX_RECORDING_SECONDS, WHISPER_MODEL_NAME,
)

app = FastAPI(title="Multilingual AI PDF Teaching Studio API", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)

init_db()
