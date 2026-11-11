from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DB_PATH = Path(os.getenv("STUDIO_DB_PATH", str(BASE_DIR / "teaching_history.sqlite3")))

PDF_DIR = Path(os.getenv("STUDIO_PDF_DIR", str(BASE_DIR / "storage" / "pdfs")))

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

MAX_PDF_BYTES = int(os.getenv("MAX_PDF_SIZE_MB", "1024")) * 1024 * 1024

MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "3000"))

PDF_OCR_LANGUAGES = os.getenv("PDF_OCR_LANGUAGES", "eng+kan+urd")

MAX_CONTEXT_CHARS = 7000

logger = logging.getLogger("teaching_studio")

WHISPER_MODEL_NAME = (os.getenv("WHISPER_MODEL", "base") or "base").strip()

WHISPER_DEVICE = (os.getenv("WHISPER_DEVICE", "cpu") or "cpu").strip().lower()

WHISPER_COMPUTE_TYPE = (os.getenv("WHISPER_COMPUTE_TYPE", "int8") or "int8").strip()

WHISPER_MAX_RECORDING_SECONDS = max(5, int(os.getenv("WHISPER_MAX_RECORDING_SECONDS", "60")))
