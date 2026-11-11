from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from config import BASE_DIR, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_NAME

WHISPER_MODEL_INSTANCE: Any | None = None

def get_whisper_model() -> Any:
    global WHISPER_MODEL_INSTANCE
    if WHISPER_MODEL_INSTANCE is not None:
        return WHISPER_MODEL_INSTANCE
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise HTTPException(
            503,
            "Speech recognition is unavailable because faster-whisper is not installed in the backend environment.",
        ) from exc
    WHISPER_MODEL_INSTANCE = WhisperModel(
        WHISPER_MODEL_NAME,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root=str(BASE_DIR / ".whisper"),
    )
    return WHISPER_MODEL_INSTANCE

def transcribe_audio_bytes(audio_bytes: bytes, filename: str | None = None, language: str | None = None) -> dict:
    if not audio_bytes or len(audio_bytes) < 1024:
        raise HTTPException(400, "The recording is empty or too short to transcribe.")
    file_ext = Path(filename or "voice.webm").suffix or ".webm"
    temp_path = BASE_DIR / f"voice_{uuid.uuid4()}{file_ext}"
    temp_path.write_bytes(audio_bytes)
    try:
        model = get_whisper_model()
        segments, info = model.transcribe(
            str(temp_path),
            language=language,
            task="transcribe",
            beam_size=5,
            best_of=5,
            vad_filter=True,
            word_timestamps=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text and segment.text.strip())
        return {
            "text": text.strip(),
            "language": (info.language or language or "unknown").strip(),
            "duration_seconds": round(float(info.duration or 0.0), 2),
            "model": WHISPER_MODEL_NAME,
        }
    finally:
        temp_path.unlink(missing_ok=True)
