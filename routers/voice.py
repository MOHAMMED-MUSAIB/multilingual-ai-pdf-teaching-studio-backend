"""Voice API routes."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from services import transcribe_audio_bytes

router = APIRouter()

@router.post("/api/voice/transcribe")
async def transcribe_voice(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac")):
        raise HTTPException(415, "Unsupported audio format. Upload WAV, MP3, M4A, OGG, WEBM, or FLAC.")
    content = await file.read()
    await file.close()
    if not content:
        raise HTTPException(400, "No audio was captured. Speak into the microphone and try again.")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "Recording is too large. Keep the voice sample under 25 MB.")
    result = transcribe_audio_bytes(content, filename=file.filename)
    if not result["text"].strip():
        raise HTTPException(422, "No speech was detected. Please try again with a clearer recording.")
    return result
