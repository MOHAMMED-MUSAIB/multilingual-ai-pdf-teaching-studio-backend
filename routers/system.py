"""System API routes."""

from fastapi import APIRouter, HTTPException

from services import SQLAlchemyError, database, logger, ollama_chat, ollama_get, OLLAMA_MODEL

router = APIRouter()

@router.get("/api/health")
def health() -> dict:
    try:
        with database() as con:
            con.execute("SELECT 1").fetchone()
    except SQLAlchemyError as exc:
        logger.exception("SQLite health check failed")
        raise HTTPException(503, "The local database is unavailable.") from exc
    return {"ok": True, "database": "connected"}

@router.get("/api/ollama/status")
async def ollama_status() -> dict:
    payload = await ollama_get("/api/tags")
    models = payload.get("models", [])
    names = [entry.get("name", "") for entry in models]
    return {
        "ok": True,
        "model": OLLAMA_MODEL,
        "model_available": OLLAMA_MODEL in names,
        "models": names,
    }

@router.get("/api/ollama/models")
async def ollama_models() -> dict:
    payload = await ollama_get("/api/tags")
    return {
        "models": [
            {"name": entry["name"]}
            for entry in payload.get("models", [])
            if isinstance(entry.get("name"), str)
        ]
    }

@router.post("/api/ollama/test")
async def test_ollama() -> dict:
    answer = await ollama_chat(
        [{"role": "user", "content": "Reply with OK only."}], OLLAMA_MODEL
    )
    return {"ok": True, "model": OLLAMA_MODEL, "response": answer}
