"""Chat API routes."""

import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from services import (
    ChatRequest, OLLAMA_MODEL, database, document_or_404, logger,
    ollama_chat, ollama_chat_stream, prepare_chat, retrieve_pages,
    save_chat_response, utc_now,
)

router = APIRouter()

@router.get("/api/chat/history")
def chat_history(document_id: str | None = Query(default=None)) -> list[dict]:
    with database() as con:
        if document_id is None:
            rows = con.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 100"
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM conversations WHERE document_id=? ORDER BY updated_at DESC LIMIT 100",
                (document_id,),
            ).fetchall()
    return [dict(row) for row in rows]

@router.get("/api/chat/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    with database() as con:
        conversation = con.execute(
            "SELECT * FROM conversations WHERE id=?", (conversation_id,)
        ).fetchone()
        if conversation is None:
            raise HTTPException(404, "Conversation not found.")
        messages = con.execute(
            "SELECT id, role, content, page_number, created_at FROM messages WHERE conversation_id=? ORDER BY created_at",
            (conversation_id,),
        ).fetchall()
    return {"conversation": dict(conversation), "messages": [dict(row) for row in messages]}

@router.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    prepared = prepare_chat(request)
    answer = await ollama_chat(prepared["messages"], prepared["model"], num_predict=384)
    save_chat_response(
        prepared["conversation_id"],
        prepared["conversation"],
        request,
        prepared["message"],
        answer,
    )
    return {
        "conversation_id": prepared["conversation_id"],
        "answer": answer,
        "model": prepared["model"],
        "context_pages": prepared["context_pages"],
        "used_selected_text": prepared["used_selected_text"],
    }

@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    prepared = prepare_chat(request)

    async def events() -> AsyncIterator[str]:
        answer_parts: list[str] = []
        try:
            async for content in ollama_chat_stream(prepared["messages"], prepared["model"]):
                answer_parts.append(content)
                yield json.dumps({"type": "token", "content": content}, ensure_ascii=False) + "\n"
            answer = "".join(answer_parts).strip()
            if not answer:
                raise HTTPException(502, "Ollama returned an empty answer. Please try again.")
            save_chat_response(
                prepared["conversation_id"],
                prepared["conversation"],
                request,
                prepared["message"],
                answer,
            )
            yield json.dumps({
                "type": "done",
                "conversation_id": prepared["conversation_id"],
                "answer": answer,
                "model": prepared["model"],
                "context_pages": prepared["context_pages"],
                "used_selected_text": prepared["used_selected_text"],
            }, ensure_ascii=False) + "\n"
        except HTTPException as exc:
            yield json.dumps({"type": "error", "error": exc.detail}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.delete("/api/chat/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    with database() as con:
        cursor = con.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        if cursor.rowcount == 0:
            raise HTTPException(404, "Conversation not found.")
    return {"deleted": True}

@router.post("/api/chat/{conversation_id}/clear")
def clear_conversation(conversation_id: str) -> dict:
    with database() as con:
        exists = con.execute(
            "SELECT 1 FROM conversations WHERE id=?", (conversation_id,)
        ).fetchone()
        if exists is None:
            raise HTTPException(404, "Conversation not found.")
        con.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        con.execute(
            "UPDATE conversations SET title='New conversation', updated_at=? WHERE id=?",
            (utc_now(), conversation_id),
        )
    return {"cleared": True}
