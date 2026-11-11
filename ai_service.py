from __future__ import annotations

import json
import re
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException
from sqlalchemy.engine import RowMapping

from config import MAX_CONTEXT_CHARS, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL, logger
from database import database, utc_now
from pdf_processing import _needs_pdf_ocr, document_or_404
from schemas import ChatRequest

async def ollama_get(path: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{OLLAMA_URL}{path}")
            response.raise_for_status()
            return response.json()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise HTTPException(
            503,
            "Ollama is not responding. Start Ollama and confirm it is available at the configured local URL.",
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Ollama status request failed: %s", exc)
        raise HTTPException(502, "Ollama returned an invalid status response.") from exc

async def ollama_chat(
    messages: list[dict],
    model: str,
    num_predict: int | None = None,
) -> str:
    options: dict[str, int | float] = {"temperature": 0.3, "num_ctx": 8192}
    if num_predict is not None:
        options["num_predict"] = num_predict
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": options,
                },
            )
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "").strip()
            if not content:
                raise HTTPException(502, "Ollama returned an empty answer. Please try again.")
            return content
    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        raise HTTPException(
            504,
            "The local model took too long to respond. Try a shorter question or a smaller page context.",
        ) from exc
    except httpx.ConnectError as exc:
        raise HTTPException(
            503,
            "Cannot connect to Ollama. Start Ollama and check its configured local URL.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.warning("Ollama chat request failed with status %s", exc.response.status_code)
        if exc.response.status_code == 404:
            raise HTTPException(
                503,
                f"The model '{model}' is unavailable in Ollama. Run `ollama pull {model}` first.",
            ) from exc
        raise HTTPException(502, "Ollama could not complete this request.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Invalid Ollama chat response: %s", exc)
        raise HTTPException(502, "Ollama returned an invalid response.") from exc

async def ollama_chat_stream(messages: list[dict], model: str) -> AsyncIterator[str]:
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "options": {"temperature": 0.3, "num_ctx": 8192, "num_predict": 384},
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    result = json.loads(line)
                    content = result.get("message", {}).get("content", "")
                    if content:
                        yield content
    except httpx.TimeoutException as exc:
        raise HTTPException(
            504,
            "The local model took too long to respond. Try a shorter question or a smaller page context.",
        ) from exc
    except httpx.ConnectError as exc:
        raise HTTPException(
            503,
            "Cannot connect to Ollama. Start Ollama and check its configured local URL.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.warning("Ollama chat stream failed with status %s", exc.response.status_code)
        if exc.response.status_code == 404:
            raise HTTPException(
                503,
                f"The model '{model}' is unavailable in Ollama. Run `ollama pull {model}` first.",
            ) from exc
        raise HTTPException(502, "Ollama could not complete this request.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Invalid Ollama chat stream: %s", exc)
        raise HTTPException(502, "Ollama returned an invalid response.") from exc

STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "can", "could",
    "does", "explain", "for", "from", "give", "help", "into", "make", "more",
    "page", "please", "section", "show", "some", "that", "the", "this", "with",
    "would", "what", "when", "where", "which", "who", "why",
}

def retrieve_pages(
    document_id: str,
    question: str,
    current_page: int | None,
    include_current_page: bool,
) -> list[tuple[int, str]]:
    with database() as con:
        pages = con.execute(
            """SELECT p.page_number, p.extracted_text, d.filename
               FROM pdf_pages p JOIN documents d ON d.id = p.document_id
               WHERE p.document_id = ? ORDER BY p.page_number""",
            (document_id,),
        ).fetchall()
    readable_pages = [
        row
        for row in pages
        if row["extracted_text"].strip()
        and not _needs_pdf_ocr(row["extracted_text"], row["filename"])
    ]
    terms = {
        term.lower()
        for term in re.findall(r"[\w.]+", question, flags=re.UNICODE)
        if len(term) > 1 and term.lower() not in STOP_WORDS
    }
    ranked: list[tuple[int, int, str]] = []
    for row in readable_pages:
        text = row["extracted_text"]
        lower_text = text.lower()
        score = sum(lower_text.count(term) for term in terms)
        if include_current_page and current_page == row["page_number"]:
            score += 1
        if score:
            ranked.append((score, row["page_number"], text))
    ranked.sort(key=lambda result: (-result[0], result[1]))
    selected: dict[int, str] = {}
    if include_current_page and current_page is not None:
        for row in readable_pages:
            if row["page_number"] == current_page:
                selected[current_page] = row["extracted_text"]
                break
    for _, page_number, text in ranked:
        selected.setdefault(page_number, text)
        if len(selected) >= 4:
            break
    if current_page is not None and include_current_page:
        for nearby in (current_page - 1, current_page + 1):
            if len(selected) >= 4:
                break
            if nearby > 0:
                for row in readable_pages:
                    if row["page_number"] == nearby:
                        selected.setdefault(nearby, row["extracted_text"])
                        break
    result: list[tuple[int, str]] = []
    chars = 0
    for page_number, text in selected.items():
        remaining = MAX_CONTEXT_CHARS - chars
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        result.append((page_number, excerpt))
        chars += len(excerpt)
    return result

def prepare_chat(request: ChatRequest) -> dict[str, Any]:
    message = request.message.strip()
    if not message:
        raise HTTPException(400, "Type a message before sending.")
    doc = document_or_404(request.document_id) if request.document_id else None
    if doc and request.current_page and request.current_page > doc["total_pages"]:
        raise HTTPException(422, "Current page is outside this document.")
    conversation_id = request.conversation_id or str(uuid.uuid4())
    with database() as con:
        conversation = con.execute(
            "SELECT * FROM conversations WHERE id=?", (conversation_id,)
        ).fetchone()
        if request.conversation_id and conversation is None:
            raise HTTPException(404, "Conversation not found. Start a new conversation.")
        if conversation and conversation["document_id"] != request.document_id:
            raise HTTPException(409, "This conversation belongs to a different document.")
        if conversation:
            previous = con.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT 6",
                (conversation_id,),
            ).fetchall()
        else:
            previous = []
    history = [
        {"role": row["role"], "content": row["content"][:2000]}
        for row in reversed(previous)
    ]
    selected_text = (request.selected_text or "").strip()[:4000]
    retrieved: list[tuple[int, str]] = []
    if doc:
        retrieved = retrieve_pages(
            doc["id"],
            message,
            request.current_page,
            request.include_current_page,
        )
    context_parts = []
    if selected_text:
        context_parts.append(f"Selected text (highest priority):\n{selected_text}")
    if retrieved:
        context_parts.extend(
            f"Document page {page_number}:\n{text}"
            for page_number, text in retrieved
        )
    context = "\n\n".join(context_parts)
    if doc and not context:
        context = "No matching extractable text was found in the document."
    system_prompt = (
        "You are a patient, beginner-friendly tutor. By default, answer in natural, simple Urdu "
        "written in Urdu script. Urdu and Arabic are different languages: do not answer in Arabic "
        "unless the student explicitly asks for Arabic. If the student explicitly requests another "
        "language, answer in that language. For requests such as 'A1 Urdu', use very simple, "
        "beginner-level Urdu. Keep quoted source text in its original language, but explain it in "
        "the requested response language. Answer the student's question clearly. "
        "Keep answers concise unless the student asks for detail. "
        "When document context is supplied, ground claims about the document in that context, "
        "cite page numbers when useful, and say plainly if the requested detail is not present. "
        "Never pretend a simplified code example is production-ready. "
        "You may explain concepts generally, but distinguish that from document facts."
    )
    if context:
        system_prompt += f"\n\nRetrieved document context (use only what is relevant):\n{context}"
    if doc and not retrieved and not selected_text:
        system_prompt += (
            "\nThe uploaded document has no extractable text for this question. "
            "Explain that OCR may be needed for scanned pages; do not claim to have read it."
        )
    model = (request.model or OLLAMA_MODEL).strip()
    if not model or len(model) > 120:
        raise HTTPException(422, "Invalid Ollama model name.")
    return {
        "conversation_id": conversation_id,
        "conversation": conversation,
        "message": message,
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": message},
        ],
        "context_pages": [page_number for page_number, _ in retrieved],
        "used_selected_text": bool(selected_text),
    }

def save_chat_response(
    conversation_id: str,
    conversation: RowMapping | None,
    request: ChatRequest,
    message: str,
    answer: str,
) -> None:
    now = utc_now()
    with database() as con:
        if conversation is None:
            con.execute(
                "INSERT INTO conversations(id, document_id, title, created_at, updated_at) VALUES(?,?,?,?,?)",
                (conversation_id, request.document_id, message[:80], now, now),
            )
        else:
            con.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )
        con.executemany(
            "INSERT INTO messages(id, conversation_id, role, content, page_number, created_at) VALUES(?,?,?,?,?,?)",
            [
                (str(uuid.uuid4()), conversation_id, "user", message, request.current_page, now),
                (str(uuid.uuid4()), conversation_id, "assistant", answer, request.current_page, now),
            ],
        )

