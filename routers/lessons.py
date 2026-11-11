"""Lessons API routes."""

import datetime as dt
import json
import uuid

from fastapi import APIRouter, HTTPException

from services import GenerateRequest, Lesson, database, ollama_chat

router = APIRouter()

@router.post("/api/generate")
async def generate_legacy_lesson(
    request: "GenerateRequest",
) -> dict:
    if not request.paragraph.strip():
        raise HTTPException(400, "Select a paragraph first.")
    language = request.language
    if language == "English":
        guidance = f"Teach in simple, natural English. Respect level {request.level}."
    elif language == "Urdu":
        guidance = "Teach and explain in clear, simple Urdu using Urdu script."
    elif language == "Kannada":
        guidance = "Teach and explain in clear, simple Kannada using Kannada script."
    else:
        guidance = f"Teach and explain in {language}."
    prompt = f"""You are a careful multilingual language teacher. {guidance}
Create a useful lesson based ONLY on the source paragraph below. Do not invent facts that are not in it. Return valid JSON only, without markdown fences, using this schema:
{{
 "title":"short lesson title",
 "summary":"short explanation",
 "translation":"faithful translation into the teaching language when useful",
 "vocabulary":[{{"term":"word","meaning":"meaning","example":"short example"}}],
 "sentences":[{{"original":"source sentence","explanation":"teaching explanation"}}],
 "grammar":[{{"point":"grammar point","explanation":"simple explanation","example":"example"}}],
 "examples":["example 1","example 2"],
 "practice":[{{"question":"practice question","answer":"model answer"}}],
 "quiz":[{{"question":"question","options":["A","B","C","D"],"answerIndex":0,"explanation":"why"}}]
}}
Create 4-8 vocabulary items, sentence explanations for the key sentences, 2-4 grammar points if applicable, 3 practice items, and 5 quiz questions. If the source is too short, scale down without fabricating. Ensure all learner-facing explanations are in {language}. For English, keep the language appropriate for level {request.level}.
SOURCE PARAGRAPH:
{request.paragraph}"""
    raw = await ollama_chat([{"role": "user", "content": prompt}], request.model)
    try:
        lesson = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(502, "Qwen did not return valid lesson JSON. Try a shorter paragraph.") from exc
    return {"lesson": lesson, "model": request.model}

@router.post("/api/lessons")
def save_lesson(item: Lesson) -> dict:
    with database() as con:
        con.execute(
            """INSERT INTO lessons VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,pdf_name=excluded.pdf_name,
               language=excluded.language,level=excluded.level,page=excluded.page,paragraph=excluded.paragraph,
               lesson_json=excluded.lesson_json,completed=excluded.completed,score=excluded.score""",
            (
                item.id,
                item.createdAt,
                item.updatedAt,
                item.pdfName,
                item.language,
                item.level,
                item.page,
                item.paragraph,
                json.dumps(item.lesson, ensure_ascii=False),
                int(item.completed),
                item.score,
            ),
        )
    return {"saved": True}

@router.get("/api/lessons")
def list_lessons() -> list[dict]:
    with database() as con:
        rows = con.execute("SELECT * FROM lessons ORDER BY updated_at DESC").fetchall()
    return [
        {
            "id": row["id"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "pdfName": row["pdf_name"],
            "language": row["language"],
            "level": row["level"],
            "page": row["page"],
            "paragraph": row["paragraph"],
            "lesson": json.loads(row["lesson_json"] or "{}"),
            "completed": bool(row["completed"]),
            "score": row["score"],
        }
        for row in rows
    ]
