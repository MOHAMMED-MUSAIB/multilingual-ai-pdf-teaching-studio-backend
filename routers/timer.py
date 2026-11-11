"""Timer API routes."""

import datetime as dt
import uuid

from fastapi import APIRouter, HTTPException, Query

from services import TimerSession, TimerSettings, database, utc_now

router = APIRouter()

@router.get("/api/timer/settings")
def get_timer_settings() -> dict:
    with database() as con:
        row = con.execute(
            "SELECT focus_minutes, break_minutes FROM timer_settings WHERE id=1"
        ).fetchone()
    return dict(row)

@router.put("/api/timer/settings")
def save_timer_settings(settings: TimerSettings) -> dict:
    with database() as con:
        con.execute(
            "UPDATE timer_settings SET focus_minutes=?, break_minutes=? WHERE id=1",
            (settings.focus_minutes, settings.break_minutes),
        )
    return settings.model_dump()

@router.post("/api/timer/sessions")
def save_timer_session(session: TimerSession) -> dict:
    now = utc_now()
    started_at = session.started_at or now
    try:
        dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, "Invalid session start time.") from exc
    session_id = str(uuid.uuid4())
    with database() as con:
        con.execute(
            "INSERT INTO focus_sessions(id, mode, duration_seconds, completed, started_at, completed_at) VALUES(?,?,?,?,?,?)",
            (session_id, session.mode, session.duration_seconds, int(session.completed), started_at, now),
        )
    return {"id": session_id, "saved": True}

@router.get("/api/timer/sessions")
def list_timer_sessions(limit: int = Query(default=100, ge=1, le=500)) -> list[dict]:
    with database() as con:
        rows = con.execute(
            "SELECT id, mode, duration_seconds, completed, started_at, completed_at FROM focus_sessions ORDER BY completed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{**dict(row), "completed": bool(row["completed"])} for row in rows]
