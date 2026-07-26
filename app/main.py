"""FastAPI entrypoint — Twilio webhooks + debug intake chat."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.graph import run_intake_turn
from app.agent.state import IntakeState
from app.config import get_settings
from app.db.models import CallSession
from app.db.session import get_db, init_db
from app.livekit_app.browser import router as browser_router
from app.logging_setup import get_logger, setup_logging
from app.twilio_app.webhooks import router as twilio_router

log = get_logger(__name__)

# In-memory conversation state for /debug/chat (keyed by call_sid).
_DEBUG_STATE: dict[str, IntakeState] = {}
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    log.info("Starting FastAPI | env=%s log_level=%s", settings.app_env, settings.log_level)
    init_db()
    log.debug(
        "Config snapshot | model=%s stt=%s tts_voice=%s db=%s livekit=%s",
        settings.gemini_model,
        settings.stt_model_size,
        settings.tts_voice,
        settings.database_url,
        settings.livekit_url,
    )
    yield
    log.info("Shutting down FastAPI")


app = FastAPI(
    title="Healthcare Voice Intake",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(twilio_router)
app.include_router(browser_router)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def browser_test_page() -> FileResponse:
    log.debug("GET / → static/index.html")
    return FileResponse(_STATIC_DIR / "index.html")



class DebugChatRequest(BaseModel):
    call_sid: str = Field(..., description="Synthetic or real CallSid key")
    text: str = Field(..., min_length=1)
    reset: bool = False


class DebugChatResponse(BaseModel):
    reply: str
    patient_name: str | None
    patient_age: int | None
    ready_to_proceed: bool | None
    diseases: str | None
    medications: str | None
    pending_field: str | None
    is_complete: bool
    should_end: bool
    validation_notes: str


@app.get("/health")
def health() -> dict[str, str]:
    log.debug("GET /health")
    return {"status": "ok"}


@app.post("/debug/chat", response_model=DebugChatResponse)
def debug_chat(body: DebugChatRequest, db: Session = Depends(get_db)) -> DebugChatResponse:
    """Text-only path to exercise LangGraph + SQLite without Twilio/LiveKit."""
    log.info(
        "POST /debug/chat | call_sid=%s reset=%s text=%r",
        body.call_sid,
        body.reset,
        body.text,
    )

    if body.reset and body.call_sid in _DEBUG_STATE:
        log.debug("Resetting debug state for call_sid=%s", body.call_sid)
        del _DEBUG_STATE[body.call_sid]

    # Ensure a DB row exists for debug sessions.
    existing = db.query(CallSession).filter(CallSession.call_sid == body.call_sid).one_or_none()
    if existing is None:
        log.debug("Creating debug CallSession row | call_sid=%s", body.call_sid)
        db.add(
            CallSession(
                call_sid=body.call_sid,
                room_name=f"debug-{body.call_sid}",
                caller_number="debug",
                status="in_progress",
            )
        )
        db.commit()

    prior = None if body.reset else _DEBUG_STATE.get(body.call_sid)
    result = run_intake_turn(call_sid=body.call_sid, user_text=body.text, prior=prior)
    _DEBUG_STATE[body.call_sid] = result

    from app.db import repository as repo

    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in (result.get("messages") or [])
    )
    if result.get("is_complete"):
        status = "complete"
    elif result.get("ready_to_proceed") is False:
        status = "not_ready"
    else:
        status = "in_progress"
    repo.update_intake(
        db,
        call_sid=body.call_sid,
        patient_name=result.get("patient_name"),
        patient_age=result.get("patient_age"),
        ready_to_proceed=result.get("ready_to_proceed"),
        diseases=result.get("diseases"),
        medications=result.get("medications"),
        transcript=transcript,
        status=status,
    )

    return DebugChatResponse(
        reply=result["reply"],
        patient_name=result.get("patient_name"),
        patient_age=result.get("patient_age"),
        ready_to_proceed=result.get("ready_to_proceed"),
        diseases=result.get("diseases"),
        medications=result.get("medications"),
        pending_field=result.get("pending_field"),
        is_complete=result["is_complete"],
        should_end=result.get("should_end", False),
        validation_notes=result.get("validation_notes") or "",
    )


@app.get("/calls/{call_sid}")
def get_call(call_sid: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    log.debug("GET /calls/%s", call_sid)
    row = db.query(CallSession).filter(CallSession.call_sid == call_sid).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="call not found")
    return {
        "id": row.id,
        "call_sid": row.call_sid,
        "room_name": row.room_name,
        "caller_number": row.caller_number,
        "status": row.status,
        "patient_name": row.patient_name,
        "patient_age": row.patient_age,
        "ready_to_proceed": row.ready_to_proceed,
        "diseases": row.diseases,
        "medications": row.medications,
        "transcript": row.transcript,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
