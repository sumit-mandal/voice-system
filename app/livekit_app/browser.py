"""Browser LiveKit session endpoints + text chat + quick STT/TTS check."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.graph import run_intake_turn, seed_prior_after_greeting
from app.agent.state import IntakeState
from app.config import get_settings
from app.continuity.ava.policy import chat_greeting
from app.db import continuity_repo as crepo
from app.db import repository as repo
from app.db.clinic_repo import get_clinic_name
from app.db.session import SessionLocal, get_db
from app.livekit_app.rooms import (
    create_participant_token,
    create_room_with_agent,
    make_room_name,
)
from app.logging_setup import get_logger
from app.voice.stt import get_stt
from app.voice.tts import get_tts

log = get_logger(__name__)
router = APIRouter(prefix="/browser", tags=["browser"])

# In-memory Ava state for browser text chat (keyed by call_sid).
_CHAT_STATE: dict[str, IntakeState] = {}


class BrowserSessionResponse(BaseModel):
    call_sid: str
    room_name: str
    token: str
    livekit_url: str
    identity: str


class BrowserSessionRequest(BaseModel):
    phone: str | None = None


class BrowserChatStartRequest(BaseModel):
    phone: str | None = None


class BrowserChatStartResponse(BaseModel):
    call_sid: str
    greeting: str
    status: str = "in_progress"
    pending_field: str | None = None


class BrowserChatMessageRequest(BaseModel):
    call_sid: str
    text: str = Field(..., min_length=1)


class BrowserChatMessageResponse(BaseModel):
    call_sid: str
    reply: str
    status: str
    pending_field: str | None = None
    is_complete: bool = False
    should_end: bool = False
    handoff_requested: bool = False
    caller_name: str | None = None
    child_first_name: str | None = None
    primary_disposition: str | None = None
    transcript: str = ""


@router.post("/session", response_model=BrowserSessionResponse)
async def start_browser_session(
    body: BrowserSessionRequest | None = None,
    db: Session = Depends(get_db),
) -> BrowserSessionResponse:
    """Create a LiveKit room, DB row, mint browser token, dispatch agent."""
    settings = get_settings()
    call_sid = f"browser-{uuid.uuid4().hex[:12]}"
    room_name = make_room_name(call_sid)
    identity = f"browser-user-{uuid.uuid4().hex[:8]}"
    phone = (body.phone if body else None) or "browser"
    user = crepo.find_or_create_user_by_phone(db, phone if phone != "browser" else None)
    user_id = user.id if user else None
    metadata = json.dumps(
        {"call_sid": call_sid, "source": "browser", "user_id": user_id}
    )

    log.info(
        "Browser session start | call_sid=%s room=%s identity=%s user_id=%s",
        call_sid,
        room_name,
        identity,
        user_id,
    )

    await create_room_with_agent(room_name, metadata=metadata)
    repo.create_call_session(
        db,
        call_sid=call_sid,
        room_name=room_name,
        caller_number=phone,
        user_id=user_id,
    )
    token = create_participant_token(
        room_name=room_name,
        identity=identity,
        name="Browser Caller",
    )

    livekit_url = _browser_livekit_url(settings)
    log.debug("Browser session ready | url=%s token_len=%s", livekit_url, len(token))
    return BrowserSessionResponse(
        call_sid=call_sid,
        room_name=room_name,
        token=token,
        livekit_url=livekit_url,
        identity=identity,
    )


def _browser_livekit_url(settings) -> str:
    """URL the browser uses for LiveKit signaling (must be reachable from the client)."""
    if settings.livekit_public_url:
        return settings.livekit_public_url.rstrip("/")
    base = settings.public_base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base.removeprefix("https://")
    if base.startswith("http://"):
        return "ws://" + base.removeprefix("http://")
    # Local-only fallback for laptop dev.
    return settings.livekit_url.replace("localhost", "127.0.0.1")


def _chat_status(result: IntakeState) -> str:
    if result.get("handoff_requested"):
        return "handoff_pending"
    if result.get("is_complete") or result.get("should_end"):
        return "complete"
    return "in_progress"


def _transcript_from_state(result: IntakeState) -> str:
    return "\n".join(
        f"{m['role']}: {m['content']}" for m in (result.get("messages") or [])
    )


@router.post("/chat/start", response_model=BrowserChatStartResponse)
async def start_browser_chat(
    body: BrowserChatStartRequest | None = None,
    db: Session = Depends(get_db),
) -> BrowserChatStartResponse:
    """Start a text-only Ava intake session (no LiveKit / mic required)."""
    call_sid = f"chat-{uuid.uuid4().hex[:12]}"
    phone = (body.phone if body else None) or "browser-chat"
    user = crepo.find_or_create_user_by_phone(db, phone if phone != "browser-chat" else None)
    user_id = user.id if user else None
    greeting = chat_greeting(get_clinic_name())

    repo.create_call_session(
        db,
        call_sid=call_sid,
        room_name=f"chat-{call_sid}",
        caller_number=phone,
        user_id=user_id,
    )
    prior = seed_prior_after_greeting(call_sid, greeting)
    _CHAT_STATE[call_sid] = prior
    repo.append_transcript(db, call_sid=call_sid, line=f"assistant: {greeting}")

    log.info("Browser chat start | call_sid=%s user_id=%s", call_sid, user_id)
    return BrowserChatStartResponse(
        call_sid=call_sid,
        greeting=greeting,
        status="in_progress",
        pending_field=prior.get("pending_field"),
    )


@router.post("/chat/message", response_model=BrowserChatMessageResponse)
async def browser_chat_message(
    body: BrowserChatMessageRequest,
    db: Session = Depends(get_db),
) -> BrowserChatMessageResponse:
    """Send one text turn to Ava and return the reply + updated transcript."""
    call_sid = body.call_sid.strip()
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    session = repo.get_by_call_sid(db, call_sid)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")

    prior = _CHAT_STATE.get(call_sid)
    if prior is None:
        # Recover after API restart: continue from a fresh seeded prior.
        prior = seed_prior_after_greeting(call_sid)
        _CHAT_STATE[call_sid] = prior

    repo.append_transcript(db, call_sid=call_sid, line=f"user: {text}")
    result = run_intake_turn(call_sid=call_sid, user_text=text, prior=prior)
    _CHAT_STATE[call_sid] = result

    reply = (result.get("reply") or "").strip() or "Thanks. Our team will follow up."
    status = _chat_status(result)
    repo.append_transcript(db, call_sid=call_sid, line=f"assistant: {reply}")
    repo.update_intake(
        db,
        call_sid=call_sid,
        patient_name=result.get("patient_name") or result.get("caller_name"),
        patient_age=result.get("patient_age"),
        ready_to_proceed=result.get("ready_to_proceed"),
        diseases=result.get("diseases") or result.get("diagnosis_stated"),
        medications=result.get("medications") or result.get("insurance_carrier"),
        transcript=None,
        status=status,
        handoff_reason=result.get("handoff_reason") or None,
        handoff_summary=result.get("handoff_summary") or None,
    )

    # Prefer DB transcript (includes greeting) over in-memory messages alone.
    row = repo.get_by_call_sid(db, call_sid)
    transcript = (row.transcript if row else None) or _transcript_from_state(result)

    log.info(
        "Browser chat turn | call_sid=%s complete=%s pending=%s",
        call_sid,
        result.get("is_complete"),
        result.get("pending_field"),
    )
    return BrowserChatMessageResponse(
        call_sid=call_sid,
        reply=reply,
        status=status,
        pending_field=result.get("pending_field"),
        is_complete=bool(result.get("is_complete")),
        should_end=bool(result.get("should_end")),
        handoff_requested=bool(result.get("handoff_requested")),
        caller_name=result.get("caller_name"),
        child_first_name=result.get("child_first_name"),
        primary_disposition=result.get("primary_disposition"),
        transcript=transcript,
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _chunk_reply_for_stream(reply: str, *, max_chars: int = 12) -> list[str]:
    """Split a final reply into small chunks for progressive UI streaming."""
    text = (reply or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    buf = ""
    for word in text.split(" "):
        piece = word if not buf else f" {word}"
        if buf and len(buf) + len(piece) > max_chars:
            chunks.append(buf)
            buf = word
        else:
            buf += piece
    if buf:
        chunks.append(buf)
    return chunks


@router.post("/chat/message/stream")
async def browser_chat_message_stream(
    body: BrowserChatMessageRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    One consistent Ava reply, streamed to the UI as SSE tokens.

    Events: status | token | done | error
    """
    call_sid = body.call_sid.strip()
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    session = repo.get_by_call_sid(db, call_sid)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")

    prior = _CHAT_STATE.get(call_sid)
    if prior is None:
        prior = seed_prior_after_greeting(call_sid)
        _CHAT_STATE[call_sid] = prior

    repo.append_transcript(db, call_sid=call_sid, line=f"user: {text}")

    async def event_gen():
        try:
            # Status only — never shown as an assistant message bubble.
            yield _sse({"type": "status", "text": "Ava is typing…"})

            result = await asyncio.to_thread(
                run_intake_turn,
                call_sid=call_sid,
                user_text=text,
                prior=prior,
            )
            _CHAT_STATE[call_sid] = result

            final_reply = (
                (result.get("reply") or "").strip()
                or "Thanks. Our team will follow up."
            )
            status = _chat_status(result)

            for chunk in _chunk_reply_for_stream(final_reply):
                yield _sse({"type": "token", "text": chunk})
                await asyncio.sleep(0.025)

            db2 = SessionLocal()
            try:
                repo.append_transcript(
                    db2, call_sid=call_sid, line=f"assistant: {final_reply}"
                )
                repo.update_intake(
                    db2,
                    call_sid=call_sid,
                    patient_name=result.get("patient_name") or result.get("caller_name"),
                    patient_age=result.get("patient_age"),
                    ready_to_proceed=result.get("ready_to_proceed"),
                    diseases=result.get("diseases") or result.get("diagnosis_stated"),
                    medications=result.get("medications")
                    or result.get("insurance_carrier"),
                    transcript=None,
                    status=status,
                    handoff_reason=result.get("handoff_reason") or None,
                    handoff_summary=result.get("handoff_summary") or None,
                )
                row = repo.get_by_call_sid(db2, call_sid)
                transcript = (row.transcript if row else None) or _transcript_from_state(
                    result
                )
            finally:
                db2.close()

            yield _sse(
                {
                    "type": "done",
                    "call_sid": call_sid,
                    "reply": final_reply,
                    "status": status,
                    "pending_field": result.get("pending_field"),
                    "is_complete": bool(result.get("is_complete")),
                    "should_end": bool(result.get("should_end")),
                    "handoff_requested": bool(result.get("handoff_requested")),
                    "caller_name": result.get("caller_name"),
                    "child_first_name": result.get("child_first_name"),
                    "primary_disposition": result.get("primary_disposition"),
                    "transcript": transcript,
                }
            )
        except Exception as exc:
            log.exception("Chat stream failed | call_sid=%s", call_sid)
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/stt-tts")
async def quick_stt_tts(audio: UploadFile = File(...)) -> Response:
    """
    Isolated STT → TTS loop (no LLM).
    Upload WAV/PCM16 mono; get WAV back of what the agent would speak
    (echo of transcript for verification).
    """
    raw = await audio.read()
    log.info(
        "Quick STT/TTS | filename=%s content_type=%s bytes=%s",
        audio.filename,
        audio.content_type,
        len(raw),
    )
    if not raw:
        return Response(content=b"empty audio", status_code=400)

    # Expect WAV or raw PCM16 @ 16k. Prefer WAV via soundfile.
    import io

    import numpy as np
    import soundfile as sf

    try:
        samples, rate = sf.read(io.BytesIO(raw), dtype="float32")
        log.debug("Decoded upload | rate=%s shape=%s", rate, getattr(samples, "shape", None))
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if rate != 16000:
            duration = len(samples) / rate
            target_len = int(duration * 16000)
            x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=max(target_len, 1), endpoint=False)
            samples = np.interp(x_new, x_old, samples)
            rate = 16000
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    except Exception:
        log.debug("soundfile decode failed — treating body as raw PCM16 @16k")
        pcm = raw
        rate = 16000

    stt = get_stt()
    result = stt.transcribe_pcm16(pcm, sample_rate=rate)
    text = result.text or "(no speech detected)"
    log.info("Quick STT text=%r", text)

    speak = f"I heard you say: {text}"
    tts = get_tts()
    synth = tts.synthesize(speak)
    log.info("Quick TTS wav_bytes=%s", len(synth.wav_bytes))
    return Response(
        content=synth.wav_bytes,
        media_type="audio/wav",
        headers={"X-Transcript": text.replace("\n", " ")[:500]},
    )
