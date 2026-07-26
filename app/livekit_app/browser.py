"""Browser LiveKit session endpoints + quick STT/TTS check."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import repository as repo
from app.db.session import get_db
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


class BrowserSessionResponse(BaseModel):
    call_sid: str
    room_name: str
    token: str
    livekit_url: str
    identity: str


@router.post("/session", response_model=BrowserSessionResponse)
async def start_browser_session(db: Session = Depends(get_db)) -> BrowserSessionResponse:
    """Create a LiveKit room, DB row, mint browser token, dispatch agent."""
    settings = get_settings()
    call_sid = f"browser-{uuid.uuid4().hex[:12]}"
    room_name = make_room_name(call_sid)
    identity = f"browser-user-{uuid.uuid4().hex[:8]}"
    metadata = json.dumps({"call_sid": call_sid, "source": "browser"})

    log.info(
        "Browser session start | call_sid=%s room=%s identity=%s",
        call_sid,
        room_name,
        identity,
    )

    await create_room_with_agent(room_name, metadata=metadata)
    repo.create_call_session(
        db,
        call_sid=call_sid,
        room_name=room_name,
        caller_number="browser",
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
