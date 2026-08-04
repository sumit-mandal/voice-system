"""LiveKit agent worker — STT → LangGraph → PocketTTS with barge-in VAD."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from typing import Any

import numpy as np
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

from app.agent.graph import run_intake_turn, seed_prior_after_greeting
from app.agent.state import IntakeState
from app.config import get_settings
from app.db import repository as repo
from app.db.clinic_repo import get_clinic_name
from app.db.session import SessionLocal, init_db
from app.logging_setup import get_logger, setup_logging
from app.twilio_app.handoff import is_twilio_call_sid, redirect_call_to_human
from app.voice.stt import get_stt
from app.voice.tts import get_tts
from app.continuity.ava.policy import phone_greeting

log = get_logger(__name__)

SILENCE_SECONDS = 1.2 #After 1.2 seconds of silence, the system assumes the user has finished speaking
MAX_UTTERANCE_SECONDS = 12.0 #The system will stop recording after 12 seconds of speech
MIN_UTTERANCE_SECONDS = 0.45 #The system will not consider the speech to be valid if it is less than 0.45 seconds
RMS_SPEECH_THRESHOLD = 250.0 #The system will consider the speech to be valid if it is greater than 250.0 decibels
# Barge-in: slightly higher + sustained frames so TTS speaker echo is less likely.
BARGE_IN_RMS = 400.0
BARGE_IN_MIN_FRAMES = 6  # ~120ms at 20ms/frame
BARGE_IN_ECHO_GUARD_S = 0.35  # ignore mic briefly after TTS starts
PARTICIPANT_WAIT_SECONDS = 60.0
TRACK_WAIT_SECONDS = 45.0


def _pcm16_rms(frame: bytes) -> float:
    if len(frame) < 2:
        return 0.0
    arr = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


class MicPump:
    """Single consumer of LiveKit AudioStream; fans frames into a queue for VAD/TTS barge-in."""

    def __init__(self, audio_stream: rtc.AudioStream) -> None:
        self._stream = audio_stream
        self._queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue(maxsize=500)
        self._task: asyncio.Task[None] | None = None
        self._preroll: deque[rtc.AudioFrame] = deque(maxlen=50)
        self.last_sample_rate = 16000
        self.last_num_channels = 1

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="mic-pump")
        log.debug("MicPump started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.debug("MicPump stopped")

    async def _run(self) -> None:
        try:
            async for event in self._stream:
                frame: rtc.AudioFrame = event.frame
                self.last_sample_rate = frame.sample_rate
                self.last_num_channels = frame.num_channels
                try:
                    self._queue.put_nowait(frame)
                except asyncio.QueueFull:
                    # Drop oldest by getting one, then put.
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        self._queue.put_nowait(frame)
                    except asyncio.QueueFull:
                        log.warning("MicPump queue full — dropping frame")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("MicPump crashed")
            raise
        finally:
            await self._queue.put(None)

    def clear(self) -> None:
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        self._preroll.clear()
        if cleared:
            log.debug("MicPump cleared %s queued frames", cleared)

    def push_preroll(self, frame: rtc.AudioFrame) -> None:
        self._preroll.append(frame)

    async def get_frame(self, timeout: float = 0.05) -> rtc.AudioFrame | None:
        if self._preroll:
            return self._preroll.popleft()
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return item


async def _publish_tts_with_barge_in(room: rtc.Room, text: str, mic: MicPump) -> bool:
    """
    Play TTS while watching the mic. Returns True if user barged in (interrupted).
    """
    tts = get_tts()
    log.debug("Publishing TTS (barge-in enabled) | text=%r", text)
    result = await asyncio.to_thread(tts.synthesize, text)

    source = rtc.AudioSource(result.sample_rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    pub = await room.local_participant.publish_track(track, options)
    log.debug("Audio track published | sid=%s", pub.sid)

    barged = asyncio.Event()
    ignore_until = time.monotonic() + BARGE_IN_ECHO_GUARD_S
    # Discard leftover mic audio so we don't false-trigger on prior speech.
    mic.clear()

    async def _watch_barge_in() -> None:
        consecutive = 0
        while not barged.is_set():
            frame = await mic.get_frame(timeout=0.05)
            if frame is None:
                continue
            if time.monotonic() < ignore_until:
                continue
            pcm = bytes(frame.data)
            rms = _pcm16_rms(pcm)
            if rms >= BARGE_IN_RMS:
                consecutive += 1
                log.debug(
                    "Barge-in candidate | rms=%.1f consecutive=%s/%s",
                    rms,
                    consecutive,
                    BARGE_IN_MIN_FRAMES,
                )
                if consecutive >= BARGE_IN_MIN_FRAMES:
                    log.info("BARGE-IN detected | rms=%.1f — stopping TTS", rms)
                    mic.push_preroll(frame)
                    barged.set()
                    return
            else:
                consecutive = 0

    watch_task = asyncio.create_task(_watch_barge_in(), name="barge-in-watch")
    samples = np.frombuffer(result.pcm_int16, dtype=np.int16)
    frame_samples = max(result.sample_rate // 50, 1)
    interrupted = False
    try:
        for i in range(0, len(samples), frame_samples):
            if barged.is_set():
                interrupted = True
                break
            chunk = samples[i : i + frame_samples]
            if len(chunk) == 0:
                break
            if len(chunk) < frame_samples:
                pad = np.zeros(frame_samples - len(chunk), dtype=np.int16)
                chunk = np.concatenate([chunk, pad])
            audio_frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=result.sample_rate,
                num_channels=1,
                samples_per_channel=frame_samples,
            )
            await source.capture_frame(audio_frame)
            await asyncio.sleep(0.02)
        if not interrupted:
            await asyncio.sleep(0.1)
    finally:
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
        try:
            await room.local_participant.unpublish_track(pub.sid)
        except Exception:
            log.exception("Failed to unpublish TTS track")

    if interrupted:
        log.info("TTS interrupted by user | chars=%s", len(text))
    else:
        log.info("TTS playback finished | chars=%s", len(text))
    return interrupted


async def _capture_utterance(mic: MicPump, *, target_rate: int = 16000) -> bytes:
    """Collect PCM16 mono until silence after speech, reading from MicPump queue."""
    log.debug(
        "Capturing utterance | silence=%.2fs max=%.2fs threshold_rms=%.1f",
        SILENCE_SECONDS,
        MAX_UTTERANCE_SECONDS,
        RMS_SPEECH_THRESHOLD,
    )
    started = time.monotonic()
    speech_started = False
    last_voice = time.monotonic()
    chunks: list[bytes] = []
    sample_rate = mic.last_sample_rate
    num_channels = mic.last_num_channels

    while True:
        frame = await mic.get_frame(timeout=0.1)
        now = time.monotonic()
        if frame is None:
            if speech_started and now - last_voice >= SILENCE_SECONDS:
                log.debug("Silence end-of-utterance (idle) | silence=%.2f", now - last_voice)
                break
            if now - started >= MAX_UTTERANCE_SECONDS:
                log.warning("Utterance hit MAX_UTTERANCE_SECONDS=%.1f", MAX_UTTERANCE_SECONDS)
                break
            continue

        sample_rate = frame.sample_rate
        num_channels = frame.num_channels
        pcm = bytes(frame.data)
        rms = _pcm16_rms(pcm)

        if rms >= RMS_SPEECH_THRESHOLD:
            if not speech_started:
                log.debug("Speech detected | rms=%.1f t=%.2f", rms, now - started)
            speech_started = True
            last_voice = now
            chunks.append(pcm)
        elif speech_started:
            chunks.append(pcm)
            if now - last_voice >= SILENCE_SECONDS:
                log.debug(
                    "Silence end-of-utterance | silence=%.2f chunks=%s",
                    now - last_voice,
                    len(chunks),
                )
                break

        if now - started >= MAX_UTTERANCE_SECONDS:
            log.warning("Utterance hit MAX_UTTERANCE_SECONDS=%.1f", MAX_UTTERANCE_SECONDS)
            break

    pcm_bytes = b"".join(chunks)
    duration = len(pcm_bytes) / 2 / max(sample_rate, 1) / max(num_channels, 1) if chunks else 0.0
    log.info(
        "Utterance captured | bytes=%s approx_duration_s=%.2f speech_started=%s",
        len(pcm_bytes),
        duration,
        speech_started,
    )
    if not speech_started or duration < MIN_UTTERANCE_SECONDS:
        log.debug("Utterance too short / no speech — returning empty")
        return b""

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1)
    if sample_rate != target_rate:
        log.debug("Resampling utterance | %s → %s", sample_rate, target_rate)
        duration_s = len(samples) / sample_rate
        target_len = int(duration_s * target_rate)
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=max(target_len, 1), endpoint=False)
        samples = np.interp(x_new, x_old, samples)
    return samples.astype(np.int16).tobytes()


def _load_metadata(ctx: JobContext) -> dict[str, Any]:
    raw = ctx.job.metadata or "{}"
    log.debug("Job metadata raw=%r", raw)
    data = json.loads(raw)
    log.info("Job metadata parsed | %s", data)
    return data


async def entrypoint(ctx: JobContext) -> None:
    setup_logging(get_settings().log_level)
    init_db()
    meta = _load_metadata(ctx)
    call_sid = meta["call_sid"]
    room_name = ctx.room.name
    log.info("Agent entrypoint | room=%s call_sid=%s", room_name, call_sid)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    log.debug("Connected to LiveKit room | participants=%s", len(ctx.room.remote_participants))

    deadline = time.monotonic() + PARTICIPANT_WAIT_SECONDS
    participant = None
    while time.monotonic() < deadline:
        for p in ctx.room.remote_participants.values():
            if not p.identity.startswith("agent"):
                participant = p
                break
        if participant is not None:
            break
        await asyncio.sleep(0.1)
    if participant is None:
        raise RuntimeError(f"No remote participants joined room={room_name}")

    log.info(
        "Using remote participant | identity=%s sid=%s",
        participant.identity,
        participant.sid,
    )

    track: rtc.RemoteAudioTrack | None = None
    for publication in participant.track_publications.values():
        log.debug(
            "Remote publication | sid=%s kind=%s subscribed=%s",
            publication.sid,
            publication.kind,
            publication.subscribed,
        )
        if publication.track and publication.kind == rtc.TrackKind.KIND_AUDIO:
            track = publication.track  # type: ignore[assignment]
            break

    if track is None:
        log.debug("Waiting for remote audio track subscription…")
        ready = asyncio.Event()

        @ctx.room.on("track_subscribed")
        def _on_track_subscribed(
            t: rtc.Track,
            _pub: rtc.RemoteTrackPublication,
            p: rtc.RemoteParticipant,
        ) -> None:
            nonlocal track
            log.debug("track_subscribed | kind=%s participant=%s", t.kind, p.identity)
            if t.kind == rtc.TrackKind.KIND_AUDIO and p.identity == participant.identity:
                track = t  # type: ignore[assignment]
                ready.set()

        await asyncio.wait_for(ready.wait(), timeout=TRACK_WAIT_SECONDS)

    assert track is not None
    audio_stream = rtc.AudioStream(track)
    mic = MicPump(audio_stream)
    await mic.start()
    log.debug("AudioStream + MicPump attached (barge-in enabled)")

    try:
        greeting = phone_greeting(get_clinic_name())
        await _publish_tts_with_barge_in(ctx.room, greeting, mic)
        db = SessionLocal()
        try:
            repo.append_transcript(db, call_sid=call_sid, line=f"assistant: {greeting}")
        finally:
            db.close()

        prior: IntakeState | None = seed_prior_after_greeting(call_sid)
        stt = get_stt()

        while True:
            pcm = await _capture_utterance(mic)
            if not pcm:
                log.debug("No speech captured — prompting again")
                await _publish_tts_with_barge_in(
                    ctx.room,
                    "Sorry, I did not catch that. Please continue.",
                    mic,
                )
                continue

            transcript = await asyncio.to_thread(stt.transcribe_pcm16, pcm, sample_rate=16000)
            if not transcript.text:
                log.warning("STT returned empty text")
                await _publish_tts_with_barge_in(
                    ctx.room,
                    "I could not understand. Could you please repeat?",
                    mic,
                )
                continue

            db = SessionLocal()
            try:
                repo.append_transcript(db, call_sid=call_sid, line=f"user: {transcript.text}")
            finally:
                db.close()

            result = await asyncio.to_thread(
                run_intake_turn,
                call_sid=call_sid,
                user_text=transcript.text,
                prior=prior,
            )
            prior = result

            db = SessionLocal()
            try:
                repo.append_transcript(
                    db, call_sid=call_sid, line=f"assistant: {result['reply']}"
                )
                status = "in_progress"
                if result.get("handoff_requested"):
                    status = "handoff_pending"
                elif result.get("is_complete"):
                    status = "complete"
                elif result.get("ready_to_proceed") is False:
                    status = "not_ready"
                repo.update_intake(
                    db,
                    call_sid=call_sid,
                    patient_name=result.get("patient_name"),
                    patient_age=result.get("patient_age"),
                    ready_to_proceed=result.get("ready_to_proceed"),
                    diseases=result.get("diseases"),
                    medications=result.get("medications"),
                    transcript=None,
                    status=status,
                    handoff_reason=result.get("handoff_reason") or None,
                    handoff_summary=result.get("handoff_summary") or None,
                )
            finally:
                db.close()

            await _publish_tts_with_barge_in(ctx.room, result["reply"], mic)

            if result.get("handoff_requested"):
                log.info(
                    "Handoff requested | call_sid=%s reason=%r",
                    call_sid,
                    result.get("handoff_reason"),
                )
                if is_twilio_call_sid(call_sid):
                    try:
                        await asyncio.to_thread(redirect_call_to_human, call_sid=call_sid)
                        log.info("Twilio redirect to human queued | call_sid=%s", call_sid)
                    except Exception:
                        log.exception(
                            "Failed to redirect Twilio call to human | call_sid=%s",
                            call_sid,
                        )
                        await _publish_tts_with_barge_in(
                            ctx.room,
                            "I'm sorry, I could not reach a human agent right now. "
                            "Please try again later.",
                            mic,
                        )
                else:
                    log.info(
                        "Handoff flagged but not a Twilio CallSid "
                        "(browser/debug) — ending agent only | call_sid=%s",
                        call_sid,
                    )
                await asyncio.sleep(0.5)
                break

            if result.get("is_complete") or result.get("should_end"):
                log.info(
                    "Ending agent loop | call_sid=%s complete=%s should_end=%s",
                    call_sid,
                    result.get("is_complete"),
                    result.get("should_end"),
                )
                await asyncio.sleep(0.5)
                break
    finally:
        await mic.stop()

    log.info("Agent entrypoint finished | room=%s", room_name)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    settings = get_settings()
    setup_logging(settings.log_level)

    os.environ["LIVEKIT_URL"] = settings.livekit_url
    os.environ["LIVEKIT_API_KEY"] = settings.livekit_api_key
    os.environ["LIVEKIT_API_SECRET"] = settings.livekit_api_secret

    log.info(
        "Starting LiveKit worker | agent_name=healthcare-intake url=%s",
        settings.livekit_url,
    )
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="healthcare-intake",
            ws_url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        )
    )
