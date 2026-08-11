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

from app.agent.graph import (
    prompt_for_pending,
    run_intake_turn,
    seed_prior_after_greeting,
    utterance_echoes_assistant,
)
from app.agent.state import IntakeState
from app.config import get_settings
from app.continuity.ava.policy import phone_greeting
from app.db import repository as repo
from app.db.clinic_repo import get_clinic_name
from app.db.session import SessionLocal, init_db
from app.latency import log_latency
from app.logging_setup import get_logger, setup_logging
from app.twilio_app.handoff import is_twilio_call_sid, redirect_call_to_human
from app.voice.stt import get_stt
from app.voice.tts import get_tts

log = get_logger(__name__)

SILENCE_SECONDS = 0.45  # snappy end-of-utterance (was 1.2s — major latency source)
MAX_UTTERANCE_SECONDS = 12.0 #The system will stop recording after 12 seconds of speech
MIN_UTTERANCE_SECONDS = 0.45 #The system will not consider the speech to be valid if it is less than 0.45 seconds
RMS_SPEECH_THRESHOLD = 250.0 #The system will consider the speech to be valid if it is greater than 250.0 decibels
# Barge-in: slightly higher + sustained frames so TTS speaker echo is less likely.
BARGE_IN_RMS = 400.0
BARGE_IN_MIN_FRAMES = 6  # ~120ms at 20ms/frame
BARGE_IN_ECHO_GUARD_S = 0.35  # ignore mic briefly after TTS starts
PARTICIPANT_WAIT_SECONDS = 60.0
TRACK_WAIT_SECONDS = 45.0

# Instant acknowledgements while STT/LLM run (pre-synthesized at call start).
_FILLER_PHRASES = ("Okay.", "Got it.", "Sure.", "Alright.")
_FILLER_CACHE: dict[str, tuple[bytes, int]] = {}  # text -> (pcm_int16, sample_rate)


def _save_transcript_line(call_sid: str, line: str) -> None:
    """Persist one STT/TTS line so the browser UI can poll the live conversation."""
    db = SessionLocal()
    try:
        repo.append_transcript(db, call_sid=call_sid, line=line)
    finally:
        db.close()


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


async def _play_pcm_with_barge_in(
    room: rtc.Room,
    pcm_int16: bytes,
    sample_rate: int,
    mic: MicPump,
    *,
    label: str = "audio",
) -> bool:
    """Play pre-rendered PCM16 mono while watching for barge-in."""
    source = rtc.AudioSource(sample_rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    pub = await room.local_participant.publish_track(track, options)

    barged = asyncio.Event()
    ignore_until = time.monotonic() + BARGE_IN_ECHO_GUARD_S
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
                if consecutive >= BARGE_IN_MIN_FRAMES:
                    log.info("BARGE-IN detected | rms=%.1f — stopping %s", rms, label)
                    mic.push_preroll(frame)
                    barged.set()
                    return
            else:
                consecutive = 0

    watch_task = asyncio.create_task(_watch_barge_in(), name="barge-in-watch")
    samples = np.frombuffer(pcm_int16, dtype=np.int16)
    frame_samples = max(sample_rate // 50, 1)
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
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=frame_samples,
            )
            await source.capture_frame(audio_frame)
            await asyncio.sleep(0.02)
        if not interrupted:
            await asyncio.sleep(0.05)
    finally:
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
        try:
            await room.local_participant.unpublish_track(pub.sid)
        except Exception:
            log.exception("Failed to unpublish %s track", label)

    return interrupted


def _warm_filler_cache() -> None:
    """Pre-synthesize short fillers so acknowledgements are instant."""
    if _FILLER_CACHE:
        return
    tts = get_tts()
    for phrase in _FILLER_PHRASES:
        try:
            result = tts.synthesize(phrase)
            _FILLER_CACHE[phrase] = (result.pcm_int16, result.sample_rate)
            log.info("Warmed filler | phrase=%r duration_bytes=%s", phrase, len(result.pcm_int16))
        except Exception:
            log.exception("Failed to warm filler | phrase=%r", phrase)


def _pick_filler() -> tuple[str, bytes, int] | None:
    if not _FILLER_CACHE:
        return None
    # Rotate by wall clock so consecutive turns don't always sound identical.
    phrases = list(_FILLER_CACHE.keys())
    phrase = phrases[int(time.time()) % len(phrases)]
    pcm, rate = _FILLER_CACHE[phrase]
    return phrase, pcm, rate


async def _publish_latency(room: rtc.Room, **metrics: float) -> None:
    """Send turn latency to browser clients over LiveKit data channel."""
    payload = {
        "type": "latency",
        **{k: round(float(v), 1) for k, v in metrics.items() if v is not None},
    }
    try:
        await room.local_participant.publish_data(
            json.dumps(payload).encode("utf-8"),
            reliable=True,
            topic="latency",
        )
    except Exception:
        log.debug("Failed to publish latency to room", exc_info=True)


async def _publish_tts_with_barge_in(room: rtc.Room, text: str, mic: MicPump) -> bool:
    """
    Play TTS while watching the mic. Returns True if user barged in (interrupted).
    """
    tts = get_tts()
    log.debug("Publishing TTS (barge-in enabled) | text=%r", text)
    t0 = time.perf_counter()
    result = await asyncio.to_thread(tts.synthesize, text)
    synth_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    interrupted = await _play_pcm_with_barge_in(
        room,
        result.pcm_int16,
        result.sample_rate,
        mic,
        label="TTS",
    )
    play_ms = (time.perf_counter() - t1) * 1000.0
    log_latency("tts_synth", synth_ms, chars=len(text))
    log_latency("tts_play", play_ms, chars=len(text))
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


async def _settle_mic(mic: MicPump, *, quiet_s: float = 0.35, timeout_s: float = 1.0) -> None:
    """Drop post-TTS echo before the next listen (common on Twilio handsets)."""
    mic.clear()
    quiet_start: float | None = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = await mic.get_frame(timeout=0.05)
        now = time.monotonic()
        if frame is None:
            if quiet_start is None:
                quiet_start = now
            elif now - quiet_start >= quiet_s:
                break
            continue
        rms = _pcm16_rms(bytes(frame.data))
        if rms < RMS_SPEECH_THRESHOLD:
            if quiet_start is None:
                quiet_start = now
            elif now - quiet_start >= quiet_s:
                break
        else:
            quiet_start = None
    mic.clear()


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
        await asyncio.to_thread(_warm_filler_cache)
        greeting = phone_greeting(get_clinic_name())
        await _publish_tts_with_barge_in(ctx.room, greeting, mic)
        _save_transcript_line(call_sid, f"assistant: {greeting}")
        await _settle_mic(mic)

        prior: IntakeState | None = seed_prior_after_greeting(call_sid, greeting)
        stt = get_stt()
        empty_stt_streak = 0
        max_empty_stt = 3

        while True:
            # Capture already caps at MAX_UTTERANCE_SECONDS — never hangs forever
            pcm = await _capture_utterance(mic)
            if not pcm:
                empty_stt_streak += 1
                log.debug("No speech captured | empty_stt_streak=%s", empty_stt_streak)
                if empty_stt_streak >= max_empty_stt:
                    bye = (
                        "I'll wrap up for now. Our team will follow up if we have "
                        "what we need. Goodbye."
                    )
                    _save_transcript_line(call_sid, f"assistant: {bye}")
                    await _publish_tts_with_barge_in(ctx.room, bye, mic)
                    break
                nudge = (
                    "Sorry, I did not catch that. "
                    + prompt_for_pending((prior or {}).get("pending_field"), prior)
                )
                _save_transcript_line(call_sid, f"assistant: {nudge}")
                await _publish_tts_with_barge_in(ctx.room, nudge, mic)
                await _settle_mic(mic)
                continue

            filler_task: asyncio.Task[bool] | None = None
            try:
                stt_t0 = time.perf_counter()
                transcript = await asyncio.wait_for(
                    asyncio.to_thread(stt.transcribe_pcm16, pcm, sample_rate=16000),
                    timeout=30.0,
                )
                stt_ms = (time.perf_counter() - stt_t0) * 1000.0
                log_latency(
                    "stt",
                    stt_ms,
                    text_len=len(transcript.text or ""),
                )
            except asyncio.TimeoutError:
                log.warning("STT timed out")
                empty_stt_streak += 1
                if empty_stt_streak >= max_empty_stt:
                    bye = (
                        "I'm having trouble hearing you, so I'll end this call for now. "
                        "Please call back when you can. Goodbye."
                    )
                    _save_transcript_line(call_sid, f"assistant: {bye}")
                    await _publish_tts_with_barge_in(ctx.room, bye, mic)
                    await _settle_mic(mic)
                    break
                continue

            if not transcript.text:
                log.warning("STT returned empty text")
                empty_stt_streak += 1
                if empty_stt_streak >= max_empty_stt:
                    bye = "I'll wrap up for now. Please call back anytime. Goodbye."
                    _save_transcript_line(call_sid, f"assistant: {bye}")
                    await _publish_tts_with_barge_in(ctx.room, bye, mic)
                    await _settle_mic(mic)
                    break
                continue

            if utterance_echoes_assistant(transcript.text, prior):
                log.info(
                    "Ignoring echo of prior assistant speech | text=%r",
                    transcript.text,
                )
                continue

            empty_stt_streak = 0
            _save_transcript_line(call_sid, f"user: {transcript.text}")

            # Filler only after a real caller utterance — not on greeting echo.
            filler = _pick_filler()
            if filler is not None:
                phrase, filler_pcm, filler_rate = filler
                log.info("Playing filler while processing | phrase=%r", phrase)
                filler_task = asyncio.create_task(
                    _play_pcm_with_barge_in(
                        ctx.room, filler_pcm, filler_rate, mic, label="filler"
                    ),
                    name="filler-play",
                )

            try:
                llm_t0 = time.perf_counter()
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_intake_turn,
                        call_sid=call_sid,
                        user_text=transcript.text,
                        prior=prior,
                    ),
                    timeout=45.0,
                )
                llm_ms = (time.perf_counter() - llm_t0) * 1000.0
                log_latency("llm_turn", llm_ms)
            except asyncio.TimeoutError:
                log.exception("Intake turn timed out — forcing graceful close")
                llm_ms = (time.perf_counter() - llm_t0) * 1000.0
                result = {
                    **(prior or {}),
                    "reply": (
                        "Thanks. I'll have our team follow up within one business day. "
                        "Goodbye."
                    ),
                    "is_complete": True,
                    "should_end": True,
                    "primary_disposition": "Callback Queue",
                    "handoff_requested": False,
                }
            except Exception:
                log.exception("Intake turn failed — forcing graceful close")
                llm_ms = (time.perf_counter() - llm_t0) * 1000.0
                result = {
                    **(prior or {}),
                    "reply": (
                        "Thanks for the information. Someone from our team will follow up. "
                        "Goodbye."
                    ),
                    "is_complete": True,
                    "should_end": True,
                    "primary_disposition": "Callback Queue",
                    "handoff_requested": False,
                }

            # Let filler finish (or cancel if still going) before main reply.
            if filler_task is not None:
                if not filler_task.done():
                    # Don't cut mid-syllable unless barge-in; wait briefly then cancel.
                    try:
                        await asyncio.wait_for(asyncio.shield(filler_task), timeout=0.8)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        filler_task.cancel()
                        try:
                            await filler_task
                        except asyncio.CancelledError:
                            pass
                else:
                    try:
                        await filler_task
                    except asyncio.CancelledError:
                        pass

            prior = result  # type: ignore[assignment]
            reply_text = (result.get("reply") or "").strip() or (
                "Thanks. Our team will follow up. Goodbye."
            )

            # Synthesize reply in parallel with DB write.
            tts = get_tts()
            synth_task = asyncio.create_task(
                asyncio.to_thread(tts.synthesize, reply_text),
                name="reply-synth",
            )

            db = SessionLocal()
            try:
                status = "in_progress"
                if result.get("handoff_requested"):
                    status = "handoff_pending"
                elif result.get("caller_ended"):
                    status = "paused"
                elif result.get("is_complete") or result.get("should_end"):
                    status = "complete"
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
                repo.append_transcript(
                    db, call_sid=call_sid, line=f"assistant: {reply_text}"
                )
            finally:
                db.close()

            try:
                tts_t0 = time.perf_counter()
                synth = await asyncio.wait_for(synth_task, timeout=45.0)
                tts_synth_ms = (time.perf_counter() - tts_t0) * 1000.0
                log_latency("tts_synth", tts_synth_ms)
                play_t0 = time.perf_counter()
                await asyncio.wait_for(
                    _play_pcm_with_barge_in(
                        ctx.room,
                        synth.pcm_int16,
                        synth.sample_rate,
                        mic,
                        label="reply",
                    ),
                    timeout=45.0,
                )
                tts_play_ms = (time.perf_counter() - play_t0) * 1000.0
                log_latency("tts_play", tts_play_ms)
                await _settle_mic(mic)
                await _publish_latency(
                    ctx.room,
                    stt_ms=stt_ms,
                    llm_ms=llm_ms,
                    tts_synth_ms=tts_synth_ms,
                    tts_play_ms=tts_play_ms,
                    total_ms=stt_ms + llm_ms + tts_synth_ms,
                )
            except asyncio.TimeoutError:
                log.warning("TTS timed out — ending call anyway")
                break
            except Exception:
                log.exception("TTS failed — ending call anyway")
                break

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
