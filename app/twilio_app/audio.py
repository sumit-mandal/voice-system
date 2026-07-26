"""Twilio Media Stream audio helpers (mulaw 8kHz ↔ PCM16)."""

from __future__ import annotations

import audioop

import numpy as np

from app.logging_setup import get_logger

log = get_logger(__name__)

TWILIO_RATE = 8000


def mulaw_b64_chunk_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Decode Twilio μ-law payload to PCM16 @ 8kHz mono."""
    return audioop.ulaw2lin(mulaw_bytes, 2)


def pcm16_8k_to_mulaw(pcm16: bytes) -> bytes:
    return audioop.lin2ulaw(pcm16, 2)


def resample_pcm16(pcm16: bytes, *, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm16
    converted, _ = audioop.ratecv(pcm16, 2, 1, src_rate, dst_rate, None)
    return converted


def pcm16_rms(pcm16: bytes) -> float:
    if len(pcm16) < 2:
        return 0.0
    arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def tts_pcm_to_twilio_mulaw(pcm16: bytes, *, sample_rate: int) -> bytes:
    """PocketTTS PCM → Twilio μ-law 8kHz."""
    pcm_8k = resample_pcm16(pcm16, src_rate=sample_rate, dst_rate=TWILIO_RATE)
    mulaw = pcm16_8k_to_mulaw(pcm_8k)
    log.debug(
        "TTS→Twilio | src_rate=%s pcm=%s mulaw=%s",
        sample_rate,
        len(pcm16),
        len(mulaw),
    )
    return mulaw


def twilio_mulaw_to_whisper_pcm16(mulaw_chunks: bytes) -> bytes:
    """Concatenate μ-law @8k → PCM16 @16k for faster-whisper."""
    pcm_8k = mulaw_b64_chunk_to_pcm16(mulaw_chunks)
    pcm_16k = resample_pcm16(pcm_8k, src_rate=TWILIO_RATE, dst_rate=16000)
    log.debug("Twilio→STT | mulaw=%s pcm16k=%s", len(mulaw_chunks), len(pcm_16k))
    return pcm_16k
