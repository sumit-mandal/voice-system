"""faster-whisper STT wrapper with debug logging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from app.config import get_settings
from app.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class TranscriptResult:
    text: str
    language: str | None
    duration: float | None


class SpeechToText:
    def __init__(self) -> None:
        settings = get_settings()
        log.debug(
            "Loading faster-whisper | model=%s device=%s compute_type=%s",
            settings.stt_model_size,
            settings.stt_device,
            settings.stt_compute_type,
        )
        self._model = WhisperModel(
            settings.stt_model_size,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
        )
        log.info("STT model loaded | model=%s", settings.stt_model_size)

    def transcribe_pcm16(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int = 16000,
    ) -> TranscriptResult:
        log.debug(
            "STT transcribe_pcm16 | bytes=%s sample_rate=%s",
            len(pcm_bytes),
            sample_rate,
        )
        if not pcm_bytes:
            raise ValueError("Empty audio buffer passed to STT")

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        log.debug("STT audio float samples=%s duration_s=%.2f", len(audio), len(audio) / sample_rate)

        segments, info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            beam_size=1,
        )
        parts: list[str] = []
        for segment in segments:
            log.debug(
                "STT segment | [%.2f→%.2f] %r",
                segment.start,
                segment.end,
                segment.text,
            )
            parts.append(segment.text.strip())

        text = " ".join(p for p in parts if p).strip()
        result = TranscriptResult(
            text=text,
            language=getattr(info, "language", None),
            duration=getattr(info, "duration", None),
        )
        log.info("STT result | text=%r language=%s", result.text, result.language)
        return result


_STT: SpeechToText | None = None


def get_stt() -> SpeechToText:
    global _STT
    if _STT is None:
        _STT = SpeechToText()
    return _STT
