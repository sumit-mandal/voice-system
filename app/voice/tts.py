"""PocketTTS wrapper — synthesizes agent replies to PCM/WAV bytes."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from pocket_tts import TTSModel

from app.config import get_settings
from app.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class SynthResult:
    pcm_int16: bytes
    sample_rate: int
    wav_bytes: bytes


class TextToSpeech:
    def __init__(self) -> None:
        settings = get_settings()
        log.debug("Loading PocketTTS | voice=%s", settings.tts_voice)
        self._model = TTSModel.load_model()
        self._voice = settings.tts_voice
        self._voice_state = self._model.get_state_for_audio_prompt(self._voice)
        self.sample_rate = int(getattr(self._model, "sample_rate", settings.tts_sample_rate))
        log.info(
            "PocketTTS ready | voice=%s sample_rate=%s",
            self._voice,
            self.sample_rate,
        )

    def synthesize(self, text: str) -> SynthResult:
        log.debug("TTS synthesize | text=%r chars=%s", text, len(text))
        if not text.strip():
            raise ValueError("Empty text passed to TTS")

        audio = self._model.generate_audio(self._voice_state, text)
        if hasattr(audio, "numpy"):
            samples = audio.numpy()
        else:
            samples = np.asarray(audio)

        samples = samples.astype(np.float32).reshape(-1)
        # Clip + convert to PCM16 for LiveKit playback.
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        pcm_bytes = pcm.tobytes()

        buf = io.BytesIO()
        sf.write(buf, clipped, self.sample_rate, format="WAV")
        wav_bytes = buf.getvalue()

        log.info(
            "TTS done | samples=%s duration_s=%.2f pcm_bytes=%s wav_bytes=%s",
            len(pcm),
            len(pcm) / self.sample_rate,
            len(pcm_bytes),
            len(wav_bytes),
        )
        return SynthResult(
            pcm_int16=pcm_bytes,
            sample_rate=self.sample_rate,
            wav_bytes=wav_bytes,
        )


_TTS: TextToSpeech | None = None


def get_tts() -> TextToSpeech:
    global _TTS
    if _TTS is None:
        _TTS = TextToSpeech()
    return _TTS
