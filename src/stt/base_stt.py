"""Speech-to-Text layer for the audio support pipeline.

Defines the :class:`BaseSTT` interface and :class:`STTService`, a local
OpenAI Whisper implementation. The Whisper model is loaded once per model
name and cached process-wide (see :func:`_load_whisper_model`) so repeated
pipeline initialisations never re-download or re-load weights.
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Whisper expects 16 kHz mono float32 audio.
_TARGET_SAMPLE_RATE = 16000

# Process-wide model cache so the (potentially large) Whisper weights load
# exactly once per model name, even across multiple STTService instances.
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _load_whisper_model(model_name: str) -> Any:
    """Load (or return a cached) Whisper model for ``model_name``.

    Thread-safe and idempotent: the first call loads the weights, every
    subsequent call with the same name returns the cached model.
    """
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        import whisper

        # Make a bundled ffmpeg binary discoverable if one is installed.
        try:
            import imageio_ffmpeg

            ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
            os.environ["PATH"] += os.pathsep + ffmpeg_dir
        except ImportError:
            pass

        logger.info("Loading Whisper model '%s' (first run may download weights)...", model_name)
        model = whisper.load_model(model_name)
        _MODEL_CACHE[model_name] = model
        logger.info("Whisper model '%s' loaded and cached", model_name)
        return model


class BaseSTT(ABC):
    """Abstract interface every Speech-to-Text implementation must satisfy."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Store configuration and mark the service uninitialised.

        Args:
            config: Implementation settings, e.g. ``{"model": "base"}``.
        """
        self.config: Dict[str, Any] = config or {}
        self.is_initialized: bool = False

    @abstractmethod
    async def initialize(self) -> None:
        """Set up the service (load models / open clients) before first use."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, **kwargs: Any) -> str:
        """Transcribe raw audio bytes to text."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release resources held by the service."""

    def is_ready(self) -> bool:
        """Return ``True`` once :meth:`initialize` has completed successfully."""
        return self.is_initialized


class STTService(BaseSTT):
    """Local OpenAI Whisper transcription service.

    Decodes incoming audio in-memory with ``soundfile`` and resamples to
    16 kHz mono before inference; if decoding fails it falls back to writing
    a temporary file and letting Whisper/ffmpeg handle the format. Inference
    runs in a worker thread so it never blocks the event loop.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.client: Any = None
        self.model_name: str = self.config.get("model", "base")

    async def initialize(self) -> None:
        """Load the Whisper model (cached) and mark the service ready.

        Raises:
            ImportError: If ``openai-whisper`` is not installed.
            RuntimeError: If model loading fails for any other reason.
        """
        try:
            self.client = await asyncio.to_thread(_load_whisper_model, self.model_name)
            self.is_initialized = True
            logger.info("Whisper STT initialised (model=%s)", self.model_name)
        except ImportError as exc:
            raise ImportError("openai-whisper is not installed. Run: pip install openai-whisper") from exc
        except Exception as exc:
            self.is_initialized = False
            raise RuntimeError(f"STT initialization failed: {exc}") from exc

    async def transcribe(self, audio_bytes: bytes, **kwargs: Any) -> str:
        """Transcribe ``audio_bytes`` to text.

        Args:
            audio_bytes: Raw audio in any ffmpeg/soundfile-decodable format.

        Returns:
            The transcribed text, or an empty string if no speech is detected.

        Raises:
            RuntimeError: If the service is not initialised or inference fails.
            ValueError: If the audio is empty or implausibly short.
        """
        if not self.is_ready():
            raise RuntimeError("STT service not initialized. Call initialize() first.")
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")
        if len(audio_bytes) < 100:
            raise ValueError("Audio data is too short to contain speech")

        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, **kwargs)

    def _transcribe_sync(self, audio_bytes: bytes, **kwargs: Any) -> str:
        """Blocking transcription helper executed in a worker thread."""
        # Strategy 1: decode entirely in memory with soundfile.
        try:
            import soundfile as sf

            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio_data.size == 0 or sample_rate <= 0:
                raise ValueError("Invalid audio data")

            if audio_data.ndim > 1:  # stereo -> mono
                audio_data = audio_data.mean(axis=1)

            if sample_rate != _TARGET_SAMPLE_RATE:
                new_length = int(audio_data.shape[0] * _TARGET_SAMPLE_RATE / sample_rate)
                if new_length <= 0:
                    raise ValueError("Invalid audio length after resampling")
                audio_data = np.interp(
                    np.linspace(0, audio_data.shape[0], num=new_length, endpoint=False),
                    np.arange(audio_data.shape[0]),
                    audio_data,
                ).astype("float32")

            transcription = self.client.transcribe(audio_data, fp16=False).get("text", "").strip()
            self._log_transcription(transcription)
            return transcription

        except Exception as decode_error:
            initial_decode_error = str(decode_error)
            logger.warning("In-memory decode failed (%s); falling back to temp file", initial_decode_error)

        # Strategy 2: write to a temp file and let Whisper/ffmpeg decode it.
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=os.getcwd()) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                tmp_path = tmp_file.name

            transcription = self.client.transcribe(tmp_path, fp16=False).get("text", "").strip()
            self._log_transcription(transcription)
            return transcription

        except FileNotFoundError as exc:
            raise RuntimeError(
                "Transcription failed: ffmpeg not found. Install ffmpeg and ensure it is on PATH, "
                "or provide 16 kHz WAV audio."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Transcription failed: {exc} (initial decode error: {initial_decode_error})"
            ) from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _log_transcription(transcription: str) -> None:
        if transcription:
            logger.info("Transcription: '%s'", transcription)
        else:
            logger.info("Whisper returned empty transcription (no speech detected)")

    async def cleanup(self) -> None:
        """Detach from the cached model and mark the service uninitialised.

        The shared model stays in the process-wide cache for reuse; only this
        instance's reference is cleared.
        """
        self.client = None
        self.is_initialized = False
        logger.info("STT cleanup completed")
