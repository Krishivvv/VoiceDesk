"""Text-to-Speech layer for the audio support pipeline.

Defines the :class:`BaseTTS` interface and :class:`TTSService`, a free
Microsoft Edge TTS implementation that streams MP3 audio with no API key.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_VOICE = "en-US-AriaNeural"


class BaseTTS(ABC):
    """Abstract interface every Text-to-Speech implementation must satisfy."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Store configuration and mark the service uninitialised.

        Args:
            config: Implementation settings, e.g. ``{"voice": "en-US-AriaNeural"}``.
        """
        self.config: dict[str, Any] = config or {}
        self.is_initialized: bool = False

    @abstractmethod
    async def initialize(self) -> None:
        """Set up the service before first use."""

    @abstractmethod
    async def synthesize(self, text: str, **kwargs: Any) -> bytes:
        """Convert ``text`` to speech and return the audio bytes (MP3)."""

    @abstractmethod
    async def synthesize_stream(self, text: str, **kwargs: Any) -> io.BytesIO:
        """Convert ``text`` to speech and return a seekable audio buffer."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release resources held by the service."""

    def is_ready(self) -> bool:
        """Return ``True`` once :meth:`initialize` has completed successfully."""
        return self.is_initialized


class TTSService(BaseTTS):
    """Microsoft Edge TTS implementation.

    Uses ``edge_tts.Communicate`` to stream MP3 audio chunks for a given
    voice. Free, no API key, and network-backed (requires connectivity).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.client: Any = None
        self.voice: str = self.config.get("voice", _DEFAULT_VOICE)

    async def initialize(self) -> None:
        """Verify the ``edge-tts`` dependency and select the voice.

        Raises:
            ImportError: If ``edge-tts`` is not installed.
            RuntimeError: If initialisation fails for any other reason.
        """
        try:
            import edge_tts  # noqa: F401  (import validates availability)

            self.client = "edge_tts"
            self.is_initialized = True
            logger.info("Edge TTS initialised (voice=%s)", self.voice)
        except ImportError as exc:
            raise ImportError("edge-tts is not installed. Run: pip install edge-tts") from exc
        except Exception as exc:
            self.is_initialized = False
            raise RuntimeError(f"TTS initialization failed: {exc}") from exc

    async def synthesize(self, text: str, **kwargs: Any) -> bytes:
        """Synthesise ``text`` to MP3 audio bytes.

        Args:
            text: Text to speak.
            **kwargs: Optional ``voice`` override.

        Returns:
            MP3-encoded audio bytes.

        Raises:
            RuntimeError: If the service is not initialised or synthesis fails.
            ValueError: If ``text`` is empty.
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized. Call initialize() first.")
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            import edge_tts

            voice = kwargs.get("voice", self.voice)
            communicate = edge_tts.Communicate(text.strip(), voice)

            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]

            if not audio_bytes:
                raise RuntimeError("Edge TTS returned empty audio — check your internet connection")

            logger.info("TTS synthesised %d bytes (voice=%s)", len(audio_bytes), voice)
            return audio_bytes

        except Exception as exc:
            raise RuntimeError(f"TTS synthesis failed: {exc}") from exc

    async def synthesize_stream(self, text: str, **kwargs: Any) -> io.BytesIO:
        """Synthesise ``text`` and return it as a seekable :class:`io.BytesIO`."""
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized. Call initialize() first.")

        audio_buffer = io.BytesIO(await self.synthesize(text, **kwargs))
        audio_buffer.seek(0)
        return audio_buffer

    async def cleanup(self) -> None:
        """Mark the service uninitialised. Edge TTS holds no persistent state."""
        self.client = None
        self.is_initialized = False
        logger.info("TTS cleanup completed")

    async def get_available_voices(self) -> list[dict[str, Any]]:
        """Return the list of Edge TTS voices available for synthesis."""
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")

        import edge_tts

        return await edge_tts.list_voices()
