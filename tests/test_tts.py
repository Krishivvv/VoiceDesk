"""Unit tests for the Text-to-Speech layer (Edge TTS network call mocked)."""

from __future__ import annotations

import edge_tts
import pytest

from src.tts.base_tts import BaseTTS, TTSService


class _FakeCommunicate:
    """Stand-in for edge_tts.Communicate that yields canned audio chunks."""

    def __init__(self, text: str, voice: str) -> None:
        self.text = text
        self.voice = voice

    async def stream(self):
        yield {"type": "audio", "data": b"ID3fake-mp3-"}
        yield {"type": "WordBoundary"}
        yield {"type": "audio", "data": b"audio-bytes"}


def test_base_tts_is_abstract():
    with pytest.raises(TypeError):
        BaseTTS()


async def test_initialize_sets_ready():
    tts = TTSService({"voice": "en-US-AriaNeural"})
    await tts.initialize()
    assert tts.is_ready()
    assert tts.voice == "en-US-AriaNeural"


async def test_synthesize_requires_initialization():
    tts = TTSService()
    with pytest.raises(RuntimeError, match="not initialized"):
        await tts.synthesize("hello")


async def test_synthesize_rejects_empty_text():
    tts = TTSService()
    await tts.initialize()
    with pytest.raises(ValueError, match="empty"):
        await tts.synthesize("   ")


async def test_synthesize_concatenates_audio_chunks(monkeypatch):
    monkeypatch.setattr(edge_tts, "Communicate", _FakeCommunicate)
    tts = TTSService()
    await tts.initialize()

    audio = await tts.synthesize("Your return window is 30 days.")
    assert audio == b"ID3fake-mp3-audio-bytes"


async def test_synthesize_stream_returns_seekable_buffer(monkeypatch):
    monkeypatch.setattr(edge_tts, "Communicate", _FakeCommunicate)
    tts = TTSService()
    await tts.initialize()

    buffer = await tts.synthesize_stream("hello")
    assert buffer.tell() == 0
    assert buffer.read() == b"ID3fake-mp3-audio-bytes"


async def test_cleanup_resets_state():
    tts = TTSService()
    await tts.initialize()
    await tts.cleanup()
    assert not tts.is_ready()
