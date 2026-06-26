"""Unit tests for the Speech-to-Text layer (Whisper mocked, no weights loaded)."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

import src.stt.base_stt as stt_mod
from src.stt.base_stt import BaseSTT, STTService


class _FakeWhisperModel:
    """Stand-in for a loaded Whisper model."""

    def __init__(self, text: str = "hello world") -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        return {"text": self.text}


def _wav_bytes(seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a small mono WAV blob for decoding tests."""
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, int(sample_rate * seconds))).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def test_base_stt_is_abstract():
    with pytest.raises(TypeError):
        BaseSTT()


async def test_transcribe_requires_initialization():
    stt = STTService({"model": "base"})
    with pytest.raises(RuntimeError, match="not initialized"):
        await stt.transcribe(b"x" * 200)


async def test_transcribe_rejects_empty_audio(monkeypatch):
    stt = STTService({"model": "base"})
    monkeypatch.setattr(stt_mod, "_load_whisper_model", lambda name: _FakeWhisperModel())
    await stt.initialize()
    with pytest.raises(ValueError, match="empty"):
        await stt.transcribe(b"")


async def test_transcribe_rejects_too_short_audio(monkeypatch):
    stt = STTService({"model": "base"})
    monkeypatch.setattr(stt_mod, "_load_whisper_model", lambda name: _FakeWhisperModel())
    await stt.initialize()
    with pytest.raises(ValueError, match="too short"):
        await stt.transcribe(b"short")


async def test_transcribe_success(monkeypatch):
    fake = _FakeWhisperModel("the return policy is thirty days")
    monkeypatch.setattr(stt_mod, "_load_whisper_model", lambda name: fake)
    stt = STTService({"model": "base"})
    await stt.initialize()

    result = await stt.transcribe(_wav_bytes())
    assert result == "the return policy is thirty days"
    assert fake.calls == 1


async def test_cleanup_resets_state(monkeypatch):
    monkeypatch.setattr(stt_mod, "_load_whisper_model", lambda name: _FakeWhisperModel())
    stt = STTService({"model": "base"})
    await stt.initialize()
    assert stt.is_ready()
    await stt.cleanup()
    assert stt.client is None
    assert not stt.is_ready()


def test_model_loaded_once_per_name(monkeypatch):
    """The process-wide cache must load each model name exactly once."""
    import whisper

    load_calls = {"n": 0}

    def fake_load(name):
        load_calls["n"] += 1
        return _FakeWhisperModel()

    monkeypatch.setattr(whisper, "load_model", fake_load)
    stt_mod._MODEL_CACHE.clear()

    first = stt_mod._load_whisper_model("base")
    second = stt_mod._load_whisper_model("base")

    assert first is second
    assert load_calls["n"] == 1
    stt_mod._MODEL_CACHE.clear()
