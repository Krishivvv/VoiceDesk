"""API tests using FastAPI's TestClient with a fake in-memory pipeline.

The TestClient is created without the ``with`` block so the real lifespan
(which would try to load Whisper/LLM) never runs; we inject a fake pipeline
onto the module global instead.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api.server as server
from src.pipeline import TranscriptData


class _FakeTTS:
    async def synthesize(self, text, **kwargs):
        return b"ID3fake-audio"


class _FakeSTT:
    async def transcribe(self, audio_bytes, **kwargs):
        return "transcribed text"


class _FakePipeline:
    is_initialized = True
    tts = _FakeTTS()
    stt = _FakeSTT()

    async def process_text(self, text, **kwargs):
        return f"Echo: {text}", b"ID3fake-audio"

    async def process_audio_with_transcript(self, audio_bytes, **kwargs):
        return b"ID3fake-audio", TranscriptData("hi", "hello there"), 123

    async def health_check(self):
        return {
            "pipeline_initialized": True,
            "stt_ready": True,
            "llm_ready": True,
            "tts_ready": True,
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "pipeline", _FakePipeline())
    return TestClient(server.app)


@pytest.fixture
def client_no_pipeline(monkeypatch):
    monkeypatch.setattr(server, "pipeline", None)
    return TestClient(server.app)


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["version"] == "1.0.0"


def test_health_healthy(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert all(body["components"].values())


def test_health_unhealthy_when_no_pipeline(client_no_pipeline):
    resp = client_no_pipeline.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unhealthy"


def test_chat_text_success(client):
    resp = client.post("/chat/text", json={"text": "What is your return policy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response_text"] == "Echo: What is your return policy?"
    assert body["audio_available"] is True


def test_chat_text_rejects_empty(client):
    resp = client.post("/chat/text", json={"text": ""})
    assert resp.status_code == 422  # pydantic min_length violation


def test_chat_text_rejects_overlong(client):
    resp = client.post("/chat/text", json={"text": "x" * 5000})
    assert resp.status_code == 422


def test_chat_text_503_without_pipeline(client_no_pipeline):
    resp = client_no_pipeline.post("/chat/text", json={"text": "hi"})
    assert resp.status_code == 503


def test_chat_audio_success(client):
    resp = client.post(
        "/chat/audio",
        files={"audio": ("clip.wav", b"RIFFfake-wav-data", "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["transcript"]["agent_response"] == "hello there"
    assert body["processing_time_ms"] == 123


def test_chat_audio_rejects_empty_file(client):
    resp = client.post("/chat/audio", files={"audio": ("clip.wav", b"", "audio/wav")})
    assert resp.status_code == 400


def test_chat_audio_rejects_unsupported_type(client):
    resp = client.post(
        "/chat/audio",
        files={"audio": ("clip.txt", b"not audio", "text/plain")},
    )
    assert resp.status_code == 415


def test_debug_stt(client):
    resp = client.post(
        "/debug/stt",
        files={"audio": ("clip.wav", b"RIFFfake-wav-data", "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json()["transcription"] == "transcribed text"


def test_text_to_audio(client):
    resp = client.get("/chat/audio/hello%20world")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"ID3fake-audio"
