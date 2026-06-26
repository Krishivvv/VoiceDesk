"""FastAPI server exposing the audio customer-support pipeline.

Provides REST endpoints for text and audio interactions plus health and
component-level debug routes. The pipeline is created once during the
application lifespan and shared across requests.
"""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.pipeline import AudioSupportPipeline, create_pipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---- Request validation limits ----
MAX_TEXT_LENGTH = 2000
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/x-flac",
    "application/octet-stream",  # browsers often send this for recorded blobs
}

# Placeholder API-key values that should be treated as "unset".
_PLACEHOLDER_KEYS = {
    "your_llm_api_key_here",
    "YOUR_REAL_OPENAI_API_KEY_HERE",
    "REPLACE_WITH_YOUR_OPENAI_API_KEY",
    "REPLACE_WITH_YOUR_GROQ_API_KEY",
    "",
}

# Shared pipeline instance, created during the lifespan handler.
pipeline: AudioSupportPipeline | None = None


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
class TextRequest(BaseModel):
    """Request body for text-based queries."""

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    parameters: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Aggregate health/readiness of the pipeline and its components."""

    status: str
    components: dict[str, bool]
    message: str


class TextResponse(BaseModel):
    """Response body for text queries."""

    response_text: str
    audio_available: bool
    processing_time_ms: int


class TranscriptData(BaseModel):
    """A single user/agent exchange."""

    user_input: str
    agent_response: str


class EnhancedAudioResponse(BaseModel):
    """Response body for audio queries, including transcript and timing."""

    success: bool
    audio_response: str
    transcript: TranscriptData
    processing_time_ms: int


# --------------------------------------------------------------------------- #
# Pipeline configuration helpers
# --------------------------------------------------------------------------- #
def _build_llm_config() -> dict[str, Any] | None:
    """Resolve LLM configuration from the environment.

    Prefers Groq (``GROQ_API_KEY``) and falls back to OpenAI
    (``OPENAI_API_KEY`` / ``LLM_API_KEY``). Returns ``None`` when no usable
    key is configured.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

    if groq_key and groq_key not in _PLACEHOLDER_KEYS:
        config = {
            "api_key": groq_key,
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            "temperature": 0.7,
        }
        logger.info("Using Groq LLM provider (model=%s)", config["model"])
        return config

    if openai_key and openai_key not in _PLACEHOLDER_KEYS:
        logger.info("Using OpenAI LLM provider (model=gpt-3.5-turbo)")
        return {"api_key": openai_key, "model": "gpt-3.5-turbo", "temperature": 0.7}

    logger.error("No LLM API key set. Provide GROQ_API_KEY or OPENAI_API_KEY and restart.")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the pipeline on startup and tear it down on shutdown."""
    global pipeline

    from dotenv import load_dotenv

    load_dotenv()
    logger.info("Starting Audio Support Agent API server...")

    llm_config = _build_llm_config()
    if llm_config is not None:
        try:
            pipeline = await create_pipeline(
                stt_config={"model": os.getenv("WHISPER_MODEL", "base")},
                llm_config=llm_config,
                tts_config={"voice": os.getenv("TTS_VOICE", "en-US-AriaNeural")},
            )
            logger.info("Pipeline initialized and ready to serve requests")
        except Exception as exc:
            logger.error("Failed to initialize pipeline: %s", exc)
            pipeline = None

    yield

    if pipeline is not None:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None


app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for the STT -> LLM -> TTS voice support pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — configurable via ALLOWED_ORIGINS (comma-separated); defaults to "*".
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_pipeline() -> AudioSupportPipeline:
    """Return the initialised pipeline or raise HTTP 503."""
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Check /health for details.")
    return pipeline


async def _read_validated_audio(audio: UploadFile) -> bytes:
    """Read an uploaded audio file, enforcing type and size limits."""
    if audio.content_type and audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported audio type: {audio.content_type}")

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file received")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(audio_bytes)} bytes; max {MAX_AUDIO_BYTES}).",
        )
    return audio_bytes


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_model=dict[str, str])
async def root() -> dict[str, str]:
    """Return basic API metadata."""
    return {"message": "Audio Customer Support Agent API", "version": "1.0.0", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Report the readiness of the pipeline and its components."""
    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            components={
                "pipeline_initialized": False,
                "stt_ready": False,
                "llm_ready": False,
                "tts_ready": False,
            },
            message="Pipeline not initialized. Check GROQ_API_KEY/OPENAI_API_KEY and restart server.",
        )

    try:
        components = await pipeline.health_check()
        all_healthy = all(components.values())
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            components=components,
            message="All components operational" if all_healthy else "Some components not ready — check logs",
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return HealthResponse(status="error", components={}, message=f"Health check error: {exc}")


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest) -> TextResponse:
    """Answer a text query through the LLM agent (no STT)."""
    active = _require_pipeline()
    try:
        import time

        start_time = time.perf_counter()
        response_text, response_audio = await active.process_text(request.text, **request.parameters)
        processing_time = int((time.perf_counter() - start_time) * 1000)
        return TextResponse(
            response_text=response_text,
            audio_available=len(response_audio) > 0,
            processing_time_ms=processing_time,
        )
    except Exception as exc:
        logger.error("Text processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/audio", response_model=EnhancedAudioResponse)
async def chat_audio(audio: UploadFile = File(...)) -> EnhancedAudioResponse:
    """Run the full STT -> LLM -> TTS pipeline on an uploaded audio file."""
    active = _require_pipeline()
    try:
        audio_bytes = await _read_validated_audio(audio)
        logger.info("Processing audio upload: %d bytes (file=%s)", len(audio_bytes), audio.filename)

        response_audio, transcript_data, processing_time_ms = await active.process_audio_with_transcript(
            audio_bytes
        )
        return EnhancedAudioResponse(
            success=True,
            audio_response=base64.b64encode(response_audio).decode("ascii"),
            transcript=TranscriptData(
                user_input=transcript_data.user_input,
                agent_response=transcript_data.agent_response,
            ),
            processing_time_ms=processing_time_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audio processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/chat/audio/{text}")
async def text_to_audio(text: str = Path(..., min_length=1, max_length=MAX_TEXT_LENGTH)) -> Response:
    """Synthesise ``text`` to speech via the TTS component."""
    active = _require_pipeline()
    try:
        if not active.tts:
            raise HTTPException(status_code=503, detail="TTS component not available")
        audio_bytes = await active.tts.synthesize(text)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=tts_output.mp3"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("TTS endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)) -> dict[str, Any]:
    """Transcribe an uploaded audio file with the STT component only."""
    active = _require_pipeline()
    try:
        audio_bytes = await _read_validated_audio(audio)
        if not active.stt:
            raise HTTPException(status_code=503, detail="STT component not available")
        transcription = await active.stt.transcribe(audio_bytes)
        return {
            "transcription": transcription,
            "length_chars": len(transcription),
            "empty": len(transcription.strip()) == 0,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("STT debug failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
