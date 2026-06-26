"""Audio customer-support pipeline.

Orchestrates the end-to-end flow: STT (Whisper) -> LLM agent (ReAct + RAG)
-> TTS (Edge TTS). The pipeline is fully asynchronous and owns the lifecycle
of all three components, including coordinated cleanup.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.llm.agent import BaseAgent, CustomerSupportAgent
from src.stt.base_stt import BaseSTT, STTService
from src.tts.base_tts import BaseTTS, TTSService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the audio support pipeline."""

    stt_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tts_config: Dict[str, Any]


@dataclass
class TranscriptData:
    """A single user/agent exchange."""

    user_input: str
    agent_response: str


class AudioSupportPipeline:
    """Orchestrates the STT -> LLM -> TTS flow for voice-based support."""

    def __init__(self, config: PipelineConfig) -> None:
        """Create the pipeline with the given component configuration."""
        self.config = config
        self.stt: Optional[BaseSTT] = None
        self.llm_agent: Optional[BaseAgent] = None
        self.tts: Optional[BaseTTS] = None
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialise every component and verify all are ready.

        Raises:
            RuntimeError: If any component fails to become ready.
            Exception: Propagated from a component's own initialisation.
        """
        try:
            logger.info("Initializing Audio Support Pipeline...")

            self.stt = STTService(self.config.stt_config)
            await self.stt.initialize()

            self.llm_agent = CustomerSupportAgent(self.config.llm_config)
            await self.llm_agent.initialize()

            self.tts = TTSService(self.config.tts_config)
            await self.tts.initialize()

            if not self.stt.is_ready():
                raise RuntimeError("STT component failed to initialize")
            if not self.llm_agent.is_initialized:
                raise RuntimeError("LLM Agent failed to initialize")
            if not self.tts.is_ready():
                raise RuntimeError("TTS component failed to initialize")

            self.is_initialized = True
            logger.info("Pipeline initialized successfully — all 3 components ready")

        except Exception as exc:
            logger.error("Pipeline initialization failed: %s", exc)
            await self.cleanup()
            raise

    async def process_audio(self, audio_bytes: bytes, **kwargs: Any) -> bytes:
        """Run the full audio pipeline and return only the response audio."""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        response_audio, _, _ = await self.process_audio_with_transcript(audio_bytes, **kwargs)
        return response_audio

    async def process_audio_with_transcript(
        self, audio_bytes: bytes, **kwargs: Any
    ) -> Tuple[bytes, TranscriptData, int]:
        """Run the full audio pipeline, returning audio, transcript and timing.

        Returns:
            A tuple of ``(response_audio, transcript_data, processing_time_ms)``.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.perf_counter()

        logger.info("Step 1/3: STT — converting speech to text")
        text_input = await self.stt.transcribe(audio_bytes, **kwargs)

        llm_input = text_input
        if not text_input or not text_input.strip():
            llm_input = "I couldn't hear you clearly. Could you please repeat your question?"

        logger.info("Step 2/3: LLM — generating agent response")
        agent_response = await self.llm_agent.process_query(llm_input, **kwargs)

        logger.info("Step 3/3: TTS — synthesizing audio response")
        response_audio = await self.tts.synthesize(agent_response, **kwargs)

        transcript_data = TranscriptData(user_input=text_input or "", agent_response=agent_response)
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("Pipeline complete in %d ms (%d bytes audio)", processing_time_ms, len(response_audio))

        return response_audio, transcript_data, processing_time_ms

    async def process_text(self, text_input: str, **kwargs: Any) -> Tuple[str, bytes]:
        """Process a text query (no STT) and return ``(response_text, audio)``."""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        agent_response = await self.llm_agent.process_query(text_input, **kwargs)
        response_audio = await self.tts.synthesize(agent_response, **kwargs)
        return agent_response, response_audio

    async def process_text_with_timing(self, text_input: str, **kwargs: Any) -> Tuple[str, int]:
        """Process a text query and return ``(response_text, processing_time_ms)``."""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.perf_counter()
        agent_response, _ = await self.process_text(text_input, **kwargs)
        return agent_response, int((time.perf_counter() - start_time) * 1000)

    async def health_check(self) -> Dict[str, bool]:
        """Return the readiness state of the pipeline and each component."""
        return {
            "pipeline_initialized": self.is_initialized,
            "stt_ready": self.stt.is_ready() if self.stt else False,
            "llm_ready": self.llm_agent.is_initialized if self.llm_agent else False,
            "tts_ready": self.tts.is_ready() if self.tts else False,
        }

    async def cleanup(self) -> None:
        """Clean up all components and reset the pipeline to uninitialised."""
        logger.info("Cleaning up pipeline resources...")
        try:
            if self.stt:
                await self.stt.cleanup()
            if self.llm_agent:
                await self.llm_agent.cleanup()
            if self.tts:
                await self.tts.cleanup()
        finally:
            self.stt = None
            self.llm_agent = None
            self.tts = None
            self.is_initialized = False
            logger.info("Pipeline cleanup completed")


async def create_pipeline(
    stt_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    tts_config: Dict[str, Any],
) -> AudioSupportPipeline:
    """Create and initialise an :class:`AudioSupportPipeline`."""
    pipeline = AudioSupportPipeline(
        PipelineConfig(stt_config=stt_config, llm_config=llm_config, tts_config=tts_config)
    )
    await pipeline.initialize()
    return pipeline


async def _demo() -> None:
    """Minimal text round-trip demo. Requires ``GROQ_API_KEY`` in the env."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.error("Set GROQ_API_KEY in your environment to run the demo.")
        return

    pipeline = await create_pipeline(
        stt_config={"model": "base"},
        llm_config={
            "api_key": groq_key,
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            "temperature": 0.7,
        },
        tts_config={"voice": "en-US-AriaNeural"},
    )
    try:
        response_text, _ = await pipeline.process_text("Hello, what is your return policy?")
        logger.info("Agent response: %s", response_text)
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(_demo())
