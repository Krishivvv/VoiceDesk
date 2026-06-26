"""Shared pytest fixtures.

Keeps the heavy, network-free RAG knowledge base built once per session and
provides lightweight fakes so component/API tests never touch the real LLM,
Whisper weights, or the network.
"""

from __future__ import annotations

import pytest_asyncio

from src.llm.agent import CustomerSupportAgent


@pytest_asyncio.fixture(scope="session")
async def rag_agent() -> CustomerSupportAgent:
    """A CustomerSupportAgent with only its ChromaDB knowledge base built.

    No LLM client is created (no API key required); embeddings are produced
    locally by sentence-transformers. Built once per test session.
    """
    agent = CustomerSupportAgent({})
    await agent._setup_knowledge_base()
    return agent
