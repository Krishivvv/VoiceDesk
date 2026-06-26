"""Lightweight RAG evaluation: retrieval hit-rate and answer-groundedness.

This is a small, deterministic offline eval over a seed Q&A set. It does not
call an LLM; "groundedness" is measured as whether an expected answer keyword
is present in the retrieved passages, which is the signal an LLM would rely on.
"""

from __future__ import annotations

import pytest

# (query, expected_category, expected_keyword_in_retrieved_text)
SEED_QA = [
    ("What is your return policy?", "returns", "30-day"),
    ("How do I return an item?", "returns", "return"),
    ("How long does standard shipping take?", "shipping", "5-7 business days"),
    ("Do you ship internationally?", "shipping", "internationally"),
    ("How can I contact customer support?", "support", "1-800"),
    ("What payment methods do you accept?", "payment", "PayPal"),
    ("Can I track my order?", "shipping", "tracking number"),
    ("What warranty do you offer on electronics?", "warranty", "1-year"),
    ("Can I cancel my order?", "orders", "1 hour"),
    ("What are your support response times?", "support", "response"),
]

TOP_K = 3
HIT_RATE_THRESHOLD = 0.8
GROUNDEDNESS_THRESHOLD = 0.8


def _evaluate(rag_agent):
    """Return (hit_rate, groundedness) over the seed set."""
    hits = 0
    grounded = 0
    for query, expected_category, expected_keyword in SEED_QA:
        results = rag_agent.collection.query(
            query_texts=[query], n_results=TOP_K, include=["documents", "metadatas"]
        )
        categories = [m["category"] for m in results["metadatas"][0]]
        retrieved_text = " ".join(results["documents"][0]).lower()

        if expected_category in categories:
            hits += 1
        if expected_keyword.lower() in retrieved_text:
            grounded += 1

    n = len(SEED_QA)
    return hits / n, grounded / n


@pytest.mark.integration
def test_rag_retrieval_hit_rate(rag_agent):
    hit_rate, _ = _evaluate(rag_agent)
    assert hit_rate >= HIT_RATE_THRESHOLD, f"retrieval hit-rate {hit_rate:.2f} below {HIT_RATE_THRESHOLD}"


@pytest.mark.integration
def test_rag_answer_groundedness(rag_agent):
    _, groundedness = _evaluate(rag_agent)
    assert groundedness >= GROUNDEDNESS_THRESHOLD, (
        f"groundedness {groundedness:.2f} below {GROUNDEDNESS_THRESHOLD}"
    )


if __name__ == "__main__":
    # Allow running the eval as a script for a quick report.
    import asyncio

    from src.llm.agent import CustomerSupportAgent

    async def _main() -> None:
        agent = CustomerSupportAgent({})
        await agent._setup_knowledge_base()
        hit_rate, groundedness = _evaluate(agent)
        print(f"RAG eval over {len(SEED_QA)} seed queries (top-{TOP_K}):")
        print(f"  retrieval hit-rate : {hit_rate:.1%}")
        print(f"  groundedness       : {groundedness:.1%}")

    asyncio.run(_main())
