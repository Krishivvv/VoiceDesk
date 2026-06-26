"""
Knowledge Base Debug Utility

Run this script to inspect the ChromaDB collection and test RAG retrieval.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.llm.agent import CustomerSupportAgent


async def setup_agent() -> CustomerSupportAgent | None:
    """Initialize agent and set up the knowledge base."""
    print("Setting up knowledge base...")
    print("=" * 50)

    try:
        agent = CustomerSupportAgent({"model": "test", "temperature": 0.7})
        await agent._setup_knowledge_base()
        print(f"Collection: {agent.collection.name}")
        print(f"Documents: {agent.collection.count()}")
        return agent
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


async def run_sample_queries(agent: CustomerSupportAgent):
    """Run sample queries and print RAG results."""
    queries = [
        "What is your return policy?",
        "How long does shipping take?",
        "How can I contact customer support?",
        "What payment methods do you accept?",
        "Can I track my order?",
        "Do you ship internationally?",
        "What is covered under warranty?",
        "How do I return an item?",
        "What are your business hours?",
        "Can I cancel my order?",
    ]

    print("\n" + "=" * 50)
    print("RAG Query Results")
    print("=" * 50)

    for i, query in enumerate(queries, 1):
        print(f"\n{i:2d}. Query: {query}")
        result = agent._rag_search(query)
        print(f"    Result: {result[:120]}{'...' if len(result) > 120 else ''}")


def show_collection_structure(agent: CustomerSupportAgent):
    """Print a sample of documents and all categories in the collection."""
    print("\n" + "=" * 50)
    print("Collection Structure")
    print("=" * 50)

    try:
        sample = agent.collection.query(
            query_texts=["return policy"],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )

        if sample["documents"] and sample["documents"][0]:
            for i, (doc, meta, dist) in enumerate(
                zip(sample["documents"][0], sample["metadatas"][0], sample["distances"][0], strict=False)
            ):
                print(f"\nDocument {i + 1}:")
                print(f"  Title:    {meta['title']}")
                print(f"  Category: {meta['category']}")
                print(f"  Distance: {dist:.4f}")
                print(f"  Content:  {doc[:150]}{'...' if len(doc) > 150 else ''}")

        all_results = agent.collection.query(
            query_texts=[""],
            n_results=agent.collection.count(),
            include=["metadatas"],
        )
        if all_results["metadatas"] and all_results["metadatas"][0]:
            categories = sorted({m["category"] for m in all_results["metadatas"][0]})
            print(f"\nCategories: {', '.join(categories)}")
            print(f"Total docs: {len(all_results['metadatas'][0])}")

    except Exception as e:
        print(f"Error: {str(e)}")


async def main():
    print("AI Audio Support Agent — Knowledge Base Debug")
    print("=" * 60)

    agent = await setup_agent()
    if agent is None:
        return

    show_collection_structure(agent)
    await run_sample_queries(agent)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
