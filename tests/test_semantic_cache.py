"""Unit tests for L2 SemanticCache and LocalHashEmbedder."""

from __future__ import annotations

from llm_cost_proxy.core.cache.embeddings import LocalHashEmbedder
from llm_cost_proxy.core.cache.semantic import SemanticCache
from llm_cost_proxy.core.redis_store import InMemoryStore
from llm_cost_proxy.models.schemas import CachedEntry, CanonicalRequest, TokenUsage


def _make_request(prompt: str, model: str = "gpt-4o-mini") -> CanonicalRequest:
    return CanonicalRequest(
        provider="openai",
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        raw_payload={"model": model, "messages": [{"role": "user", "content": prompt}]},
    )


async def test_semantic_cache_hits_near_duplicate_prompt(memory_store: InMemoryStore) -> None:
    embedder = LocalHashEmbedder(dimension=256)
    cache = SemanticCache(memory_store, embedder)

    original = _make_request("How do I configure Redis persistence with AOF and RDB snapshots?")
    paraphrase = _make_request("How to configure Redis persistence using AOF and RDB snapshots?")
    unrelated = _make_request("Write a Python function to invert a binary tree recursively.")

    entry = CachedEntry(
        provider="openai",
        model="gpt-4o-mini",
        response_body={"choices": [{"message": {"content": "Enable appendonly yes."}}]},
        usage=TokenUsage(prompt_tokens=20, completion_tokens=30),
        estimated_cost_usd=0.0002,
    )

    await cache.store("tenant-a", original, entry, ttl_seconds=120)

    hit = await cache.lookup("tenant-a", paraphrase, threshold=0.75)
    assert hit.hit is True
    assert hit.status == "SEMANTIC_HIT"
    assert hit.similarity is not None
    assert hit.similarity >= 0.75
    assert hit.entry is not None
    assert hit.entry.response_body == entry.response_body

    miss = await cache.lookup("tenant-a", unrelated, threshold=0.75)
    assert miss.hit is False
    assert miss.status == "MISS"


async def test_semantic_cache_isolates_by_model_and_tenant(
    memory_store: InMemoryStore,
) -> None:
    embedder = LocalHashEmbedder(dimension=256)
    cache = SemanticCache(memory_store, embedder)

    req_mini = _make_request("Explain FastAPI dependency injection.", model="gpt-4o-mini")
    req_4o = _make_request("Explain FastAPI dependency injection.", model="gpt-4o")

    entry = CachedEntry(
        provider="openai",
        model="gpt-4o-mini",
        response_body={"id": "mini-resp"},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
        estimated_cost_usd=0.0001,
    )
    await cache.store("tenant-a", req_mini, entry, ttl_seconds=120)

    assert (await cache.lookup("tenant-a", req_4o, threshold=0.5)).hit is False
    assert (await cache.lookup("tenant-b", req_mini, threshold=0.5)).hit is False
