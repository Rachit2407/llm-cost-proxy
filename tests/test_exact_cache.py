"""Unit tests for L1 ExactCache."""

from __future__ import annotations

import asyncio

from llm_cost_proxy.core.cache.exact import ExactCache
from llm_cost_proxy.core.redis_store import InMemoryStore
from llm_cost_proxy.models.schemas import CachedEntry, CanonicalRequest, TokenUsage


def _make_request(
    *,
    model: str = "gpt-4o-mini",
    content: str = "Explain Redis pipelining.",
    temperature: float = 0.0,
) -> CanonicalRequest:
    return CanonicalRequest(
        provider="openai",
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        raw_payload={"model": model, "messages": [{"role": "user", "content": content}]},
    )


async def test_exact_cache_hit_and_tenant_isolation(memory_store: InMemoryStore) -> None:
    cache = ExactCache(memory_store)
    req = _make_request()
    entry = CachedEntry(
        provider="openai",
        model="gpt-4o-mini",
        response_body={"id": "resp-1"},
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        estimated_cost_usd=0.0001,
    )

    miss = await cache.lookup("tenant-a", req)
    assert miss.hit is False
    assert miss.status == "MISS"

    await cache.store("tenant-a", req, entry, ttl_seconds=60)

    hit = await cache.lookup("tenant-a", req)
    assert hit.hit is True
    assert hit.status == "EXACT_HIT"
    assert hit.similarity == 1.0
    assert hit.entry is not None
    assert hit.entry.response_body == {"id": "resp-1"}

    # Different tenant or different model must miss
    other_tenant = await cache.lookup("tenant-b", req)
    assert other_tenant.hit is False
    other_model = await cache.lookup("tenant-a", _make_request(model="gpt-4o"))
    assert other_model.hit is False


async def test_exact_cache_ttl_expiration(memory_store: InMemoryStore) -> None:
    cache = ExactCache(memory_store)
    req = _make_request()
    entry = CachedEntry(
        provider="openai",
        model="gpt-4o-mini",
        response_body={"id": "resp-ttl"},
        usage=TokenUsage(prompt_tokens=5, completion_tokens=5),
        estimated_cost_usd=0.00005,
    )

    await cache.store("tenant-a", req, entry, ttl_seconds=1)
    assert (await cache.lookup("tenant-a", req)).hit is True

    await asyncio.sleep(1.05)
    assert (await cache.lookup("tenant-a", req)).hit is False
