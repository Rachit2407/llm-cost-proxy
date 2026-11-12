"""L2 Semantic Cache using prompt embeddings and StateStore vector search (RediSearch HNSW)."""

from __future__ import annotations

import hashlib

from llm_cost_proxy.core.cache.embeddings import Embedder
from llm_cost_proxy.core.redis_store import StateStore
from llm_cost_proxy.models.schemas import (
    CachedEntry,
    CacheLookupResult,
    CanonicalRequest,
)


class SemanticCache:
    """Tenant- and model-scoped L2 semantic vector cache."""

    def __init__(
        self,
        store: StateStore,
        embedder: Embedder,
        max_candidates_per_scope: int = 500,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._max_candidates = max_candidates_per_scope

    @staticmethod
    def _scope(tenant_id: str, request: CanonicalRequest) -> str:
        return f"{tenant_id}:{request.provider}:{request.model}"

    @staticmethod
    def _entry_id(prompt_text: str) -> str:
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:24]

    async def lookup(
        self,
        tenant_id: str,
        request: CanonicalRequest,
        threshold: float,
    ) -> CacheLookupResult:
        """Embed the prompt and return the highest-similarity cached entry >= threshold."""
        prompt_text = request.prompt_text()
        query_vec = await self._embedder.embed(prompt_text)
        matches = await self._store.vector_search(
            scope=self._scope(tenant_id, request),
            query_vector=query_vec,
            top_k=1,
        )
        if not matches:
            return CacheLookupResult(hit=False, status="MISS")

        best_score, payload_json = matches[0]
        if best_score >= threshold:
            entry = CachedEntry.model_validate_json(payload_json)
            return CacheLookupResult(
                hit=True,
                status="SEMANTIC_HIT",
                entry=entry,
                similarity=round(best_score, 4),
            )
        return CacheLookupResult(hit=False, status="MISS")

    async def store(
        self,
        tenant_id: str,
        request: CanonicalRequest,
        entry: CachedEntry,
        ttl_seconds: int,
    ) -> None:
        """Compute embedding and store the response under the tenant/model scope."""
        prompt_text = request.prompt_text()
        vec = await self._embedder.embed(prompt_text)
        await self._store.vector_upsert(
            scope=self._scope(tenant_id, request),
            doc_id=self._entry_id(prompt_text),
            vector=vec,
            payload_json=entry.model_dump_json(),
            ttl_seconds=ttl_seconds,
        )
