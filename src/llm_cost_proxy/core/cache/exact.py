"""L1 Exact Cache using canonical SHA-256 request hashing."""

from __future__ import annotations

import hashlib
import json

from llm_cost_proxy.core.redis_store import StateStore
from llm_cost_proxy.models.schemas import (
    CachedEntry,
    CacheLookupResult,
    CanonicalRequest,
)


class ExactCache:
    """Deterministic SHA-256 exact prompt/parameter cache."""

    def __init__(self, store: StateStore) -> None:
        self._store = store

    @staticmethod
    def build_key(tenant_id: str, request: CanonicalRequest) -> str:
        canonical_obj = {
            "provider": request.provider,
            "model": request.model,
            "system_prompt": request.system_prompt,
            "messages": request.messages,
            "temperature": round(request.temperature, 4),
            "max_tokens": request.max_tokens,
        }
        serialized = json.dumps(canonical_obj, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"cache:exact:{tenant_id}:{digest}"

    async def lookup(self, tenant_id: str, request: CanonicalRequest) -> CacheLookupResult:
        """Look up an exact cached response for the canonicalized request."""
        key = self.build_key(tenant_id, request)
        raw = await self._store.get(key)
        if raw is None:
            return CacheLookupResult(hit=False, status="MISS")
        entry = CachedEntry.model_validate_json(raw)
        return CacheLookupResult(
            hit=True,
            status="EXACT_HIT",
            entry=entry,
            similarity=1.0,
        )

    async def store(
        self,
        tenant_id: str,
        request: CanonicalRequest,
        entry: CachedEntry,
        ttl_seconds: int,
    ) -> None:
        """Persist a completion in the L1 exact cache."""
        key = self.build_key(tenant_id, request)
        await self._store.set(key, entry.model_dump_json(), ttl_seconds=ttl_seconds)
