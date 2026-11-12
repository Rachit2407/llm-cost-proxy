"""Redis and in-memory state backends for caching, rate limiting, and budgets."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Awaitable
from typing import Protocol, cast
from uuid import uuid4

import numpy as np
import redis.asyncio as redis_async
from redis.exceptions import ResponseError


async def _resolve[T](value: Awaitable[T] | T) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


class StateStore(Protocol):
    """Unified async key-value, hash, sliding-window, and counter store."""

    async def ping(self) -> bool:
        """Check backend health."""
        ...

    async def get(self, key: str) -> str | None:
        """Return string value for key, or None if missing/expired."""
        ...

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Set string value with optional TTL in seconds."""
        ...

    async def hset(
        self,
        key: str,
        field: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Set a hash field with optional TTL on the parent hash key."""
        ...

    async def hgetall(self, key: str) -> dict[str, str]:
        """Return all fields of a hash key."""
        ...

    async def get_float(self, key: str) -> float:
        """Return float counter value or 0.0 if unset."""
        ...

    async def incr_float(
        self,
        key: str,
        amount: float,
        ttl_seconds: int | None = None,
    ) -> float:
        """Atomically increment a float counter and return the new value."""
        ...

    async def window_sum(
        self,
        key: str,
        window_seconds: int,
        now_ts: float | None = None,
    ) -> int:
        """Return sum of weights in the sliding window [now - window, now]."""
        ...

    async def window_add_if_under(
        self,
        key: str,
        window_seconds: int,
        limit: int,
        weight: int = 1,
        now_ts: float | None = None,
    ) -> tuple[bool, int]:
        """Atomically check current window sum + weight <= limit and add if allowed."""
        ...

    async def window_add(
        self,
        key: str,
        window_seconds: int,
        weight: int,
        now_ts: float | None = None,
    ) -> int:
        """Unconditionally add a weighted event to the sliding window and return new sum."""
        ...

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all keys matching prefix and return the number of deleted keys."""
        ...

    async def vector_upsert(
        self,
        *,
        scope: str,
        doc_id: str,
        vector: list[float],
        payload_json: str,
        ttl_seconds: int,
    ) -> None:
        """Store an L2-normalized vector and JSON payload under a scoped namespace."""
        ...

    async def vector_search(
        self,
        *,
        scope: str,
        query_vector: list[float],
        top_k: int = 1,
    ) -> list[tuple[float, str]]:
        """Return up to top_k (cosine_similarity, payload_json) pairs in scope."""
        ...

    async def close(self) -> None:
        """Release underlying connections."""
        ...


class InMemoryStore:
    """Thread/async-safe in-memory StateStore for tests and local --no-redis runs."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._kv: dict[str, tuple[str, float | None]] = {}
        self._hashes: dict[str, tuple[dict[str, str], float | None]] = {}
        self._windows: dict[str, list[tuple[float, int]]] = {}

    def _is_expired(self, expires_at: float | None, now: float) -> bool:
        return expires_at is not None and now >= expires_at

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> str | None:
        async with self._lock:
            now = time.time()
            entry = self._kv.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if self._is_expired(expires_at, now):
                del self._kv[key]
                return None
            return value

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        async with self._lock:
            expires_at = (time.time() + ttl_seconds) if ttl_seconds is not None else None
            self._kv[key] = (value, expires_at)

    async def hset(
        self,
        key: str,
        field: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> None:
        async with self._lock:
            now = time.time()
            existing = self._hashes.get(key)
            if existing is None or self._is_expired(existing[1], now):
                mapping: dict[str, str] = {}
            else:
                mapping = existing[0]
            mapping[field] = value
            expires_at = (now + ttl_seconds) if ttl_seconds is not None else None
            self._hashes[key] = (mapping, expires_at)

    async def hgetall(self, key: str) -> dict[str, str]:
        async with self._lock:
            now = time.time()
            existing = self._hashes.get(key)
            if existing is None:
                return {}
            mapping, expires_at = existing
            if self._is_expired(expires_at, now):
                del self._hashes[key]
                return {}
            return dict(mapping)

    async def get_float(self, key: str) -> float:
        raw = await self.get(key)
        return float(raw) if raw is not None else 0.0

    async def incr_float(
        self,
        key: str,
        amount: float,
        ttl_seconds: int | None = None,
    ) -> float:
        async with self._lock:
            now = time.time()
            entry = self._kv.get(key)
            current = 0.0
            existing_exp: float | None = None
            if entry is not None and not self._is_expired(entry[1], now):
                current = float(entry[0])
                existing_exp = entry[1]
            updated = round(current + amount, 10)
            expires_at = (
                (now + ttl_seconds)
                if ttl_seconds is not None and existing_exp is None
                else existing_exp
            )
            self._kv[key] = (str(updated), expires_at)
            return updated

    def _prune_window(
        self,
        key: str,
        window_seconds: int,
        now: float,
    ) -> list[tuple[float, int]]:
        cutoff = now - window_seconds
        events = [item for item in self._windows.get(key, []) if item[0] > cutoff]
        self._windows[key] = events
        return events

    async def window_sum(
        self,
        key: str,
        window_seconds: int,
        now_ts: float | None = None,
    ) -> int:
        async with self._lock:
            now = now_ts if now_ts is not None else time.time()
            events = self._prune_window(key, window_seconds, now)
            return sum(weight for _, weight in events)

    async def window_add_if_under(
        self,
        key: str,
        window_seconds: int,
        limit: int,
        weight: int = 1,
        now_ts: float | None = None,
    ) -> tuple[bool, int]:
        async with self._lock:
            now = now_ts if now_ts is not None else time.time()
            events = self._prune_window(key, window_seconds, now)
            current_sum = sum(w for _, w in events)
            if current_sum + weight > limit:
                return False, current_sum
            events.append((now, weight))
            return True, current_sum + weight

    async def window_add(
        self,
        key: str,
        window_seconds: int,
        weight: int,
        now_ts: float | None = None,
    ) -> int:
        async with self._lock:
            now = now_ts if now_ts is not None else time.time()
            events = self._prune_window(key, window_seconds, now)
            events.append((now, weight))
            return sum(w for _, w in events)

    async def delete_prefix(self, prefix: str) -> int:
        async with self._lock:
            kv_keys = [k for k in self._kv if k.startswith(prefix)]
            hash_keys = [k for k in self._hashes if k.startswith(prefix)]
            for k in kv_keys:
                del self._kv[k]
            for k in hash_keys:
                del self._hashes[k]
            return len(kv_keys) + len(hash_keys)

    async def vector_upsert(
        self,
        *,
        scope: str,
        doc_id: str,
        vector: list[float],
        payload_json: str,
        ttl_seconds: int,
    ) -> None:
        record = json.dumps(
            {
                "vector": vector,
                "payload": payload_json,
                "expires_at": time.time() + ttl_seconds,
            }
        )
        await self.hset(f"cache:vec:{scope}", doc_id, record, ttl_seconds=ttl_seconds)

    async def vector_search(
        self,
        *,
        scope: str,
        query_vector: list[float],
        top_k: int = 1,
    ) -> list[tuple[float, str]]:
        raw_records = await self.hgetall(f"cache:vec:{scope}")
        if not raw_records:
            return []
        now = time.time()
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        scored: list[tuple[float, str]] = []
        for raw in raw_records.values():
            obj = json.loads(raw)
            if now >= float(obj["expires_at"]):
                continue
            v = np.asarray(obj["vector"], dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0.0 or v.shape != q.shape:
                continue
            score = float(np.dot(q, v) / (q_norm * v_norm))
            scored.append((score, str(obj["payload"])))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    async def close(self) -> None:
        return None


# Atomic Lua script for Redis sorted-set sliding windows storing "<uuid>:<weight>".
_LUA_WINDOW_ADD_IF_UNDER = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local cutoff = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local weight = tonumber(ARGV[4])
local member = ARGV[5]
local ttl = tonumber(ARGV[6])

redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local members = redis.call('ZRANGE', key, 0, -1)
local total = 0
for _, m in ipairs(members) do
    local sep = string.find(m, ':')
    if sep then
        total = total + (tonumber(string.sub(m, sep + 1)) or 0)
    end
end

if total + weight > limit then
    return {0, total}
end

redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, ttl)
return {1, total + weight}
"""


_SCOPE_TAG_RE = re.compile(r"[^a-zA-Z0-9_]")
_VECTOR_INDEX_NAME = "idx:semantic_v1"


async def _exec(client: redis_async.Redis, *args: object) -> object:
    return await client.execute_command(*args)  # type: ignore[no-untyped-call]


class RedisStore:
    """Async Redis StateStore backed by redis.asyncio with RediSearch HNSW support."""

    def __init__(self, redis_url: str) -> None:
        self._client: redis_async.Redis = redis_async.Redis.from_url(
            redis_url,
            decode_responses=True,
        )
        self._bytes_client: redis_async.Redis = redis_async.Redis.from_url(
            redis_url,
            decode_responses=False,
        )
        self._vector_index_ready: bool | None = None

    async def ping(self) -> bool:
        result = await _resolve(self._client.ping())
        return bool(result)

    async def get(self, key: str) -> str | None:
        val = await self._client.get(key)
        return str(val) if val is not None else None

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        if ttl_seconds is not None:
            await self._client.set(key, value, ex=ttl_seconds)
        else:
            await self._client.set(key, value)

    async def hset(
        self,
        key: str,
        field: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> None:
        await _resolve(self._client.hset(key, field, value))
        if ttl_seconds is not None:
            await self._client.expire(key, ttl_seconds)

    async def hgetall(self, key: str) -> dict[str, str]:
        raw = await _resolve(self._client.hgetall(key))
        return {str(k): str(v) for k, v in raw.items()}

    async def get_float(self, key: str) -> float:
        raw = await self._client.get(key)
        return float(raw) if raw is not None else 0.0

    async def incr_float(
        self,
        key: str,
        amount: float,
        ttl_seconds: int | None = None,
    ) -> float:
        updated = float(await self._client.incrbyfloat(key, amount))
        if ttl_seconds is not None:
            ttl_remaining = await self._client.ttl(key)
            if ttl_remaining is not None and int(ttl_remaining) < 0:
                await self._client.expire(key, ttl_seconds)
        return updated

    async def window_sum(
        self,
        key: str,
        window_seconds: int,
        now_ts: float | None = None,
    ) -> int:
        now = now_ts if now_ts is not None else time.time()
        cutoff = now - window_seconds
        await self._client.zremrangebyscore(key, "-inf", cutoff)
        members = await self._client.zrange(key, 0, -1)
        total = 0
        for member in members:
            _, _, weight_str = str(member).partition(":")
            if weight_str:
                total += int(weight_str)
        return total

    async def window_add_if_under(
        self,
        key: str,
        window_seconds: int,
        limit: int,
        weight: int = 1,
        now_ts: float | None = None,
    ) -> tuple[bool, int]:
        now = now_ts if now_ts is not None else time.time()
        cutoff = now - window_seconds
        member = f"{uuid4().hex}:{weight}"
        raw_res = await _resolve(
            self._client.eval(
                _LUA_WINDOW_ADD_IF_UNDER,
                1,
                key,
                str(now),
                str(cutoff),
                str(limit),
                str(weight),
                member,
                str(window_seconds + 5),
            )
        )
        res = cast(list[object], raw_res)
        allowed = bool(int(str(res[0])))
        total = int(str(res[1]))
        return allowed, total

    async def window_add(
        self,
        key: str,
        window_seconds: int,
        weight: int,
        now_ts: float | None = None,
    ) -> int:
        now = now_ts if now_ts is not None else time.time()
        cutoff = now - window_seconds
        member = f"{uuid4().hex}:{weight}"
        await self._client.zremrangebyscore(key, "-inf", cutoff)
        await self._client.zadd(key, {member: now})
        await self._client.expire(key, window_seconds + 5)
        return await self.window_sum(key, window_seconds, now_ts=now)

    async def delete_prefix(self, prefix: str) -> int:
        keys = [str(k) async for k in self._client.scan_iter(match=f"{prefix}*")]
        if not keys:
            return 0
        deleted = await _resolve(self._client.delete(*keys))
        return int(deleted)

    @staticmethod
    def _sanitize_scope(scope: str) -> str:
        return _SCOPE_TAG_RE.sub("_", scope)

    async def _ensure_vector_index(self, dim: int) -> bool:
        if self._vector_index_ready is not None:
            return self._vector_index_ready
        try:
            await _exec(
                self._bytes_client,
                "FT.CREATE",
                _VECTOR_INDEX_NAME,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                "cache:vec:",
                "SCHEMA",
                "scope",
                "TAG",
                "vector",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(dim),
                "DISTANCE_METRIC",
                "COSINE",
            )
            self._vector_index_ready = True
        except ResponseError as exc:
            if "Index already exists" in str(exc):
                self._vector_index_ready = True
            else:
                # RediSearch module unavailable (e.g. plain Redis); fall back to hash scan.
                self._vector_index_ready = False
        return self._vector_index_ready

    async def vector_upsert(
        self,
        *,
        scope: str,
        doc_id: str,
        vector: list[float],
        payload_json: str,
        ttl_seconds: int,
    ) -> None:
        safe_scope = self._sanitize_scope(scope)
        if await self._ensure_vector_index(len(vector)):
            key = f"cache:vec:{safe_scope}:{doc_id}"
            vec_bytes = np.asarray(vector, dtype=np.float32).tobytes()
            await _resolve(
                self._bytes_client.hset(
                    key,
                    mapping={
                        b"scope": safe_scope.encode("utf-8"),
                        b"payload": payload_json.encode("utf-8"),
                        b"vector": vec_bytes,
                    },
                )
            )
            await self._bytes_client.expire(key, ttl_seconds)
            return

        # Fallback for non-RediSearch Redis servers
        record = json.dumps(
            {
                "vector": vector,
                "payload": payload_json,
                "expires_at": time.time() + ttl_seconds,
            }
        )
        await self.hset(f"cache:vec:{safe_scope}", doc_id, record, ttl_seconds=ttl_seconds)

    async def vector_search(
        self,
        *,
        scope: str,
        query_vector: list[float],
        top_k: int = 1,
    ) -> list[tuple[float, str]]:
        safe_scope = self._sanitize_scope(scope)
        if await self._ensure_vector_index(len(query_vector)):
            q_bytes = np.asarray(query_vector, dtype=np.float32).tobytes()
            query = f"(@scope:{{{safe_scope}}})=>[KNN {top_k} @vector $vec AS vector_score]"
            raw = await _exec(
                self._bytes_client,
                "FT.SEARCH",
                _VECTOR_INDEX_NAME,
                query,
                "PARAMS",
                "2",
                "vec",
                q_bytes,
                "SORTBY",
                "vector_score",
                "ASC",
                "RETURN",
                "2",
                "vector_score",
                "payload",
                "DIALECT",
                "2",
            )
            if not isinstance(raw, list) or len(raw) < 2:
                return []
            results: list[tuple[float, str]] = []
            # FT.SEARCH returns [total_count, key1, [field, val, ...], key2, ...]
            idx = 2
            while idx < len(raw):
                fields = raw[idx]
                idx += 2
                if not isinstance(fields, list):
                    continue
                field_map: dict[str, bytes] = {}
                for f_idx in range(0, len(fields) - 1, 2):
                    k_raw = fields[f_idx]
                    v_raw = fields[f_idx + 1]
                    k_str = k_raw.decode("utf-8") if isinstance(k_raw, bytes) else str(k_raw)
                    v_bytes = v_raw if isinstance(v_raw, bytes) else str(v_raw).encode("utf-8")
                    field_map[k_str] = v_bytes
                score_raw = field_map.get("vector_score")
                payload_raw = field_map.get("payload")
                if score_raw is not None and payload_raw is not None:
                    cosine_dist = float(score_raw.decode("utf-8"))
                    similarity = max(0.0, min(1.0, 1.0 - cosine_dist))
                    results.append((similarity, payload_raw.decode("utf-8")))
            return results

        raw_records = await self.hgetall(f"cache:vec:{safe_scope}")
        if not raw_records:
            return []
        now = time.time()
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        scored: list[tuple[float, str]] = []
        for raw_json in raw_records.values():
            obj = json.loads(raw_json)
            if now >= float(obj["expires_at"]):
                continue
            v = np.asarray(obj["vector"], dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0.0 or v.shape != q.shape:
                continue
            score = float(np.dot(q, v) / (q_norm * v_norm))
            scored.append((score, str(obj["payload"])))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    async def close(self) -> None:
        await self._client.aclose()
        await self._bytes_client.aclose()


def build_state_store(redis_url: str) -> StateStore:
    """Instantiate the appropriate StateStore from REDIS_URL."""
    if redis_url.startswith("memory://"):
        return InMemoryStore()
    return RedisStore(redis_url)
