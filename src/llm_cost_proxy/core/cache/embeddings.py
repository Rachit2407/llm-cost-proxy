"""Pluggable text embedders and cosine similarity utilities for semantic caching."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np

from llm_cost_proxy.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "for",
        "from",
        "i",
        "in",
        "into",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "tell",
        "that",
        "the",
        "this",
        "to",
        "using",
        "what",
        "with",
        "you",
        "your",
    }
)


class Embedder(Protocol):
    """Protocol for producing L2-normalized embedding vectors."""

    @property
    def dimension(self) -> int:
        ...

    async def embed(self, text: str) -> list[float]:
        ...


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class LocalHashEmbedder:
    """Zero-dependency deterministic feature-hashing embedder (word + char-trigram).

    Uses signed feature hashing into a fixed-dimension L2-normalized vector so
    paraphrased or slightly varied prompts achieve high cosine similarity without
    requiring external API calls or heavyweight ML runtimes.
    """

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _accumulate(self, vec: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw_int = int.from_bytes(digest, "little")
        bucket = raw_int % self._dimension
        sign = 1.0 if ((raw_int >> 63) & 1) == 0 else -1.0
        vec[bucket] += sign * weight

    async def embed(self, text: str) -> list[float]:
        normalized = " ".join(_TOKEN_RE.findall(text.lower()))
        vec = np.zeros(self._dimension, dtype=np.float32)
        if not normalized:
            return vec.tolist()

        words = normalized.split()
        content_words = [w for w in words if w not in _STOPWORDS]
        active_words = content_words if content_words else words
        for word in active_words:
            self._accumulate(vec, f"w:{word}", 2.0)
            if len(word) >= 3:
                for idx in range(len(word) - 2):
                    self._accumulate(vec, f"c:{word[idx : idx + 3]}", 0.6)

        for idx in range(len(active_words) - 1):
            self._accumulate(vec, f"b:{active_words[idx]}_{active_words[idx + 1]}", 0.5)

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        result: list[float] = vec.tolist()
        return result


class OpenAIEmbedder:
    """Async OpenAI /v1/embeddings adapter with L2 normalization."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com",
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._timeout = timeout_seconds

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY must be set when EMBEDDING_PROVIDER=openai")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self._model, "input": text},
            )
            response.raise_for_status()
            payload = response.json()
        raw_vec = payload["data"][0]["embedding"]
        arr = np.asarray(raw_vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            arr /= norm
        result: list[float] = arr.tolist()
        return result


def build_embedder(settings: Settings) -> Embedder:
    """Instantiate the configured Embedder."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
            timeout_seconds=settings.upstream_timeout_seconds,
        )
    return LocalHashEmbedder(dimension=settings.embedding_dim)
