"""Shared pytest fixtures for llm-cost-proxy unit and integration tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from llm_cost_proxy.config import Settings
from llm_cost_proxy.core.redis_store import InMemoryStore
from llm_cost_proxy.models.schemas import CanonicalRequest, ProviderResponse, TokenUsage


class FakeLLMProvider:
    """Deterministic mock LLM provider counting upstream invocations."""

    def __init__(
        self,
        *,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        reply_text: str = "Cached-capable response",
    ) -> None:
        self.calls: list[CanonicalRequest] = []
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reply_text = reply_text

    async def complete(
        self,
        request: CanonicalRequest,
        api_key: str,
    ) -> ProviderResponse:
        self.calls.append(request)
        if request.provider == "openai":
            body: dict[str, object] = {
                "id": f"chatcmpl-test-{len(self.calls)}",
                "object": "chat.completion",
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.reply_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.prompt_tokens + self.completion_tokens,
                },
            }
        else:
            body = {
                "id": f"msg-test-{len(self.calls)}",
                "type": "message",
                "role": "assistant",
                "model": request.model,
                "content": [{"type": "text", "text": self.reply_text}],
                "usage": {
                    "input_tokens": self.prompt_tokens,
                    "output_tokens": self.completion_tokens,
                },
            }
        return ProviderResponse(
            status_code=200,
            body=body,
            usage=TokenUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            ),
        )


@pytest.fixture
def memory_store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
async def sqlite_settings(tmp_path: Path) -> AsyncIterator[Settings]:
    db_file = tmp_path / "telemetry_test.db"
    yield Settings(
        database_url=f"sqlite+aiosqlite:///{db_file.as_posix()}",
        redis_url="memory://",
        allow_anonymous_dev=True,
        semantic_similarity_threshold=0.80,
    )
