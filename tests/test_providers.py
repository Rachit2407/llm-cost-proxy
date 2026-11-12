"""Unit tests for OpenAI/Anthropic HTTP adapters and budget alert webhooks."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_cost_proxy.core.alerts import WebhookAlertNotifier
from llm_cost_proxy.core.redis_store import InMemoryStore
from llm_cost_proxy.models.schemas import CanonicalRequest, TenantPolicy
from llm_cost_proxy.providers.anthropic import AnthropicProvider
from llm_cost_proxy.providers.base import UpstreamError
from llm_cost_proxy.providers.openai import OpenAIProvider


def _openai_req(*, stream: bool = False) -> CanonicalRequest:
    return CanonicalRequest(
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=stream,
        raw_payload={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": stream,
        },
    )


async def test_openai_provider_json_and_streaming_and_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(request.content.decode("utf-8"))
        if body.get("stream"):
            sse = (
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi "}}]}\n\n'
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"there"}}],'
                '"usage":{"prompt_tokens":12,"completion_tokens":4}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-sync",
                "choices": [{"message": {"role": "assistant", "content": "Pong"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    provider = OpenAIProvider(transport=httpx.MockTransport(handler))
    sync_resp = await provider.complete(_openai_req(stream=False), api_key="sk-test")
    assert sync_resp.status_code == 200
    assert sync_resp.usage.prompt_tokens == 10
    assert sync_resp.usage.completion_tokens == 5

    stream_resp = await provider.complete(_openai_req(stream=True), api_key="sk-test")
    assert stream_resp.status_code == 200
    assert stream_resp.usage.prompt_tokens == 12
    assert stream_resp.usage.completion_tokens == 4
    assert stream_resp.raw_sse_chunks is not None

    with pytest.raises(UpstreamError) as exc_info:
        await provider.complete(_openai_req(stream=False), api_key="")
    assert exc_info.value.status_code == 401


async def test_anthropic_provider_headers_usage_and_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "sk-ant-test"
        assert request.headers["anthropic-version"] == "2023-06-01"
        if request.url.path == "/v1/messages":
            return httpx.Response(
                200,
                json={
                    "id": "msg-1",
                    "content": [{"type": "text", "text": "Anthropic OK"}],
                    "usage": {"input_tokens": 25, "output_tokens": 15},
                },
            )
        return httpx.Response(404, text="not found")

    provider = AnthropicProvider(transport=httpx.MockTransport(handler))
    req = CanonicalRequest(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Ping"}],
        raw_payload={"model": "claude-3-5-haiku-20241022", "max_tokens": 64, "messages": []},
    )
    resp = await provider.complete(req, api_key="sk-ant-test")
    assert resp.usage.prompt_tokens == 25
    assert resp.usage.completion_tokens == 15

    with pytest.raises(UpstreamError) as exc_info:
        await provider.complete(req, api_key="")
    assert exc_info.value.status_code == 401


async def test_webhook_alert_notifier_threshold_and_deduplication(
    memory_store: InMemoryStore,
) -> None:
    sent_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"ok": True})

    notifier = WebhookAlertNotifier(
        store=memory_store,
        webhook_url="https://hooks.example.test/slack",
        threshold_ratio=0.80,
        transport=httpx.MockTransport(handler),
    )
    policy = TenantPolicy(tenant_id="acme", monthly_budget_usd=10.0)

    # Below 80% -> no alert
    assert await notifier.check_and_notify(policy=policy, current_spend_usd=5.0) is False
    assert len(sent_payloads) == 0

    # Crosses 80% -> sends once
    assert await notifier.check_and_notify(policy=policy, current_spend_usd=8.5) is True
    assert len(sent_payloads) == 1
    assert sent_payloads[0]["tenant_id"] == "acme"

    # Subsequent calls in same month are deduplicated
    assert await notifier.check_and_notify(policy=policy, current_spend_usd=9.5) is False
    assert len(sent_payloads) == 1
