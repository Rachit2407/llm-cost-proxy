"""Async Anthropic /v1/messages upstream adapter."""

from __future__ import annotations

import httpx

from llm_cost_proxy.core.pricing import estimate_input_tokens
from llm_cost_proxy.models.schemas import (
    CanonicalRequest,
    ProviderResponse,
    TokenUsage,
)
from llm_cost_proxy.providers.base import UpstreamError

_ANTHROPIC_VERSION = "2023-06-01"


def _parse_int(value: object, default: int = 0) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return default


class AnthropicProvider:
    """Forwards message requests to Anthropic Messages API."""

    def __init__(
        self,
        base_url: str = "https://api.anthropic.com",
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport

    async def complete(
        self,
        request: CanonicalRequest,
        api_key: str,
    ) -> ProviderResponse:
        if not api_key:
            raise UpstreamError(
                provider="anthropic",
                status_code=401,
                message=(
                    "Missing upstream Anthropic API key "
                    "(set ANTHROPIC_API_KEY or X-Upstream-API-Key)."
                ),
            )

        url = f"{self._base_url}/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        payload = dict(request.raw_payload)
        payload["stream"] = False  # Proxy executes canonical JSON call & replays SSE if requested.

        async with httpx.AsyncClient(
            timeout=self._timeout,
            transport=self._transport,
        ) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code >= 400:
            raise UpstreamError(
                provider="anthropic",
                status_code=response.status_code,
                message=response.text,
            )

        body: dict[str, object] = response.json()
        usage_raw = body.get("usage")
        if isinstance(usage_raw, dict):
            usage = TokenUsage(
                prompt_tokens=_parse_int(usage_raw.get("input_tokens")),
                completion_tokens=_parse_int(usage_raw.get("output_tokens")),
            )
        else:
            usage = TokenUsage(
                prompt_tokens=estimate_input_tokens(request.prompt_text()),
                completion_tokens=0,
            )
        return ProviderResponse(status_code=response.status_code, body=body, usage=usage)
