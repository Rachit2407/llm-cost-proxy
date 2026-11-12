"""Async OpenAI /v1/chat/completions upstream adapter."""

from __future__ import annotations

import json

import httpx

from llm_cost_proxy.core.pricing import estimate_input_tokens
from llm_cost_proxy.models.schemas import (
    CanonicalRequest,
    ProviderResponse,
    TokenUsage,
)
from llm_cost_proxy.providers.base import UpstreamError


def _parse_int(value: object, default: int = 0) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return default


class OpenAIProvider:
    """Forwards chat completion requests to OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com",
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
                provider="openai",
                status_code=401,
                message=(
                    "Missing upstream OpenAI API key "
                    "(set OPENAI_API_KEY or X-Upstream-API-Key)."
                ),
            )

        url = f"{self._base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = dict(request.raw_payload)
        if request.stream:
            return await self._complete_streaming(url, headers, payload, request)

        async with httpx.AsyncClient(
            timeout=self._timeout,
            transport=self._transport,
        ) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code >= 400:
            raise UpstreamError(
                provider="openai",
                status_code=response.status_code,
                message=response.text,
            )

        body: dict[str, object] = response.json()
        usage_raw = body.get("usage")
        if isinstance(usage_raw, dict):
            usage = TokenUsage(
                prompt_tokens=_parse_int(usage_raw.get("prompt_tokens")),
                completion_tokens=_parse_int(usage_raw.get("completion_tokens")),
            )
        else:
            usage = TokenUsage(
                prompt_tokens=estimate_input_tokens(request.prompt_text()),
                completion_tokens=0,
            )
        return ProviderResponse(status_code=response.status_code, body=body, usage=usage)

    async def _complete_streaming(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        request: CanonicalRequest,
    ) -> ProviderResponse:
        stream_payload = dict(payload)
        stream_options = stream_payload.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_payload["stream_options"] = {"include_usage": True}

        sse_lines: list[str] = []
        content_parts: list[str] = []
        usage = TokenUsage(prompt_tokens=estimate_input_tokens(request.prompt_text()))
        response_id = "chatcmpl-proxy"

        async with (
            httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client,
            client.stream("POST", url, headers=headers, json=stream_payload) as resp,
        ):
            if resp.status_code >= 400:
                err_body = await resp.aread()
                raise UpstreamError(
                    provider="openai",
                    status_code=resp.status_code,
                    message=err_body.decode("utf-8", errors="replace"),
                )
            async for line in resp.aiter_lines():
                if not line:
                    continue
                sse_lines.append(line)
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    if "id" in chunk:
                        response_id = str(chunk["id"])
                    choices = chunk.get("choices")
                    if isinstance(choices, list) and choices:
                        first_choice = choices[0]
                        delta = (
                            first_choice.get("delta")
                            if isinstance(first_choice, dict)
                            else None
                        )
                        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                            content_parts.append(delta["content"])
                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage = TokenUsage(
                            prompt_tokens=_parse_int(chunk_usage.get("prompt_tokens")),
                            completion_tokens=_parse_int(chunk_usage.get("completion_tokens")),
                        )

        full_text = "".join(content_parts)
        if usage.completion_tokens == 0 and full_text:
            usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=estimate_input_tokens(full_text),
            )
        reconstructed_body: dict[str, object] = {
            "id": response_id,
            "object": "chat.completion",
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }
        return ProviderResponse(
            status_code=200,
            body=reconstructed_body,
            usage=usage,
            raw_sse_chunks=sse_lines,
        )
