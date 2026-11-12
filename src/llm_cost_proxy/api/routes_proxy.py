"""OpenAI- and Anthropic-compatible proxy endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from llm_cost_proxy.config import Settings
from llm_cost_proxy.core.auth import AuthenticationError, resolve_auth_context
from llm_cost_proxy.core.proxy_service import PolicyBlockedError, ProxyService
from llm_cost_proxy.models.schemas import (
    AnthropicMessageRequest,
    CanonicalRequest,
    OpenAIChatRequest,
    ProviderName,
    ProxyExecutionResult,
)
from llm_cost_proxy.providers.base import UpstreamError

router = APIRouter(tags=["proxy"])


def _should_bypass_cache(x_cache_control: str | None, cache_control: str | None) -> bool:
    for header in (x_cache_control, cache_control):
        if header and "no-cache" in header.lower():
            return True
    return False


def _proxy_headers(result: ProxyExecutionResult) -> dict[str, str]:
    headers: dict[str, str] = {
        "X-Proxy-Cache": result.cache_status,
        "X-Proxy-Upstream-Cost-USD": f"{result.upstream_cost_usd:.8f}",
        "X-Proxy-Cost-Saved-USD": f"{result.cost_saved_usd:.8f}",
    }
    if result.similarity is not None:
        headers["X-Proxy-Similarity"] = f"{result.similarity:.4f}"
    if result.downgraded_from and result.model_used:
        headers["X-Proxy-Model-Downgraded"] = f"{result.downgraded_from}->{result.model_used}"
    return headers


def _extract_anthropic_system(system: str | list[dict[str, object]] | None) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = [
            str(block.get("text", ""))
            for block in system
            if isinstance(block, dict) and "text" in block
        ]
        return "\n".join(parts)
    return ""


async def _handle_proxy_request(
    *,
    fastapi_request: Request,
    provider: ProviderName,
    canonical: CanonicalRequest,
    authorization: str | None,
    x_api_key: str | None,
    x_upstream_api_key: str | None,
    x_cache_control: str | None,
    cache_control: str | None,
) -> Response:
    settings = cast(Settings, fastapi_request.app.state.settings)
    service = cast(ProxyService, fastapi_request.app.state.proxy_service)

    try:
        auth = resolve_auth_context(
            settings=settings,
            provider=provider,
            authorization=authorization,
            x_api_key=x_api_key,
            x_upstream_api_key=x_upstream_api_key,
        )
    except AuthenticationError as exc:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": str(exc), "type": "authentication_error"}},
        )

    bypass_cache = _should_bypass_cache(x_cache_control, cache_control)

    try:
        result = await service.execute(
            request=canonical,
            auth=auth,
            bypass_cache=bypass_cache,
        )
    except PolicyBlockedError as exc:
        decision = exc.decision
        status_code = 402 if decision.reason == "budget_exceeded" else 429
        return JSONResponse(
            status_code=status_code,
            headers={"Retry-After": str(decision.retry_after_seconds), "X-Proxy-Cache": "BLOCKED"},
            content={
                "error": {
                    "message": f"Policy limit reached: {decision.reason}",
                    "type": decision.reason,
                    "current_spend_usd": decision.current_spend_usd,
                    "budget_limit_usd": decision.budget_limit_usd,
                }
            },
        )
    except UpstreamError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": "upstream_error",
                    "provider": exc.provider,
                }
            },
        )

    headers = _proxy_headers(result)
    if canonical.stream and result.stream_sse_lines is not None:
        lines = result.stream_sse_lines

        async def _sse_generator() -> AsyncIterator[str]:
            for line in lines:
                yield f"{line}\n\n"

        return StreamingResponse(
            _sse_generator(),
            status_code=result.status_code,
            media_type="text/event-stream",
            headers=headers,
        )

    return JSONResponse(
        status_code=result.status_code,
        content=result.body,
        headers=headers,
    )


@router.post("/v1/chat/completions")
async def openai_chat_completions(
    payload: OpenAIChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_upstream_api_key: str | None = Header(default=None),
    x_cache_control: str | None = Header(default=None),
    cache_control: str | None = Header(default=None),
) -> Response:
    """OpenAI-compatible /v1/chat/completions proxy endpoint."""
    raw_messages = [msg.model_dump() for msg in payload.messages]
    canonical = CanonicalRequest(
        provider="openai",
        model=payload.model,
        messages=raw_messages,
        system_prompt="",
        temperature=payload.temperature if payload.temperature is not None else 0.0,
        max_tokens=payload.max_tokens,
        stream=payload.stream,
        raw_payload=payload.model_dump(exclude_none=True),
    )
    return await _handle_proxy_request(
        fastapi_request=request,
        provider="openai",
        canonical=canonical,
        authorization=authorization,
        x_api_key=x_api_key,
        x_upstream_api_key=x_upstream_api_key,
        x_cache_control=x_cache_control,
        cache_control=cache_control,
    )


@router.post("/v1/messages")
async def anthropic_messages(
    payload: AnthropicMessageRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_upstream_api_key: str | None = Header(default=None),
    x_cache_control: str | None = Header(default=None),
    cache_control: str | None = Header(default=None),
) -> Response:
    """Anthropic-compatible /v1/messages proxy endpoint."""
    raw_messages = [msg.model_dump() for msg in payload.messages]
    canonical = CanonicalRequest(
        provider="anthropic",
        model=payload.model,
        messages=raw_messages,
        system_prompt=_extract_anthropic_system(payload.system),
        temperature=payload.temperature if payload.temperature is not None else 0.0,
        max_tokens=payload.max_tokens,
        stream=payload.stream,
        raw_payload=payload.model_dump(exclude_none=True),
    )
    return await _handle_proxy_request(
        fastapi_request=request,
        provider="anthropic",
        canonical=canonical,
        authorization=authorization,
        x_api_key=x_api_key,
        x_upstream_api_key=x_upstream_api_key,
        x_cache_control=x_cache_control,
        cache_control=cache_control,
    )
