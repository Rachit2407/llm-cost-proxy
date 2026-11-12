"""Main proxy lifecycle orchestrator: Rate Limit -> L1 Exact -> L2 Semantic -> Upstream."""

from __future__ import annotations

import json
import time
from uuid import uuid4

from llm_cost_proxy.config import Settings
from llm_cost_proxy.core.alerts import BudgetAlertNotifier
from llm_cost_proxy.core.auth import AuthContext
from llm_cost_proxy.core.cache.exact import ExactCache
from llm_cost_proxy.core.cache.semantic import SemanticCache
from llm_cost_proxy.core.pricing import (
    calculate_cost_usd,
    estimate_input_tokens,
    get_downgrade_model,
)
from llm_cost_proxy.core.rate_limiter import RateLimiter
from llm_cost_proxy.db.session import TelemetryRepository
from llm_cost_proxy.models.schemas import (
    CachedEntry,
    CacheStatus,
    CanonicalRequest,
    ProviderName,
    ProxyExecutionResult,
    RateLimitDecision,
)
from llm_cost_proxy.providers.base import LLMProvider


class PolicyBlockedError(Exception):
    """Raised when RPM, TPM, or monthly USD budget cap blocks a request."""

    def __init__(self, decision: RateLimitDecision) -> None:
        self.decision = decision
        super().__init__(f"Request blocked by policy: {decision.reason}")


def _build_sse_replay(request: CanonicalRequest, body: dict[str, object]) -> list[str]:
    """Convert a cached JSON completion into OpenAI or Anthropic SSE data lines."""
    if request.provider == "openai":
        text = ""
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                text = msg["content"]
        chunk = {
            "id": str(body.get("id", "chatcmpl-cached")),
            "object": "chat.completion.chunk",
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
        return [f"data: {json.dumps(chunk)}", "data: [DONE]"]

    # Anthropic SSE replay
    text = ""
    content = body.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and isinstance(first.get("text"), str):
            text = first["text"]
    delta_event = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    }
    stop_event = {"type": "message_stop"}
    return [
        f"event: content_block_delta\ndata: {json.dumps(delta_event)}",
        f"event: message_stop\ndata: {json.dumps(stop_event)}",
    ]


class ProxyService:
    """Coordinates auth policy enforcement, caching, upstream routing, and telemetry."""

    def __init__(
        self,
        *,
        settings: Settings,
        rate_limiter: RateLimiter,
        exact_cache: ExactCache,
        semantic_cache: SemanticCache,
        providers: dict[ProviderName, LLMProvider],
        telemetry: TelemetryRepository,
        alert_notifier: BudgetAlertNotifier | None = None,
    ) -> None:
        self._settings = settings
        self._rate_limiter = rate_limiter
        self._exact_cache = exact_cache
        self._semantic_cache = semantic_cache
        self._providers = providers
        self._telemetry = telemetry
        self._alert_notifier = alert_notifier

    async def execute(
        self,
        *,
        request: CanonicalRequest,
        auth: AuthContext,
        bypass_cache: bool = False,
    ) -> ProxyExecutionResult:
        """Run the full proxy pipeline for a normalized LLM request."""
        start_ns = time.perf_counter_ns()
        request_id = uuid4().hex
        policy = auth.policy

        # 1. Pre-flight rate limit & budget check (before any embedding or upstream call)
        estimated_tokens = estimate_input_tokens(request.prompt_text())
        decision = await self._rate_limiter.check_request(
            policy=policy,
            estimated_input_tokens=estimated_tokens,
        )
        if not decision.allowed:
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            await self._telemetry.record_event(
                request_id=request_id,
                tenant_id=policy.tenant_id,
                provider=request.provider,
                model=request.model,
                cache_status="BLOCKED",
                similarity=None,
                prompt_tokens=0,
                completion_tokens=0,
                upstream_cost_usd=0.0,
                cost_saved_usd=0.0,
                latency_ms=round(latency_ms, 3),
            )
            raise PolicyBlockedError(decision)

        downgraded_from: str | None = None
        if (
            self._settings.enable_budget_downgrade
            and policy.monthly_budget_usd > 0.0
            and (decision.current_spend_usd / policy.monthly_budget_usd)
            >= self._settings.budget_downgrade_ratio
        ):
            cheaper_model = get_downgrade_model(request.model)
            if cheaper_model is not None:
                downgraded_from = request.model
                request = request.model_copy(
                    update={
                        "model": cheaper_model,
                        "raw_payload": {**request.raw_payload, "model": cheaper_model},
                    }
                )

        cache_eligible = (
            not bypass_cache
            and request.temperature <= self._settings.max_cache_temperature
        )

        # 2. L1 Exact Cache & L2 Semantic Cache lookups
        if cache_eligible:
            exact_res = await self._exact_cache.lookup(policy.tenant_id, request)
            if exact_res.hit and exact_res.entry is not None:
                return await self._finalize_cache_hit(
                    request_id=request_id,
                    tenant_id=policy.tenant_id,
                    request=request,
                    entry=exact_res.entry,
                    status="EXACT_HIT",
                    similarity=1.0,
                    start_ns=start_ns,
                    downgraded_from=downgraded_from,
                )

            semantic_res = await self._semantic_cache.lookup(
                policy.tenant_id,
                request,
                threshold=policy.semantic_threshold,
            )
            if semantic_res.hit and semantic_res.entry is not None:
                return await self._finalize_cache_hit(
                    request_id=request_id,
                    tenant_id=policy.tenant_id,
                    request=request,
                    entry=semantic_res.entry,
                    status="SEMANTIC_HIT",
                    similarity=semantic_res.similarity,
                    start_ns=start_ns,
                    downgraded_from=downgraded_from,
                )

        # 3. Cache Miss -> Upstream provider
        provider = self._providers[request.provider]
        upstream_resp = await provider.complete(request, api_key=auth.upstream_api_key)
        actual_cost = calculate_cost_usd(request.model, upstream_resp.usage)

        new_spend = await self._rate_limiter.record_upstream_usage(
            policy=policy,
            total_tokens=upstream_resp.usage.total_tokens,
            cost_usd=actual_cost,
        )
        if self._alert_notifier is not None:
            await self._alert_notifier.check_and_notify(
                policy=policy,
                current_spend_usd=new_spend,
            )

        if cache_eligible and 200 <= upstream_resp.status_code < 300:
            cached_entry = CachedEntry(
                provider=request.provider,
                model=request.model,
                response_body=upstream_resp.body,
                usage=upstream_resp.usage,
                estimated_cost_usd=actual_cost,
            )
            await self._exact_cache.store(
                policy.tenant_id,
                request,
                cached_entry,
                ttl_seconds=policy.cache_ttl_seconds,
            )
            await self._semantic_cache.store(
                policy.tenant_id,
                request,
                cached_entry,
                ttl_seconds=policy.cache_ttl_seconds,
            )

        status: CacheStatus = "BYPASS" if not cache_eligible else "MISS"
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        await self._telemetry.record_event(
            request_id=request_id,
            tenant_id=policy.tenant_id,
            provider=request.provider,
            model=request.model,
            cache_status=status,
            similarity=None,
            prompt_tokens=upstream_resp.usage.prompt_tokens,
            completion_tokens=upstream_resp.usage.completion_tokens,
            upstream_cost_usd=actual_cost,
            cost_saved_usd=0.0,
            latency_ms=round(latency_ms, 3),
        )

        sse_lines = upstream_resp.raw_sse_chunks
        if request.stream and sse_lines is None:
            sse_lines = _build_sse_replay(request, upstream_resp.body)

        return ProxyExecutionResult(
            status_code=upstream_resp.status_code,
            body=upstream_resp.body,
            cache_status=status,
            similarity=None,
            upstream_cost_usd=actual_cost,
            cost_saved_usd=0.0,
            usage=upstream_resp.usage,
            stream_sse_lines=sse_lines,
            downgraded_from=downgraded_from,
            model_used=request.model,
        )

    async def _finalize_cache_hit(
        self,
        *,
        request_id: str,
        tenant_id: str,
        request: CanonicalRequest,
        entry: CachedEntry,
        status: CacheStatus,
        similarity: float | None,
        start_ns: int,
        downgraded_from: str | None = None,
    ) -> ProxyExecutionResult:
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        await self._telemetry.record_event(
            request_id=request_id,
            tenant_id=tenant_id,
            provider=request.provider,
            model=request.model,
            cache_status=status,
            similarity=similarity,
            prompt_tokens=0,
            completion_tokens=0,
            upstream_cost_usd=0.0,
            cost_saved_usd=entry.estimated_cost_usd,
            latency_ms=round(latency_ms, 3),
        )
        sse_lines = _build_sse_replay(request, entry.response_body) if request.stream else None
        return ProxyExecutionResult(
            status_code=200,
            body=entry.response_body,
            cache_status=status,
            similarity=similarity,
            upstream_cost_usd=0.0,
            cost_saved_usd=entry.estimated_cost_usd,
            usage=entry.usage,
            stream_sse_lines=sse_lines,
            downgraded_from=downgraded_from,
            model_used=request.model,
        )
