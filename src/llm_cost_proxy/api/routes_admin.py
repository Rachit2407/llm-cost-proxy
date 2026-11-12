"""Health check, Prometheus metrics, model catalog, budget, and cache admin routes."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from llm_cost_proxy.api.dashboard import DASHBOARD_HTML
from llm_cost_proxy.config import Settings
from llm_cost_proxy.core.auth import AuthenticationError, resolve_auth_context
from llm_cost_proxy.core.pricing import MODEL_PRICING_TABLE
from llm_cost_proxy.core.rate_limiter import RateLimiter
from llm_cost_proxy.core.redis_store import RedisStore, StateStore
from llm_cost_proxy.db.session import TelemetryRepository
from llm_cost_proxy.models.schemas import (
    ModelCatalogEntry,
    ModelCatalogResponse,
    TelemetryStats,
    TenantBudgetStatus,
)

router = APIRouter(tags=["admin"])


@router.get("/dashboard", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def dashboard_ui() -> HTMLResponse:
    """Serve the modern dark-themed live proxy dashboard UI."""
    return HTMLResponse(content=DASHBOARD_HTML)


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Liveness and state-store connectivity probe."""
    store = cast(StateStore, request.app.state.state_store)
    store_ok = await store.ping()
    return {
        "status": "ok" if store_ok else "degraded",
        "state_store": "connected" if store_ok else "unreachable",
    }


@router.get("/v1/stats", response_model=TelemetryStats)
async def get_stats(
    request: Request,
    tenant_id: str | None = Query(default=None),
) -> TelemetryStats:
    """Return aggregated request, cache-hit, spend, and cost-savings metrics."""
    telemetry = cast(TelemetryRepository, request.app.state.telemetry_repo)
    return await telemetry.get_stats(tenant_id=tenant_id)


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(request: Request) -> PlainTextResponse:
    """Expose Prometheus-compatible text exposition metrics (v0.0.4)."""
    telemetry = cast(TelemetryRepository, request.app.state.telemetry_repo)
    stats = await telemetry.get_stats()
    lines = [
        "# HELP llm_proxy_requests_total Total proxy requests by cache outcome.",
        "# TYPE llm_proxy_requests_total counter",
        f'llm_proxy_requests_total{{status="exact_hit"}} {stats.exact_hits}',
        f'llm_proxy_requests_total{{status="semantic_hit"}} {stats.semantic_hits}',
        f'llm_proxy_requests_total{{status="miss"}} {stats.cache_misses}',
        f'llm_proxy_requests_total{{status="blocked"}} {stats.blocked_requests}',
        "# HELP llm_proxy_cache_hit_ratio Cache hit ratio across cacheable requests.",
        "# TYPE llm_proxy_cache_hit_ratio gauge",
        f"llm_proxy_cache_hit_ratio {stats.hit_rate}",
        "# HELP llm_proxy_tokens_total Upstream tokens consumed by token type.",
        "# TYPE llm_proxy_tokens_total counter",
        f'llm_proxy_tokens_total{{type="prompt"}} {stats.total_prompt_tokens}',
        f'llm_proxy_tokens_total{{type="completion"}} {stats.total_completion_tokens}',
        "# HELP llm_proxy_spend_usd_total Total upstream LLM spend in USD.",
        "# TYPE llm_proxy_spend_usd_total counter",
        f"llm_proxy_spend_usd_total {stats.total_spend_usd}",
        "# HELP llm_proxy_saved_usd_total Total estimated USD saved by cache hits.",
        "# TYPE llm_proxy_saved_usd_total counter",
        f"llm_proxy_saved_usd_total {stats.total_saved_usd}",
    ]
    return PlainTextResponse(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/v1/models", response_model=ModelCatalogResponse)
async def list_models() -> ModelCatalogResponse:
    """List supported OpenAI and Anthropic models with configured pricing metadata."""
    entries = [
        ModelCatalogEntry(
            id=model_id,
            owned_by="anthropic" if model_id.startswith("claude") else "openai",
            input_per_1m_usd=pricing.input_per_1m_usd,
            output_per_1m_usd=pricing.output_per_1m_usd,
        )
        for model_id, pricing in MODEL_PRICING_TABLE.items()
    ]
    return ModelCatalogResponse(data=entries)


@router.get("/v1/budget", response_model=TenantBudgetStatus)
async def get_tenant_budget(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> TenantBudgetStatus | JSONResponse:
    """Return live RPM/TPM sliding-window usage and monthly USD budget status."""
    settings = cast(Settings, request.app.state.settings)
    rate_limiter = cast(RateLimiter, request.app.state.rate_limiter)
    try:
        auth = resolve_auth_context(
            settings=settings,
            provider="openai",
            authorization=authorization,
            x_api_key=x_api_key,
            x_upstream_api_key=None,
        )
    except AuthenticationError as exc:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": str(exc), "type": "authentication_error"}},
        )
    return await rate_limiter.get_budget_status(auth.policy)


@router.delete("/v1/cache")
async def purge_tenant_cache(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    """Purge all L1 exact and L2 semantic cache entries for the authenticated tenant."""
    settings = cast(Settings, request.app.state.settings)
    store = cast(StateStore, request.app.state.state_store)
    try:
        auth = resolve_auth_context(
            settings=settings,
            provider="openai",
            authorization=authorization,
            x_api_key=x_api_key,
            x_upstream_api_key=None,
        )
    except AuthenticationError as exc:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": str(exc), "type": "authentication_error"}},
        )

    tenant_id = auth.policy.tenant_id
    safe_tenant = RedisStore._sanitize_scope(tenant_id)
    deleted_exact = await store.delete_prefix(f"cache:exact:{tenant_id}:")
    deleted_vec_colon = await store.delete_prefix(f"cache:vec:{tenant_id}:")
    deleted_vec_under = await store.delete_prefix(f"cache:vec:{safe_tenant}_")
    return JSONResponse(
        status_code=200,
        content={
            "tenant_id": tenant_id,
            "deleted_exact_keys": deleted_exact,
            "deleted_semantic_buckets": deleted_vec_colon + deleted_vec_under,
        },
    )
