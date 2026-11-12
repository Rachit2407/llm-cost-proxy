"""FastAPI application factory and lifespan management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_cost_proxy.api.routes_admin import router as admin_router
from llm_cost_proxy.api.routes_proxy import router as proxy_router
from llm_cost_proxy.config import Settings
from llm_cost_proxy.core.alerts import BudgetAlertNotifier, build_alert_notifier
from llm_cost_proxy.core.cache.embeddings import Embedder, build_embedder
from llm_cost_proxy.core.cache.exact import ExactCache
from llm_cost_proxy.core.cache.semantic import SemanticCache
from llm_cost_proxy.core.proxy_service import ProxyService
from llm_cost_proxy.core.rate_limiter import RateLimiter
from llm_cost_proxy.core.redis_store import StateStore, build_state_store
from llm_cost_proxy.db.session import (
    TelemetryRepository,
    create_engine_and_session_factory,
    init_db,
)
from llm_cost_proxy.models.schemas import ProviderName
from llm_cost_proxy.providers.anthropic import AnthropicProvider
from llm_cost_proxy.providers.base import LLMProvider
from llm_cost_proxy.providers.openai import OpenAIProvider


def create_app(
    *,
    settings: Settings | None = None,
    state_store: StateStore | None = None,
    embedder: Embedder | None = None,
    providers: dict[ProviderName, LLMProvider] | None = None,
    alert_notifier: BudgetAlertNotifier | None = None,
) -> FastAPI:
    """Build and configure the FastAPI proxy application."""
    cfg = settings or Settings()
    store = state_store or build_state_store(cfg.redis_url)
    active_embedder = embedder or build_embedder(cfg)
    engine, session_factory = create_engine_and_session_factory(cfg.database_url)
    telemetry_repo = TelemetryRepository(session_factory)
    rate_limiter = RateLimiter(store)
    notifier = alert_notifier or build_alert_notifier(cfg, store)

    active_providers: dict[ProviderName, LLMProvider] = providers or {
        "openai": OpenAIProvider(
            base_url=cfg.openai_base_url,
            timeout_seconds=cfg.upstream_timeout_seconds,
        ),
        "anthropic": AnthropicProvider(
            base_url=cfg.anthropic_base_url,
            timeout_seconds=cfg.upstream_timeout_seconds,
        ),
    }

    proxy_service = ProxyService(
        settings=cfg,
        rate_limiter=rate_limiter,
        exact_cache=ExactCache(store),
        semantic_cache=SemanticCache(
            store,
            active_embedder,
            max_candidates_per_scope=cfg.semantic_max_candidates_per_scope,
        ),
        providers=active_providers,
        telemetry=telemetry_repo,
        alert_notifier=notifier,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await init_db(engine)
        try:
            yield
        finally:
            await store.close()
            await engine.dispose()

    app = FastAPI(
        title="LLM Cost-Control & Semantic Cache Proxy",
        version="0.1.0",
        description=(
            "Open-core MIT proxy providing L1 exact + L2 semantic caching, "
            "sliding-window RPM/TPM rate limiting, and monthly USD budget enforcement."
        ),
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.state_store = store
    app.state.rate_limiter = rate_limiter
    app.state.telemetry_repo = telemetry_repo
    app.state.proxy_service = proxy_service

    app.include_router(admin_router)
    app.include_router(proxy_router)
    return app


app = create_app()
