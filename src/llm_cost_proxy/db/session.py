"""Async SQLAlchemy engine, schema initialization, and telemetry repository."""

from __future__ import annotations

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from llm_cost_proxy.models.schemas import CacheStatus, ProviderName, TelemetryStats
from llm_cost_proxy.models.telemetry import Base, UsageEvent


def create_engine_and_session_factory(
    database_url: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Create an async SQLAlchemy engine and session factory."""
    engine = create_async_engine(database_url, echo=False, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


async def init_db(engine: AsyncEngine) -> None:
    """Create database tables if they do not exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


class TelemetryRepository:
    """Persists and aggregates request-level LLM usage and cache savings."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record_event(
        self,
        *,
        request_id: str,
        tenant_id: str,
        provider: ProviderName,
        model: str,
        cache_status: CacheStatus,
        similarity: float | None,
        prompt_tokens: int,
        completion_tokens: int,
        upstream_cost_usd: float,
        cost_saved_usd: float,
        latency_ms: float,
    ) -> None:
        """Persist a single telemetry event."""
        async with self._session_factory() as session:
            event = UsageEvent(
                request_id=request_id,
                tenant_id=tenant_id,
                provider=provider,
                model=model,
                cache_status=cache_status,
                similarity=similarity,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                upstream_cost_usd=upstream_cost_usd,
                cost_saved_usd=cost_saved_usd,
                latency_ms=latency_ms,
            )
            session.add(event)
            await session.commit()

    async def get_stats(self, tenant_id: str | None = None) -> TelemetryStats:
        """Return aggregated request counts, cache hit rate, spend, and savings."""
        async with self._session_factory() as session:
            stmt = select(
                func.count(UsageEvent.id),
                func.coalesce(
                    func.sum(case((UsageEvent.cache_status == "EXACT_HIT", 1), else_=0)),
                    0,
                ),
                func.coalesce(
                    func.sum(case((UsageEvent.cache_status == "SEMANTIC_HIT", 1), else_=0)),
                    0,
                ),
                func.coalesce(
                    func.sum(case((UsageEvent.cache_status == "MISS", 1), else_=0)),
                    0,
                ),
                func.coalesce(
                    func.sum(case((UsageEvent.cache_status == "BLOCKED", 1), else_=0)),
                    0,
                ),
                func.coalesce(func.sum(UsageEvent.prompt_tokens), 0),
                func.coalesce(func.sum(UsageEvent.completion_tokens), 0),
                func.coalesce(func.sum(UsageEvent.upstream_cost_usd), 0.0),
                func.coalesce(func.sum(UsageEvent.cost_saved_usd), 0.0),
            )
            if tenant_id is not None:
                stmt = stmt.where(UsageEvent.tenant_id == tenant_id)

            row = (await session.execute(stmt)).one()
            total = int(row[0] or 0)
            exact_hits = int(row[1] or 0)
            semantic_hits = int(row[2] or 0)
            cache_misses = int(row[3] or 0)
            blocked = int(row[4] or 0)
            cacheable_total = exact_hits + semantic_hits + cache_misses
            hit_rate = (
                round((exact_hits + semantic_hits) / cacheable_total, 4)
                if cacheable_total > 0
                else 0.0
            )
            return TelemetryStats(
                tenant_id=tenant_id,
                total_requests=total,
                exact_hits=exact_hits,
                semantic_hits=semantic_hits,
                cache_misses=cache_misses,
                blocked_requests=blocked,
                hit_rate=hit_rate,
                total_prompt_tokens=int(row[5] or 0),
                total_completion_tokens=int(row[6] or 0),
                total_spend_usd=round(float(row[7] or 0.0), 6),
                total_saved_usd=round(float(row[8] or 0.0), 6),
            )
