"""Sliding-window RPM/TPM rate limiter and monthly USD spend budget enforcer."""

from __future__ import annotations

from datetime import UTC, datetime

from llm_cost_proxy.core.redis_store import StateStore
from llm_cost_proxy.models.schemas import (
    RateLimitDecision,
    TenantBudgetStatus,
    TenantPolicy,
)

_WINDOW_SECONDS = 60
_BUDGET_TTL_SECONDS = 35 * 24 * 3600


class RateLimiter:
    """Enforces per-tenant RPM, TPM, and monthly USD budget caps using StateStore."""

    def __init__(self, store: StateStore) -> None:
        self._store = store

    @staticmethod
    def _rpm_key(tenant_id: str) -> str:
        return f"rl:rpm:{tenant_id}"

    @staticmethod
    def _tpm_key(tenant_id: str) -> str:
        return f"rl:tpm:{tenant_id}"

    @staticmethod
    def _budget_key(tenant_id: str, now: datetime | None = None) -> str:
        current = now or datetime.now(UTC)
        return f"budget:usd:{tenant_id}:{current.strftime('%Y-%m')}"

    async def check_request(
        self,
        policy: TenantPolicy,
        estimated_input_tokens: int,
        now_ts: float | None = None,
    ) -> RateLimitDecision:
        """Check monthly USD budget, TPM headroom, and reserve 1 RPM slot."""
        budget_key = self._budget_key(policy.tenant_id)
        current_spend = await self._store.get_float(budget_key)
        if policy.monthly_budget_usd >= 0.0 and current_spend >= policy.monthly_budget_usd:
            return RateLimitDecision(
                allowed=False,
                reason="budget_exceeded",
                retry_after_seconds=3600,
                remaining_rpm=0,
                remaining_tpm=0,
                current_spend_usd=round(current_spend, 6),
                budget_limit_usd=policy.monthly_budget_usd,
            )

        tpm_key = self._tpm_key(policy.tenant_id)
        current_tpm = await self._store.window_sum(tpm_key, _WINDOW_SECONDS, now_ts=now_ts)
        if current_tpm + estimated_input_tokens > policy.tpm_limit:
            return RateLimitDecision(
                allowed=False,
                reason="tpm_exceeded",
                retry_after_seconds=_WINDOW_SECONDS,
                remaining_rpm=max(0, policy.rpm_limit),
                remaining_tpm=max(0, policy.tpm_limit - current_tpm),
                current_spend_usd=round(current_spend, 6),
                budget_limit_usd=policy.monthly_budget_usd,
            )

        rpm_key = self._rpm_key(policy.tenant_id)
        allowed, used_rpm = await self._store.window_add_if_under(
            rpm_key,
            window_seconds=_WINDOW_SECONDS,
            limit=policy.rpm_limit,
            weight=1,
            now_ts=now_ts,
        )
        if not allowed:
            return RateLimitDecision(
                allowed=False,
                reason="rpm_exceeded",
                retry_after_seconds=_WINDOW_SECONDS,
                remaining_rpm=0,
                remaining_tpm=max(0, policy.tpm_limit - current_tpm),
                current_spend_usd=round(current_spend, 6),
                budget_limit_usd=policy.monthly_budget_usd,
            )

        return RateLimitDecision(
            allowed=True,
            reason="ok",
            retry_after_seconds=0,
            remaining_rpm=max(0, policy.rpm_limit - used_rpm),
            remaining_tpm=max(0, policy.tpm_limit - current_tpm),
            current_spend_usd=round(current_spend, 6),
            budget_limit_usd=policy.monthly_budget_usd,
        )

    async def record_upstream_usage(
        self,
        policy: TenantPolicy,
        total_tokens: int,
        cost_usd: float,
        now_ts: float | None = None,
    ) -> float:
        """Record actual upstream tokens (TPM) and USD spend after a cache miss."""
        if total_tokens > 0:
            await self._store.window_add(
                self._tpm_key(policy.tenant_id),
                window_seconds=_WINDOW_SECONDS,
                weight=total_tokens,
                now_ts=now_ts,
            )
        if cost_usd > 0.0:
            return await self._store.incr_float(
                self._budget_key(policy.tenant_id),
                amount=cost_usd,
                ttl_seconds=_BUDGET_TTL_SECONDS,
            )
        return await self._store.get_float(self._budget_key(policy.tenant_id))

    async def get_budget_status(
        self,
        policy: TenantPolicy,
        now_ts: float | None = None,
    ) -> TenantBudgetStatus:
        """Read current RPM, TPM, and monthly USD spend without consuming quota."""
        used_rpm = await self._store.window_sum(
            self._rpm_key(policy.tenant_id),
            _WINDOW_SECONDS,
            now_ts=now_ts,
        )
        used_tpm = await self._store.window_sum(
            self._tpm_key(policy.tenant_id),
            _WINDOW_SECONDS,
            now_ts=now_ts,
        )
        current_spend = await self._store.get_float(self._budget_key(policy.tenant_id))
        remaining_usd = max(0.0, round(policy.monthly_budget_usd - current_spend, 6))
        return TenantBudgetStatus(
            tenant_id=policy.tenant_id,
            used_rpm=used_rpm,
            rpm_limit=policy.rpm_limit,
            used_tpm=used_tpm,
            tpm_limit=policy.tpm_limit,
            current_spend_usd=round(current_spend, 6),
            monthly_budget_usd=policy.monthly_budget_usd,
            remaining_budget_usd=remaining_usd,
        )
