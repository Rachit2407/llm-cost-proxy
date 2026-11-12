"""Budget threshold alerting seam (OSS webhook/logger + Enterprise extension hook)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Protocol

import httpx

from llm_cost_proxy.config import Settings
from llm_cost_proxy.core.redis_store import StateStore
from llm_cost_proxy.models.schemas import TenantPolicy

logger = logging.getLogger("llm_cost_proxy.alerts")

_ALERT_DEDUP_TTL_SECONDS = 35 * 24 * 3600


class BudgetAlertNotifier(Protocol):
    """Extension hook invoked when tenant spend crosses a configured budget threshold."""

    async def check_and_notify(
        self,
        *,
        policy: TenantPolicy,
        current_spend_usd: float,
    ) -> bool:
        """Emit an alert if threshold crossed and not already sent this month."""
        ...


class LoggingAlertNotifier:
    """Default MIT-core notifier that logs deduplicated threshold warnings."""

    def __init__(self, store: StateStore, threshold_ratio: float = 0.80) -> None:
        self._store = store
        self._threshold_ratio = threshold_ratio

    async def check_and_notify(
        self,
        *,
        policy: TenantPolicy,
        current_spend_usd: float,
    ) -> bool:
        if policy.monthly_budget_usd <= 0.0:
            return False
        ratio = current_spend_usd / policy.monthly_budget_usd
        if ratio < self._threshold_ratio:
            return False

        month_key = datetime.now(UTC).strftime("%Y-%m")
        threshold_pct = int(round(self._threshold_ratio * 100))
        dedup_key = f"alert:sent:{policy.tenant_id}:{month_key}:{threshold_pct}"
        if await self._store.get(dedup_key) is not None:
            return False

        await self._store.set(dedup_key, "1", ttl_seconds=_ALERT_DEDUP_TTL_SECONDS)
        logger.warning(
            "Tenant %s reached %.1f%% of monthly budget ($%.4f / $%.2f)",
            policy.tenant_id,
            ratio * 100.0,
            current_spend_usd,
            policy.monthly_budget_usd,
        )
        return True


class WebhookAlertNotifier:
    """Posts Slack-compatible JSON budget alerts to ALERT_WEBHOOK_URL (once per month)."""

    def __init__(
        self,
        *,
        store: StateStore,
        webhook_url: str,
        threshold_ratio: float = 0.80,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._store = store
        self._webhook_url = webhook_url
        self._threshold_ratio = threshold_ratio
        self._timeout = timeout_seconds
        self._transport = transport

    async def check_and_notify(
        self,
        *,
        policy: TenantPolicy,
        current_spend_usd: float,
    ) -> bool:
        if policy.monthly_budget_usd <= 0.0:
            return False
        ratio = current_spend_usd / policy.monthly_budget_usd
        if ratio < self._threshold_ratio:
            return False

        month_key = datetime.now(UTC).strftime("%Y-%m")
        threshold_pct = int(round(self._threshold_ratio * 100))
        dedup_key = f"alert:sent:{policy.tenant_id}:{month_key}:{threshold_pct}"
        if await self._store.get(dedup_key) is not None:
            return False

        payload: dict[str, object] = {
            "text": (
                f"[llm-cost-proxy] Tenant `{policy.tenant_id}` reached "
                f"{ratio * 100:.1f}% of monthly USD budget "
                f"(${current_spend_usd:.4f} / ${policy.monthly_budget_usd:.2f})."
            ),
            "tenant_id": policy.tenant_id,
            "current_spend_usd": round(current_spend_usd, 6),
            "monthly_budget_usd": policy.monthly_budget_usd,
            "threshold_ratio": self._threshold_ratio,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                resp = await client.post(self._webhook_url, json=payload)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Failed sending budget alert webhook: %s", exc)
            return False

        await self._store.set(dedup_key, "1", ttl_seconds=_ALERT_DEDUP_TTL_SECONDS)
        return True


def build_alert_notifier(settings: Settings, store: StateStore) -> BudgetAlertNotifier:
    """Instantiate webhook or logging budget alert notifier from Settings."""
    if settings.alert_webhook_url.strip():
        return WebhookAlertNotifier(
            store=store,
            webhook_url=settings.alert_webhook_url.strip(),
            threshold_ratio=settings.alert_threshold_ratio,
        )
    return LoggingAlertNotifier(
        store=store,
        threshold_ratio=settings.alert_threshold_ratio,
    )
