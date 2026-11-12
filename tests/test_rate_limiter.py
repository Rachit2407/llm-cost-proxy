"""Unit tests for sliding-window RPM/TPM rate limiter and monthly USD budget guard."""

from __future__ import annotations

from llm_cost_proxy.core.rate_limiter import RateLimiter
from llm_cost_proxy.core.redis_store import InMemoryStore
from llm_cost_proxy.models.schemas import TenantPolicy


async def test_rpm_sliding_window_blocks_and_recovers(memory_store: InMemoryStore) -> None:
    limiter = RateLimiter(memory_store)
    policy = TenantPolicy(
        tenant_id="startup-1",
        rpm_limit=2,
        tpm_limit=10_000,
        monthly_budget_usd=10.0,
    )

    d1 = await limiter.check_request(policy, estimated_input_tokens=50, now_ts=1000.0)
    d2 = await limiter.check_request(policy, estimated_input_tokens=50, now_ts=1010.0)
    d3 = await limiter.check_request(policy, estimated_input_tokens=50, now_ts=1020.0)

    assert d1.allowed is True
    assert d1.remaining_rpm == 1
    assert d2.allowed is True
    assert d2.remaining_rpm == 0
    assert d3.allowed is False
    assert d3.reason == "rpm_exceeded"

    # After 60s window expires for t=1000.0, t=1061.0 is allowed again.
    d4 = await limiter.check_request(policy, estimated_input_tokens=50, now_ts=1061.0)
    assert d4.allowed is True


async def test_tpm_limit_blocks_when_tokens_exhausted(memory_store: InMemoryStore) -> None:
    limiter = RateLimiter(memory_store)
    policy = TenantPolicy(
        tenant_id="startup-2",
        rpm_limit=100,
        tpm_limit=500,
        monthly_budget_usd=10.0,
    )

    d1 = await limiter.check_request(policy, estimated_input_tokens=100, now_ts=2000.0)
    assert d1.allowed is True

    await limiter.record_upstream_usage(policy, total_tokens=450, cost_usd=0.001, now_ts=2001.0)

    d2 = await limiter.check_request(policy, estimated_input_tokens=100, now_ts=2002.0)
    assert d2.allowed is False
    assert d2.reason == "tpm_exceeded"
    assert d2.remaining_tpm == 50


async def test_monthly_usd_budget_enforcement(memory_store: InMemoryStore) -> None:
    limiter = RateLimiter(memory_store)
    policy = TenantPolicy(
        tenant_id="startup-3",
        rpm_limit=100,
        tpm_limit=100_000,
        monthly_budget_usd=0.05,
    )

    first = await limiter.check_request(policy, estimated_input_tokens=50, now_ts=3000.0)
    assert first.allowed is True

    new_spend = await limiter.record_upstream_usage(
        policy,
        total_tokens=500,
        cost_usd=0.05,
        now_ts=3001.0,
    )
    assert new_spend >= 0.05

    blocked = await limiter.check_request(policy, estimated_input_tokens=10, now_ts=3002.0)
    assert blocked.allowed is False
    assert blocked.reason == "budget_exceeded"
