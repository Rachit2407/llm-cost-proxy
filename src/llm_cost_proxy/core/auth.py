"""Virtual proxy API key authentication and tenant policy resolution."""

from __future__ import annotations

from dataclasses import dataclass

from llm_cost_proxy.config import Settings
from llm_cost_proxy.models.schemas import ProviderName, TenantPolicy


class AuthenticationError(Exception):
    """Raised when a proxy request lacks a valid virtual API key."""


@dataclass(frozen=True)
class AuthContext:
    """Resolved tenant policy and upstream API key for the current request."""

    policy: TenantPolicy
    upstream_api_key: str


def _extract_proxy_token(
    authorization: str | None,
    x_api_key: str | None,
) -> str | None:
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    return None


def _coerce_int(value: object, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return fallback
    return fallback


def _coerce_float(value: object, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return fallback
    return fallback


def resolve_auth_context(
    *,
    settings: Settings,
    provider: ProviderName,
    authorization: str | None,
    x_api_key: str | None,
    x_upstream_api_key: str | None,
) -> AuthContext:
    """Resolve virtual proxy key to a TenantPolicy and select upstream API key."""
    token = _extract_proxy_token(authorization, x_api_key)
    key_map = settings.parsed_api_keys()

    if token and token in key_map:
        raw_policy = key_map[token]
        policy = TenantPolicy(
            tenant_id=str(raw_policy.get("tenant_id", "default")),
            rpm_limit=_coerce_int(raw_policy.get("rpm_limit"), settings.default_rpm_limit),
            tpm_limit=_coerce_int(raw_policy.get("tpm_limit"), settings.default_tpm_limit),
            monthly_budget_usd=_coerce_float(
                raw_policy.get("monthly_budget_usd"),
                settings.default_monthly_budget_usd,
            ),
            semantic_threshold=_coerce_float(
                raw_policy.get("semantic_threshold"),
                settings.semantic_similarity_threshold,
            ),
            cache_ttl_seconds=_coerce_int(
                raw_policy.get("cache_ttl_seconds"),
                settings.cache_ttl_seconds,
            ),
        )
    elif settings.allow_anonymous_dev:
        policy = TenantPolicy(
            tenant_id="dev-default",
            rpm_limit=settings.default_rpm_limit,
            tpm_limit=settings.default_tpm_limit,
            monthly_budget_usd=settings.default_monthly_budget_usd,
            semantic_threshold=settings.semantic_similarity_threshold,
            cache_ttl_seconds=settings.cache_ttl_seconds,
        )
    else:
        raise AuthenticationError("Invalid or missing proxy API key.")

    if x_upstream_api_key and x_upstream_api_key.strip():
        upstream_key = x_upstream_api_key.strip()
    elif provider == "openai":
        upstream_key = settings.openai_api_key
    else:
        upstream_key = settings.anthropic_api_key

    return AuthContext(policy=policy, upstream_api_key=upstream_key)
