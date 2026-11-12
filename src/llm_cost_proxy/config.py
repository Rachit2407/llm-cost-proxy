"""Environment-driven application configuration with zero hardcoded secrets."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    proxy_host: str = "0.0.0.0"
    proxy_port: int = 8080
    log_level: str = "INFO"

    # Storage backends
    database_url: str = "sqlite+aiosqlite:///./llm_cost_proxy.db"
    redis_url: str = "memory://"

    # Cache parameters
    cache_ttl_seconds: int = Field(default=86400, ge=1)
    semantic_similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    max_cache_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    semantic_max_candidates_per_scope: int = Field(default=500, ge=1)
    embedding_provider: Literal["local_hash", "openai"] = "local_hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = Field(default=256, ge=32, le=4096)

    # Default rate limits & budgets
    default_rpm_limit: int = Field(default=60, ge=1)
    default_tpm_limit: int = Field(default=100_000, ge=1)
    default_monthly_budget_usd: float = Field(default=50.0, ge=0.0)
    enable_budget_downgrade: bool = True
    budget_downgrade_ratio: float = Field(default=0.90, ge=0.0, le=1.0)
    alert_webhook_url: str = ""
    alert_threshold_ratio: float = Field(default=0.80, ge=0.0, le=1.0)

    # Upstream providers (keys must come from environment or request headers)
    openai_base_url: str = "https://api.openai.com"
    openai_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_key: str = ""
    upstream_timeout_seconds: float = Field(default=60.0, gt=0.0)

    # Virtual key policies JSON string: {"pk-xxx": {"tenant_id": "acme", ...}}
    proxy_api_keys_json: str = "{}"
    allow_anonymous_dev: bool = True

    @field_validator("proxy_api_keys_json")
    @classmethod
    def _validate_api_keys_json(cls, value: str) -> str:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("PROXY_API_KEYS_JSON must decode to a JSON object")
        return value

    def parsed_api_keys(self) -> dict[str, dict[str, object]]:
        """Parse the virtual API key policy mapping from JSON."""
        raw: object = json.loads(self.proxy_api_keys_json)
        if not isinstance(raw, dict):
            return {}
        return {
            str(k): dict(v)
            for k, v in raw.items()
            if isinstance(v, dict)
        }
