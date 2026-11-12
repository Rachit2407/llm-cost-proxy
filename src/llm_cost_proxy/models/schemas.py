"""Strictly typed request, response, cache, and policy schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["openai", "anthropic"]
CacheStatus = Literal["EXACT_HIT", "SEMANTIC_HIT", "MISS", "BYPASS", "BLOCKED"]


class ChatMessage(BaseModel):
    """Canonical message representation compatible with OpenAI and Anthropic."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, object]]


class OpenAIChatRequest(BaseModel):
    """Subset of OpenAI /v1/chat/completions request schema with pass-through support."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False


class AnthropicMessageRequest(BaseModel):
    """Subset of Anthropic /v1/messages request schema with pass-through support."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    system: str | list[dict[str, object]] | None = None
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False


class CanonicalRequest(BaseModel):
    """Normalized internal representation used across cache, rate limiter, and providers."""

    provider: ProviderName
    model: str
    messages: list[dict[str, object]]
    system_prompt: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    stream: bool = False
    raw_payload: dict[str, object]

    def prompt_text(self) -> str:
        """Flatten system and message text for token estimation and semantic embedding."""
        parts: list[str] = []
        if self.system_prompt:
            parts.append(f"system: {self.system_prompt}")
        for msg in self.messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                chunks = [
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and "text" in item
                ]
                text = " ".join(chunks)
            else:
                text = str(content)
            parts.append(f"{role}: {text}")
        return "\n".join(parts)


class TokenUsage(BaseModel):
    """Normalized token usage counts."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ProviderResponse(BaseModel):
    """Normalized upstream LLM response payload."""

    status_code: int
    body: dict[str, object]
    usage: TokenUsage
    raw_sse_chunks: list[str] | None = None


class CachedEntry(BaseModel):
    """Serialized response stored in L1 Exact and L2 Semantic caches."""

    provider: ProviderName
    model: str
    response_body: dict[str, object]
    usage: TokenUsage
    estimated_cost_usd: float = Field(ge=0.0)


class CacheLookupResult(BaseModel):
    """Result returned by ExactCache or SemanticCache lookups."""

    hit: bool
    status: Literal["EXACT_HIT", "SEMANTIC_HIT", "MISS"]
    entry: CachedEntry | None = None
    similarity: float | None = None


class TenantPolicy(BaseModel):
    """Per-tenant rate limit, budget, and cache policy."""

    tenant_id: str = "default"
    rpm_limit: int = Field(default=60, ge=1)
    tpm_limit: int = Field(default=100_000, ge=1)
    monthly_budget_usd: float = Field(default=50.0, ge=0.0)
    semantic_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=86400, ge=1)


class RateLimitDecision(BaseModel):
    """Outcome of pre-flight rate limit and budget enforcement."""

    allowed: bool
    reason: Literal["ok", "rpm_exceeded", "tpm_exceeded", "budget_exceeded"] = "ok"
    retry_after_seconds: int = 0
    remaining_rpm: int = 0
    remaining_tpm: int = 0
    current_spend_usd: float = 0.0
    budget_limit_usd: float = 0.0


class ProxyExecutionResult(BaseModel):
    """Completed proxy pipeline output consumed by API routes."""

    status_code: int
    body: dict[str, object]
    cache_status: CacheStatus
    similarity: float | None = None
    upstream_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    stream_sse_lines: list[str] | None = None
    downgraded_from: str | None = None
    model_used: str | None = None


class TelemetryStats(BaseModel):
    """Aggregated telemetry and savings metrics."""

    tenant_id: str | None = None
    total_requests: int
    exact_hits: int
    semantic_hits: int
    cache_misses: int
    blocked_requests: int
    hit_rate: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_spend_usd: float
    total_saved_usd: float


class TenantBudgetStatus(BaseModel):
    """Live per-tenant spend and sliding-window rate-limit usage."""

    tenant_id: str
    used_rpm: int
    rpm_limit: int
    used_tpm: int
    tpm_limit: int
    current_spend_usd: float
    monthly_budget_usd: float
    remaining_budget_usd: float


class ModelCatalogEntry(BaseModel):
    """OpenAI-compatible /v1/models entry enriched with proxy pricing metadata."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str
    input_per_1m_usd: float
    output_per_1m_usd: float


class ModelCatalogResponse(BaseModel):
    """Response schema for GET /v1/models."""

    object: Literal["list"] = "list"
    data: list[ModelCatalogEntry]
