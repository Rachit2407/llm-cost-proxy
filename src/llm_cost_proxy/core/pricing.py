"""Model pricing registry, token estimation, and USD cost calculation."""

from __future__ import annotations

from dataclasses import dataclass

from llm_cost_proxy.models.schemas import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    """USD cost per 1,000,000 tokens."""

    input_per_1m_usd: float
    output_per_1m_usd: float


# Reference pricing (USD per 1M tokens). Unlisted models fall back conservatively.
MODEL_PRICING_TABLE: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(input_per_1m_usd=2.50, output_per_1m_usd=10.00),
    "gpt-4o-mini": ModelPricing(input_per_1m_usd=0.15, output_per_1m_usd=0.60),
    "gpt-4.1": ModelPricing(input_per_1m_usd=2.00, output_per_1m_usd=8.00),
    "gpt-4.1-mini": ModelPricing(input_per_1m_usd=0.40, output_per_1m_usd=1.60),
    "gpt-4.1-nano": ModelPricing(input_per_1m_usd=0.10, output_per_1m_usd=0.40),
    "o3-mini": ModelPricing(input_per_1m_usd=1.10, output_per_1m_usd=4.40),
    # Anthropic
    "claude-3-5-sonnet-20241022": ModelPricing(input_per_1m_usd=3.00, output_per_1m_usd=15.00),
    "claude-3-5-haiku-20241022": ModelPricing(input_per_1m_usd=0.80, output_per_1m_usd=4.00),
    "claude-3-7-sonnet-20250219": ModelPricing(input_per_1m_usd=3.00, output_per_1m_usd=15.00),
    "claude-sonnet-4-20250514": ModelPricing(input_per_1m_usd=3.00, output_per_1m_usd=15.00),
}

DEFAULT_MODEL_PRICING = ModelPricing(input_per_1m_usd=2.50, output_per_1m_usd=10.00)

MODEL_DOWNGRADE_MAP: dict[str, str] = {
    "gpt-4o": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1-mini",
    "claude-3-5-sonnet-20241022": "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219": "claude-3-5-haiku-20241022",
    "claude-sonnet-4-20250514": "claude-3-5-haiku-20241022",
}


def get_downgrade_model(model: str) -> str | None:
    """Return a cheaper fallback model when approaching monthly USD budget."""
    return MODEL_DOWNGRADE_MAP.get(model.strip().lower())


def get_model_pricing(model: str) -> ModelPricing:
    """Resolve exact or prefix-matched model pricing."""
    normalized = model.strip().lower()
    if normalized in MODEL_PRICING_TABLE:
        return MODEL_PRICING_TABLE[normalized]
    for known_model, pricing in MODEL_PRICING_TABLE.items():
        if normalized.startswith(known_model):
            return pricing
    return DEFAULT_MODEL_PRICING


def calculate_cost_usd(model: str, usage: TokenUsage) -> float:
    """Calculate USD cost for a request given its model and token counts."""
    pricing = get_model_pricing(model)
    input_cost = (usage.prompt_tokens / 1_000_000.0) * pricing.input_per_1m_usd
    output_cost = (usage.completion_tokens / 1_000_000.0) * pricing.output_per_1m_usd
    return round(input_cost + output_cost, 8)


def estimate_input_tokens(text: str) -> int:
    """Fast heuristic token estimator (~4 chars per token) for pre-flight TPM checks."""
    if not text:
        return 1
    return max(1, (len(text) + 3) // 4)
