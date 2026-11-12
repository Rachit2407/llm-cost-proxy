"""Base protocol and errors for upstream LLM providers."""

from __future__ import annotations

from typing import Protocol

from llm_cost_proxy.models.schemas import CanonicalRequest, ProviderName, ProviderResponse


class UpstreamError(Exception):
    """Raised when an upstream LLM provider call fails or lacks credentials."""

    def __init__(
        self,
        *,
        provider: ProviderName,
        status_code: int,
        message: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.message = message
        self.payload = payload


class LLMProvider(Protocol):
    """Protocol implemented by OpenAI, Anthropic, and test mock providers."""

    async def complete(
        self,
        request: CanonicalRequest,
        api_key: str,
    ) -> ProviderResponse:
        """Forward a canonical request to the upstream LLM API."""
        ...
