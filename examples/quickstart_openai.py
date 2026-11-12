"""Drop-in quickstart showing how to route OpenAI-compatible requests via llm-cost-proxy."""

from __future__ import annotations

import os

import httpx


def main() -> None:
    proxy_url = os.getenv("LLM_PROXY_URL", "http://localhost:8080")
    proxy_key = os.getenv("LLM_PROXY_KEY", "pk-startup-dev")
    upstream_key = os.getenv("OPENAI_API_KEY", "")

    headers = {"Authorization": f"Bearer {proxy_key}"}
    if upstream_key:
        headers["X-Upstream-API-Key"] = upstream_key

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": "Explain Redis pipelining in two concise sentences."}
        ],
    }

    with httpx.Client(base_url=proxy_url, timeout=30.0) as client:
        for attempt in (1, 2):
            resp = client.post("/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            print(
                f"Call #{attempt}: cache={resp.headers.get('X-Proxy-Cache')} "
                f"saved_usd={resp.headers.get('X-Proxy-Cost-Saved-USD')}"
            )

        stats = client.get("/v1/stats").json()
        print("Proxy stats:", stats)


if __name__ == "__main__":
    main()
