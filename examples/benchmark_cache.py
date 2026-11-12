"""Self-contained latency & cost-savings benchmark against a running llm-cost-proxy."""

from __future__ import annotations

import os
import statistics
import time

import httpx


def main() -> None:
    proxy_url = os.getenv("LLM_PROXY_URL", "http://localhost:8080")
    proxy_key = os.getenv("LLM_PROXY_KEY", "pk-startup-dev")
    upstream_key = os.getenv("OPENAI_API_KEY", "")

    headers = {"Authorization": f"Bearer {proxy_key}"}
    if upstream_key:
        headers["X-Upstream-API-Key"] = upstream_key

    prompts = [
        ("seed", "How do I configure Redis persistence with AOF snapshots?"),
        ("exact", "How do I configure Redis persistence with AOF snapshots?"),
        ("semantic", "How to configure Redis persistence using AOF snapshots?"),
        ("exact", "How do I configure Redis persistence with AOF snapshots?"),
        ("semantic", "How do I set up Redis persistence with AOF snapshots?"),
    ]

    latencies_by_status: dict[str, list[float]] = {}
    with httpx.Client(base_url=proxy_url, timeout=30.0) as client:
        for label, content in prompts:
            t0 = time.perf_counter()
            resp = client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": content}],
                },
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resp.raise_for_status()
            status = resp.headers.get("X-Proxy-Cache", "UNKNOWN")
            latencies_by_status.setdefault(status, []).append(elapsed_ms)
            print(f"[{label:8s}] status={status:12s} latency={elapsed_ms:6.2f}ms")

        stats = client.get("/v1/stats").json()

    print("\n--- Summary ---")
    for status, vals in latencies_by_status.items():
        print(f"{status:12s}: count={len(vals)} mean={statistics.mean(vals):.2f}ms")
    print(f"Hit rate: {stats['hit_rate'] * 100:.1f}% | Saved USD: ${stats['total_saved_usd']:.6f}")


if __name__ == "__main__":
    main()
