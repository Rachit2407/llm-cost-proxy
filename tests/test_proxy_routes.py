"""End-to-end FastAPI route tests for caching, rate limiting, auth, and telemetry."""

from __future__ import annotations

import json
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from llm_cost_proxy.config import Settings
from llm_cost_proxy.main import create_app
from tests.conftest import FakeLLMProvider


async def test_openai_exact_and_semantic_cache_flow(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'routes.db').as_posix()}",
        redis_url="memory://",
        semantic_similarity_threshold=0.75,
        allow_anonymous_dev=True,
    )
    openai_mock = FakeLLMProvider(prompt_tokens=1000, completion_tokens=500)
    anthropic_mock = FakeLLMProvider(prompt_tokens=200, completion_tokens=100)

    app = create_app(
        settings=settings,
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
    )

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload_1 = {
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": "How do I configure Redis persistence with AOF snapshots?",
                    }
                ],
            }
            r1 = await client.post("/v1/chat/completions", json=payload_1)
            assert r1.status_code == 200
            assert r1.headers["X-Proxy-Cache"] == "MISS"
            assert len(openai_mock.calls) == 1

            # Identical request -> L1 Exact Hit (no second upstream call)
            r2 = await client.post("/v1/chat/completions", json=payload_1)
            assert r2.status_code == 200
            assert r2.headers["X-Proxy-Cache"] == "EXACT_HIT"
            assert float(r2.headers["X-Proxy-Cost-Saved-USD"]) > 0.0
            assert len(openai_mock.calls) == 1

            # Paraphrased request -> L2 Semantic Hit (still no second upstream call)
            payload_para = {
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": "How to configure Redis persistence using AOF snapshots?",
                    }
                ],
            }
            r3 = await client.post("/v1/chat/completions", json=payload_para)
            assert r3.status_code == 200
            assert r3.headers["X-Proxy-Cache"] == "SEMANTIC_HIT"
            assert float(r3.headers["X-Proxy-Similarity"]) >= 0.75
            assert len(openai_mock.calls) == 1

            # Cache bypass header forces upstream call
            r4 = await client.post(
                "/v1/chat/completions",
                json=payload_1,
                headers={"X-Cache-Control": "no-cache"},
            )
            assert r4.status_code == 200
            assert r4.headers["X-Proxy-Cache"] == "BYPASS"
            assert len(openai_mock.calls) == 2

            stats_resp = await client.get("/v1/stats")
            assert stats_resp.status_code == 200
            stats = stats_resp.json()
            assert stats["total_requests"] == 4
            assert stats["exact_hits"] == 1
            assert stats["semantic_hits"] == 1
            assert stats["cache_misses"] == 1
            assert stats["total_saved_usd"] > 0.0


async def test_anthropic_route_and_streaming_cache_replay(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'anthro.db').as_posix()}",
        redis_url="memory://",
        allow_anonymous_dev=True,
    )
    openai_mock = FakeLLMProvider()
    anthropic_mock = FakeLLMProvider(reply_text="Anthropic hello")

    app = create_app(
        settings=settings,
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
    )

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 256,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": "Say hi in 2 words."}],
            }
            r1 = await client.post("/v1/messages", json=payload)
            assert r1.status_code == 200
            assert r1.headers["X-Proxy-Cache"] == "MISS"

            # Streamed replay from cache
            stream_payload = {**payload, "stream": True}
            r2 = await client.post("/v1/messages", json=stream_payload)
            assert r2.status_code == 200
            assert r2.headers["X-Proxy-Cache"] == "EXACT_HIT"
            assert "Anthropic hello" in r2.text
            assert len(anthropic_mock.calls) == 1


async def test_auth_and_rate_limit_and_budget_http_codes(tmp_path: Path) -> None:
    keys = {
        "pk-limited": {
            "tenant_id": "acme",
            "rpm_limit": 2,
            "tpm_limit": 100_000,
            "monthly_budget_usd": 0.0001,
        }
    }
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'auth.db').as_posix()}",
        redis_url="memory://",
        allow_anonymous_dev=False,
        proxy_api_keys_json=json.dumps(keys),
    )
    # High token usage on first miss will exceed $0.0001 budget for subsequent calls.
    openai_mock = FakeLLMProvider(prompt_tokens=1000, completion_tokens=1000)

    app = create_app(
        settings=settings,
        providers={"openai": openai_mock, "anthropic": FakeLLMProvider()},
    )

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            }

            # Missing key -> 401
            unauth = await client.post("/v1/chat/completions", json=payload)
            assert unauth.status_code == 401

            # First valid call succeeds and spends > $0.0001
            ok = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer pk-limited"},
            )
            assert ok.status_code == 200

            # Second call is blocked by monthly budget -> 402
            budget_blocked = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer pk-limited"},
            )
            assert budget_blocked.status_code == 402
            assert budget_blocked.headers["X-Proxy-Cache"] == "BLOCKED"

            health_resp = await client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "ok"


async def test_admin_metrics_models_budget_and_cache_purge(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'admin.db').as_posix()}",
        redis_url="memory://",
        allow_anonymous_dev=True,
        default_rpm_limit=10,
        default_monthly_budget_usd=25.0,
    )
    openai_mock = FakeLLMProvider(prompt_tokens=100, completion_tokens=50)
    app = create_app(
        settings=settings,
        providers={"openai": openai_mock, "anthropic": FakeLLMProvider()},
    )

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "messages": [{"role": "user", "content": "Test admin endpoints"}],
            }
            miss = await client.post("/v1/chat/completions", json=payload)
            assert miss.headers["X-Proxy-Cache"] == "MISS"

            hit = await client.post("/v1/chat/completions", json=payload)
            assert hit.headers["X-Proxy-Cache"] == "EXACT_HIT"

            models_resp = await client.get("/v1/models")
            assert models_resp.status_code == 200
            model_ids = {item["id"] for item in models_resp.json()["data"]}
            assert "gpt-4o-mini" in model_ids

            dash_resp = await client.get("/dashboard")
            assert dash_resp.status_code == 200
            assert "LLM Semantic Cache & Proxy" in dash_resp.text
            root_resp = await client.get("/")
            assert root_resp.status_code == 200

            budget_resp = await client.get("/v1/budget")
            assert budget_resp.status_code == 200
            budget = budget_resp.json()
            assert budget["tenant_id"] == "dev-default"
            assert budget["used_rpm"] == 2
            assert budget["current_spend_usd"] > 0.0

            metrics_resp = await client.get("/metrics")
            assert metrics_resp.status_code == 200
            assert "llm_proxy_requests_total" in metrics_resp.text
            assert 'llm_proxy_requests_total{status="exact_hit"} 1' in metrics_resp.text

            purge_resp = await client.delete("/v1/cache")
            assert purge_resp.status_code == 200
            assert purge_resp.json()["deleted_exact_keys"] == 1
            assert purge_resp.json()["deleted_semantic_buckets"] == 1

            # After purge, the same prompt misses and calls upstream again.
            after_purge = await client.post("/v1/chat/completions", json=payload)
            assert after_purge.headers["X-Proxy-Cache"] == "MISS"
            assert len(openai_mock.calls) == 2


async def test_budget_auto_downgrade_routing(tmp_path: Path) -> None:
    keys = {
        "pk-downgrade": {
            "tenant_id": "acme-dg",
            "rpm_limit": 20,
            "tpm_limit": 100_000,
            "monthly_budget_usd": 0.01,
        }
    }
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'downgrade.db').as_posix()}",
        redis_url="memory://",
        allow_anonymous_dev=False,
        proxy_api_keys_json=json.dumps(keys),
        enable_budget_downgrade=True,
        budget_downgrade_ratio=0.50,
    )
    # 1000 input + 500 output tokens on gpt-4o costs $0.0025 + $0.0050 = $0.0075 (75% of $0.01)
    openai_mock = FakeLLMProvider(prompt_tokens=1000, completion_tokens=500)
    app = create_app(
        settings=settings,
        providers={"openai": openai_mock, "anthropic": FakeLLMProvider()},
    )

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer pk-downgrade"}
            r1 = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "First"}]},
            )
            assert r1.status_code == 200
            assert "X-Proxy-Model-Downgraded" not in r1.headers
            assert openai_mock.calls[0].model == "gpt-4o"

            # Spend is 75% (>= 50% downgrade ratio, < 100% hard cap) -> gpt-4o-mini
            r2 = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Second"}]},
            )
            assert r2.status_code == 200
            assert r2.headers["X-Proxy-Model-Downgraded"] == "gpt-4o->gpt-4o-mini"
            assert openai_mock.calls[1].model == "gpt-4o-mini"
