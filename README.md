# LLM Cost-Control & Semantic Cache Proxy

An open-core (MIT) B2B LLM gateway built with **FastAPI**, **Redis**, and **SQLAlchemy (SQLite / PostgreSQL)** that acts as a drop-in proxy for OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) SDKs to cut API spend via:

- **L1 Exact Cache:** Deterministic SHA-256 hash lookup on canonicalized `(provider, model, messages, temperature, max_tokens)`.
- **L2 Semantic Cache:** Vector similarity matching (`cosine >= threshold`) scoped per tenant, provider, and model—with a zero-dependency local feature-hashing embedder (`local_hash`) or optional OpenAI `text-embedding-3-small`.
- **Sliding-Window Rate Limiter & Budget Enforcer:** Atomic Redis Lua / sorted-set RPM & TPM guards plus monthly USD spend caps evaluated **before** embedding or upstream calls.
- **Usage & Savings Telemetry:** Tracks `EXACT_HIT`, `SEMANTIC_HIT`, `MISS`, `BYPASS`, and `BLOCKED` requests along with tokens and USD saved via `/v1/stats`.

## Architecture & Request Lifecycle

```text
Client App (OpenAI / Anthropic SDK)
   │
   ▼
[ FastAPI Gateway ] ──► Resolve virtual key -> TenantPolicy
   │
   ├──► [1. Rate Limit & Monthly Budget Guard (Redis)]
   │       └─ Exceeded? -> 429 (RPM/TPM) or 402 (USD Budget)
   │
   ├──► [2. L1 Exact Cache (SHA-256)]
   │       └─ Hit? -> 200 + X-Proxy-Cache: EXACT_HIT (0 upstream cost)
   │
   ├──► [3. L2 Semantic Cache (Embedding + Cosine >= threshold)]
   │       └─ Hit? -> 200 + X-Proxy-Cache: SEMANTIC_HIT (0 upstream cost)
   │
   └──► [4. Cache Miss -> Upstream OpenAI / Anthropic]
           └─ Record TPM/USD spend, populate L1 + L2 cache, log telemetry
```

## Quickstart (Docker Compose)

1. Copy `.env.example` to `.env` and supply your upstream provider keys (never commit real keys):

   ```bash
   cp .env.example .env
   ```

2. Start Redis Stack, PostgreSQL, and the proxy:

   ```bash
   docker compose up --build
   ```

3. Point your OpenAI SDK at `http://localhost:8080/v1`:

   ```python
   from openai import OpenAI

   client = OpenAI(
       base_url="http://localhost:8080/v1",
       api_key="pk-startup-dev",  # virtual key defined in PROXY_API_KEYS_JSON
   )
   resp = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": "Explain Redis pipelining in 2 sentences."}],
   )
   ```

4. Inspect cost savings, live budget, Prometheus metrics, or open the visual Web Dashboard:

   - **Visual Dashboard:** Open [http://localhost:8080/dashboard](http://localhost:8080/dashboard) in your browser.
   - **Admin API:**
     ```bash
     curl http://localhost:8080/v1/stats
     curl http://localhost:8080/v1/budget
     curl http://localhost:8080/v1/models
     curl http://localhost:8080/metrics
     curl -X DELETE http://localhost:8080/v1/cache
     ```

   Configure `ALERT_WEBHOOK_URL` and `ALERT_THRESHOLD_RATIO` (default `0.80`) to receive a deduplicated Slack-compatible webhook alert when a tenant crosses its monthly USD budget threshold.

## Local Development & Tests

```bash
pip install -e ".[dev]"
pytest
mypy src tests
```

## Open-Core Roadmap

- **MIT Core (this repo):** Drop-in OpenAI/Anthropic proxy, L1 exact + L2 semantic cache, RPM/TPM/USD budget enforcement, SQLite/Postgres telemetry, Docker Compose.
- **Enterprise Add-ons (self-hosted commercial tier):** SAML/OIDC SSO, Slack/PagerDuty budget threshold alerts, multi-tenant web dashboard, and RBAC policy management.
