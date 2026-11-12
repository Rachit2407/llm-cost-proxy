"""Modern, zero-dependency HTML dashboard for live proxy telemetry and budget inspection."""

from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LLM Cost-Control Proxy | Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #3b82f6;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font-sans: 'Inter', -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      line-height: 1.5;
      padding: 2rem;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 1.5rem;
    }
    .brand { display: flex; align-items: center; gap: 0.75rem; }
    .badge {
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-core { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .status-pill {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.85rem;
      padding: 0.35rem 0.75rem;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 9999px;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 0.75rem;
      padding: 1.25rem;
      transition: border-color 0.2s;
    }
    .card:hover { border-color: #374151; }
    .card-title { font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .card-value { font-size: 1.75rem; font-weight: 700; font-family: var(--font-mono); }
    .card-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }
    .highlight-success { color: var(--success); }
    .highlight-accent { color: #60a5fa; }
    .section-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
    .progress-bar-bg { background: #1f2937; height: 8px; border-radius: 9999px; overflow: hidden; margin-top: 0.75rem; }
    .progress-bar-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
    table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
    th { text-align: left; padding: 0.75rem 1rem; color: var(--text-muted); border-bottom: 1px solid var(--card-border); font-weight: 600; }
    td { padding: 0.75rem 1rem; border-bottom: 1px solid var(--card-border); font-family: var(--font-mono); font-size: 0.8rem; }
    tr:last-child td { border-bottom: none; }
    .btn {
      background: var(--accent);
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      font-size: 0.85rem;
      font-weight: 600;
      border-radius: 0.5rem;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .btn:hover { opacity: 0.9; }
    .btn-danger { background: #dc2626; }
    .controls { display: flex; gap: 0.5rem; align-items: center; }
    input[type="text"] {
      background: #1f2937;
      border: 1px solid #374151;
      color: white;
      padding: 0.45rem 0.75rem;
      border-radius: 0.5rem;
      font-size: 0.85rem;
      font-family: var(--font-mono);
      outline: none;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    .nav-links { display: flex; gap: 1rem; font-size: 0.85rem; }
    .nav-links a { color: var(--text-muted); text-decoration: none; }
    .nav-links a:hover { color: var(--text); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <div class="brand">
          <h1 style="font-size: 1.5rem; font-weight: 700;">LLM Semantic Cache & Proxy</h1>
          <span class="badge badge-core">Open-Core MIT</span>
        </div>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">
          Sub-5ms exact hashing + RediSearch HNSW vector cache + sliding-window budget guard
        </p>
      </div>
      <div style="display: flex; gap: 1rem; align-items: center;">
        <div class="nav-links">
          <a href="/docs" target="_blank">Swagger Docs</a>
          <a href="/metrics" target="_blank">Prometheus /metrics</a>
        </div>
        <div class="status-pill">
          <span class="dot" id="health-dot"></span>
          <span id="health-text">Connecting...</span>
        </div>
      </div>
    </header>

    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem;">
      <div class="controls">
        <label style="font-size: 0.85rem; color: var(--text-muted);">Virtual Key:</label>
        <input type="text" id="api-key" placeholder="Bearer / x-api-key" value="">
        <button class="btn" onclick="fetchData()">Refresh</button>
      </div>
      <div class="controls">
        <button class="btn btn-danger" onclick="purgeCache()">Purge Tenant Cache</button>
      </div>
    </div>

    <!-- Stat Cards -->
    <div class="grid">
      <div class="card">
        <div class="card-title">Cache Hit Rate</div>
        <div class="card-value highlight-accent" id="stat-hit-rate">0.0%</div>
        <div class="card-sub" id="stat-hits-breakdown">0 Exact / 0 Semantic</div>
      </div>
      <div class="card">
        <div class="card-title">USD Cost Saved</div>
        <div class="card-value highlight-success" id="stat-saved-usd">$0.000000</div>
        <div class="card-sub">Estimated API bills avoided</div>
      </div>
      <div class="card">
        <div class="card-title">Upstream Spend</div>
        <div class="card-value" id="stat-spend-usd">$0.000000</div>
        <div class="card-sub" id="stat-tokens">0 prompt / 0 completion tokens</div>
      </div>
      <div class="card">
        <div class="card-title">Total Requests</div>
        <div class="card-value" id="stat-total-requests">0</div>
        <div class="card-sub" id="stat-misses">0 misses / 0 blocked</div>
      </div>
    </div>

    <!-- Live Budget & Quota Section -->
    <div class="card" style="margin-bottom: 2rem;">
      <div class="section-title">
        <span>Tenant Budget & Rate Limits</span>
        <span id="tenant-label" style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-muted);">Tenant: dev-default</span>
      </div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
        <div>
          <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
            <span>Monthly USD Spend</span>
            <span id="budget-spend-text" style="font-family: var(--font-mono); font-weight: 600;">$0.00 / $50.00</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="budget-bar" style="width: 0%;"></div>
          </div>
          <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.35rem;">
            Auto-downgrade enabled at 90% spend (gpt-4o → gpt-4o-mini)
          </div>
        </div>
        <div>
          <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
            <span>Requests Per Minute (60s Window)</span>
            <span id="rpm-text" style="font-family: var(--font-mono); font-weight: 600;">0 / 60 RPM</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="rpm-bar" style="width: 0%; background: #60a5fa;"></div>
          </div>
        </div>
        <div>
          <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
            <span>Tokens Per Minute (60s Window)</span>
            <span id="tpm-text" style="font-family: var(--font-mono); font-weight: 600;">0 / 100,000 TPM</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="tpm-bar" style="width: 0%; background: #a78bfa;"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Models Table -->
    <div class="card">
      <div class="section-title">Supported Models & Pricing Catalog</div>
      <table>
        <thead>
          <tr>
            <th>Model ID</th>
            <th>Provider</th>
            <th>Input (USD / 1M tokens)</th>
            <th>Output (USD / 1M tokens)</th>
          </tr>
        </thead>
        <tbody id="models-body">
          <tr><td colspan="4" style="color: var(--text-muted);">Loading models...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    function getHeaders() {
      const key = document.getElementById('api-key').value.trim();
      const headers = {};
      if (key) {
        if (key.toLowerCase().startsWith('bearer ')) {
          headers['Authorization'] = key;
        } else {
          headers['Authorization'] = 'Bearer ' + key;
        }
      }
      return headers;
    }

    async function checkHealth() {
      try {
        const res = await fetch('/health');
        const data = await res.json();
        const dot = document.getElementById('health-dot');
        const text = document.getElementById('health-text');
        if (data.status === 'ok') {
          dot.style.background = 'var(--success)';
          text.textContent = 'Backend: Redis + Postgres OK';
        } else {
          dot.style.background = 'var(--warning)';
          text.textContent = 'Backend: Degraded';
        }
      } catch (err) {
        document.getElementById('health-dot').style.background = 'var(--danger)';
        document.getElementById('health-text').textContent = 'Backend: Offline';
      }
    }

    async function fetchData() {
      const headers = getHeaders();
      try {
        const [statsRes, budgetRes, modelsRes] = await Promise.all([
          fetch('/v1/stats'),
          fetch('/v1/budget', { headers }),
          fetch('/v1/models')
        ]);

        if (statsRes.ok) {
          const stats = await statsRes.json();
          document.getElementById('stat-hit-rate').textContent = (stats.hit_rate * 100).toFixed(1) + '%';
          document.getElementById('stat-hits-breakdown').textContent = `${stats.exact_hits} Exact / ${stats.semantic_hits} Semantic`;
          document.getElementById('stat-saved-usd').textContent = '$' + stats.total_saved_usd.toFixed(6);
          document.getElementById('stat-spend-usd').textContent = '$' + stats.total_spend_usd.toFixed(6);
          document.getElementById('stat-tokens').textContent = `${stats.total_prompt_tokens.toLocaleString()} prompt / ${stats.total_completion_tokens.toLocaleString()} completion`;
          document.getElementById('stat-total-requests').textContent = stats.total_requests.toLocaleString();
          document.getElementById('stat-misses').textContent = `${stats.cache_misses} misses / ${stats.blocked_requests} blocked`;
        }

        if (budgetRes.ok) {
          const budget = await budgetRes.json();
          document.getElementById('tenant-label').textContent = 'Tenant: ' + budget.tenant_id;
          document.getElementById('budget-spend-text').textContent = `$${budget.current_spend_usd.toFixed(4)} / $${budget.monthly_budget_usd.toFixed(2)}`;
          const spendPct = budget.monthly_budget_usd > 0 ? Math.min(100, (budget.current_spend_usd / budget.monthly_budget_usd) * 100) : 0;
          const bar = document.getElementById('budget-bar');
          bar.style.width = spendPct + '%';
          bar.style.background = spendPct >= 90 ? 'var(--danger)' : (spendPct >= 75 ? 'var(--warning)' : 'var(--accent)');

          document.getElementById('rpm-text').textContent = `${budget.used_rpm} / ${budget.rpm_limit} RPM`;
          const rpmPct = Math.min(100, (budget.used_rpm / budget.rpm_limit) * 100);
          document.getElementById('rpm-bar').style.width = rpmPct + '%';

          document.getElementById('tpm-text').textContent = `${budget.used_tpm.toLocaleString()} / ${budget.tpm_limit.toLocaleString()} TPM`;
          const tpmPct = Math.min(100, (budget.used_tpm / budget.tpm_limit) * 100);
          document.getElementById('tpm-bar').style.width = tpmPct + '%';
        }

        if (modelsRes.ok) {
          const modelsData = await modelsRes.json();
          const tbody = document.getElementById('models-body');
          tbody.innerHTML = modelsData.data.map(m => `
            <tr>
              <td><strong>${m.id}</strong></td>
              <td><span class="badge ${m.owned_by === 'openai' ? 'badge-core' : ''}" style="border: 1px solid #374151;">${m.owned_by}</span></td>
              <td>$${m.input_per_1m_usd.toFixed(2)}</td>
              <td>$${m.output_per_1m_usd.toFixed(2)}</td>
            </tr>
          `).join('');
        }
      } catch (err) {
        console.error('Failed fetching telemetry:', err);
      }
    }

    async function purgeCache() {
      if (!confirm('Purge all exact and semantic cache entries for this tenant?')) return;
      const headers = getHeaders();
      try {
        const res = await fetch('/v1/cache', { method: 'DELETE', headers });
        const data = await res.json();
        alert(`Purged ${data.deleted_exact_keys} exact keys and ${data.deleted_semantic_buckets} semantic buckets.`);
        fetchData();
      } catch (err) {
        alert('Cache purge failed: ' + err.message);
      }
    }

    checkHealth();
    fetchData();
    setInterval(() => { checkHealth(); fetchData(); }, 5000);
  </script>
</body>
</html>
"""
