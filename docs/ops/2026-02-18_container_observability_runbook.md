# 2026-02-18 - Container observability and hardening runbook

## Scope
This runbook documents the post-hardening behavior introduced on February 18, 2026 (issues `#9`, `#10`, `#11`, `#13`, `#16`) and how to keep Grafana dashboards populated in container deployments.

## Security-sensitive runtime defaults

### Required environment behavior
- `VAULT_TOKEN` has no hardcoded fallback.
- `POSTGRES_PASSWORD` has no hardcoded fallback.
- Trade History API defaults to local bind:
  - `TRADE_HISTORY_API_HOST=127.0.0.1`
  - `TRADE_HISTORY_API_PORT=5050`
- Flask trade-history API now requires `X-API-Key: <API_KEY>` for all routes except `/health`.

### Practical impact
- If you do not provide `VAULT_TOKEN`, Vault-backed secret retrieval is unavailable.
- If you do not provide `POSTGRES_PASSWORD`, DB auth may fail.
- If you keep default API host binding (`127.0.0.1`), remote/container scrapers cannot reach the API unless they share the same namespace.

## Container startup (monitoring stack)

From project root:

```bash
docker compose up -d postgres redis elvis-bot prometheus grafana loki promtail
docker compose ps
```

Expected host endpoints:
- Grafana: `http://localhost:3001`
- Prometheus: `http://localhost:9090`
- Trade API health: `http://localhost:5050/health`

## Verify metrics pipeline end-to-end

1. Verify ELVIS API process is healthy:

```bash
curl -s http://localhost:5050/health
```

2. Verify metrics endpoint responds:

Without API auth:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5050/metrics
```

With API auth:
```bash
curl -s -H "X-API-Key: ${API_KEY}" -o /dev/null -w "%{http_code}\n" http://localhost:5050/metrics
```

3. Verify Prometheus target status:

```bash
curl -s http://localhost:9090/api/v1/targets
```

For job `elvis`, confirm `health` is `up`.

4. In Grafana, verify datasource and dashboard:
- Datasource: `Prometheus` (provisioned via `grafana/provisioning/datasources/prometheus.yml`)
- Dashboard folder: `ELVIS`

## Grafana "No data" troubleshooting

### 1) ELVIS service not running
Symptom:
- Prometheus target state is `down` with connection refused.

Fix:
```bash
docker compose up -d elvis-bot
docker compose logs --tail=200 elvis-bot
```

### 2) Scrape target mismatch
If ELVIS runs inside the same Compose network as Prometheus:
- Preferred target: `elvis-bot:5050`

If ELVIS runs on host and Prometheus runs in container:
- Target can be: `host.docker.internal:5050`

Update `observability/prometheus.yml` accordingly and restart Prometheus:
```bash
docker compose restart prometheus
```

### 3) API auth blocking `/metrics`
Because API auth now applies to most routes, `/metrics` may return `401`/`503`.

Fix option A (recommended):
- Configure Prometheus to send `X-API-Key`.

Example:
```yaml
- job_name: 'elvis'
  static_configs:
    - targets: ['elvis-bot:5050']
  metrics_path: '/metrics'
  scheme: 'http'
  scrape_interval: 10s
  http_config:
    headers:
      X-API-Key: '<same value as API_KEY>'
```

Fix option B:
- Exempt `/metrics` from Flask auth middleware if your deployment model requires anonymous local scrape.

### 4) API bound to localhost only
If Prometheus is remote (container/host boundary), local-only bind can prevent access.

Fix:
- Set `TRADE_HISTORY_API_HOST=0.0.0.0` intentionally in deployment env.

## Repository hygiene policy

To prevent repo bloat and accidental secret/artifact commits:
- Do not commit local environments (`env*/`, `venv*/`, `.venv/`).
- Do not commit local ML source/build directories (for example `tensorflow/`).
- Keep generated logs/data/model artifacts out of git unless explicitly versioned.

If large local directories are accidentally tracked:

```bash
git rm -r --cached env-coreml env-ydf .venv venv tensorflow
git commit -m "chore(repo): stop tracking local environment/build artifacts"
```

## Related files
- `README.md`
- `SECURITY.md`
- `docker-compose.yml`
- `observability/prometheus.yml`
- `trading/utils/trade_history_api.py`
