# 2026-02-10 — ELVIS Grafana "No data" debug (dashboard)

## Goal
Fix Grafana dashboard panels showing **No data** by restoring the metrics pipeline.

## Symptoms
- Grafana panels show **No data** for ELVIS Trading Bot dashboard.

## Findings
### 1) Prometheus scrape target down
Prometheus job `elvis_trading` was **DOWN** because `localhost:8888/metrics` refused connection.

Command:
```bash
curl -s http://localhost:9090/api/v1/targets
```
Observed:
- `elvis_trading`: DOWN — dial tcp connect refused

### 2) ELVIS service/container missing
Monitoring stack currently running is the compose project:
- `/Users/maxime/.openclaw/workspace/elvis-monitoring/docker-compose-lite.yml`

Containers up:
- `elvis-grafana`
- `elvis-prometheus`
- `elvis-pushgateway`

But the expected trading service container **`elvis-trading`** (ports `8888:8888`) was not running.

### 3) ELVIS codebase located
ELVIS repo lives at:
- `/Users/maxime/BTC_BOT/BTC_BOT`
Remote:
- `https://github.com/cluster2600/ELVIS.git`

## Action taken
### Start ELVIS service in monitoring stack
Triggered:
```bash
cd /Users/maxime/.openclaw/workspace/elvis-monitoring
docker compose -f docker-compose-lite.yml up -d elvis
```

This required building image `elvis-monitoring-elvis` (Dockerfile installs TA-Lib and dependencies).

## Next verification steps
After build completes:
1) `curl http://localhost:8888/health`
2) `curl http://localhost:8888/metrics` (should return Prometheus text exposition)
3) Confirm Prometheus target `elvis_trading` becomes **UP**
4) Confirm Grafana panels show time-series.
