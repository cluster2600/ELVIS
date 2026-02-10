# 2026-02-10 — Docker build SIGKILL (ELVIS)

## Context
Goal: bring `elvis-trading` container up so Prometheus target `elvis_trading` (`localhost:8888/metrics`) becomes **UP** and Grafana stops showing **“No data”**.

Compose stack: `~/.openclaw/workspace/elvis-monitoring/docker-compose-lite.yml`.

## Symptoms
- `docker compose ... build elvis` repeatedly fails with **signal SIGKILL**.
- The SIGKILL occurs at inconsistent steps:
  - `load build context`
  - `apt-get update`
  - `pip install ...`

## Findings
### 1) Docker VM memory was initially too low
- Docker Desktop VM was running with ~8GB (`--memoryMiB 8092`).
- After increasing Docker Desktop resources, VM now runs with ~32GB:
  - `docker info` shows `MemTotal≈33GB`
  - `com.docker.virtualization` shows `--memoryMiB 32512`

### 2) Build context was huge due to repo size + missing dockerignore patterns
- Repo size: ~12GB.
- Context tarball (before exclusions): ~1.4GB.

Actions taken to reduce build context:
- Tightened `.dockerignore` to exclude:
  - `venv_new/`
  - `loki/`
  - `grafana/`
  - `*.tar`
  - (and already-existing exclusions for `data/`, `logs/`, `models/`, etc.)

Result:
- Context tarball dropped to ~411MB.

### 3) Even after cache prune + memory increase, SIGKILL persists
- Ran prunes to remove build cache + containers:
  - `docker builder prune -f`
  - `docker system prune -f`

- SIGKILL still occurs very early in the build, sometimes immediately after “load build context”.

## Hypothesis
- Not a deterministic application error: likely a Docker Desktop / BuildKit instability or resource pressure causing abrupt termination.

## Next steps (planned)
1) Implement a **metrics-only minimal image** (runtime-only requirements) to restore observability quickly:
   - minimal Dockerfile for `:8888/health` + `:8888/metrics`
   - avoid heavy training dependencies
2) Once metrics flow is restored, revisit full image build:
   - split `requirements.txt` into runtime vs training
   - or build the heavy image via CI runner

