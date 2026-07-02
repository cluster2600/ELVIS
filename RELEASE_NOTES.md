# ELVIS v0.1.0

First tagged release of the ELVIS trading bot (Enhanced Leveraged Virtual
Investment System). This release makes the bot run cleanly, removes dead
code, hardens security, and moves the runtime to Python 3.14.

## Docker image

Published to the GitHub Container Registry:

```bash
docker pull ghcr.io/cluster2600/elvis:0.1.0   # or :latest
```

Run in paper mode (needs Postgres + Redis; see `docker-compose.yml`):

```bash
docker run --rm \
  -e POSTGRES_HOST=... -e POSTGRES_PASSWORD=... \
  -e REDIS_HOST=... \
  -p 5050:5050 \
  ghcr.io/cluster2600/elvis:0.1.0
```

The image is a lean runtime built on `python:3.14-slim` (numpy/pandas/sklearn,
Flask API, Binance/CCXT, Redis, Postgres, TA-Lib). Model files and secrets are
**not** baked in — mount `models/` as a volume and supply credentials via
environment variables or Vault. The container runs as a non-root `elvis` user
and exposes a `/health` check on port 5050.

## What's in this release

**Runtime**
- Migrated to Python 3.14; all dependencies refreshed to current versions.
- Fixed a macOS startup segfault (duplicate OpenMP runtimes) and pandas 3.0 API
  removals (`fillna(method=)`, deprecated frequency aliases).
- Hardened the macOS Keychain secret read against indefinite blocking.

**Security**
- Trading API refuses to start without `API_SECRET_KEY`; login fails closed
  unless `API_USERNAME`/`API_PASSWORD` are configured (previously `admin`/`admin`
  with a hard-coded JWT signing key).
- `--mode` is restricted to `paper|live`; the committed Vault token file was
  untracked. **Rotate any previously committed secrets.**
- Default leverage reduced to 3x with a startup safety gate above 10x
  (`OVERRIDE_HIGH_LEVERAGE=true` to bypass); kill-switch and Binance rate
  limiting included.

**Cleanup**
- Removed ~113 verified-dead scripts/modules and large tracked artifacts, plus
  long-tracked virtualenv directories. Historical status reports moved to
  `docs/archive/`.

## Known limitations

- `ydf` and `tensorflow` have no Python 3.14 wheels, and `coremltools` lacks
  native bindings on 3.14. The ensemble runs on the scikit-learn / research /
  torch members and skips the others via existing guards.
- The Docker image is a lean runtime and does not bundle `torch`; the DRL/RL
  strategies are skipped when torch is absent.
