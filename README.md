# ELVIS V2 operator preview

![Project image](./images/elvis.png)

[![Release](https://img.shields.io/github/v/release/cluster2600/ELVIS?include_prereleases)](https://github.com/cluster2600/ELVIS/releases)
[![CI](https://github.com/cluster2600/ELVIS/actions/workflows/ci.yml/badge.svg)](https://github.com/cluster2600/ELVIS/actions)
[![Python](https://img.shields.io/badge/python-3.14-blue)](INSTALL_V2.md)
[![License](https://img.shields.io/badge/license-BTC__BOT-lightgrey)](LICENSE)

ELVIS is a Python 3.14 paper-trading research system moving to a smaller,
durable V2 architecture. The V2 preview packages the offline PostgreSQL
bootstrap, cut-over preflight, raw-snapshot import, and reconciliation tools so
operators can rehearse migration without granting the new runtime authority.

> **Safety boundary:** paper trading is the only executable bot mode. The
> current compatibility process remains authoritative. The V2 operator tools
> are offline and non-activating; `ACTIVE` remains a **NO-GO**. This software is
> experimental and makes no profitability claim or financial recommendation.

## Install the V2 preview

Use the pinned operator image and release bundle from the GitHub prerelease.
The exact commands, checksum verification, external-secret contract, and
uninstall steps are in [INSTALL_V2.md](INSTALL_V2.md). No Python wheel is
published for this preview.

Do not treat installation, a `COMPLETE` bootstrap receipt, or a healthy
PostgreSQL rehearsal as a production cut-over. The remaining authority gates
are listed in the [V2 roadmap](docs/architecture_migration/04-migration-roadmap.md).

## What is included

- typed, fail-closed signal, policy, risk, and execution contracts;
- durable order, fill, position, and paper-account journals with replay;
- atomic owners and a generation-bound database writer fence;
- least-authority PostgreSQL bootstrap and isolated PostgreSQL 15 rehearsal;
- read-only fresh-target admission;
- bounded raw V1 snapshot import with exact recovery; and
- non-authoritative opening-candidate reconciliation.

The database capabilities above are implemented and tested but not composed
into the bot. Source/runtime provenance, V2 account opening, history replay,
dedicated runtime composition, fail-closed startup, shadow comparison,
rollback rehearsal, soak, and explicit operator approval remain open.

## Compatibility paper runtime

The repository still contains the paper-only compatibility runtime because it
is the rollback authority until V2 cut-over is proven. For local source work:

```bash
git clone https://github.com/cluster2600/ELVIS.git
cd ELVIS
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python main.py --mode paper
```

The root `docker-compose.yml` is retained as compatibility evidence and is not
the V2 installer or an approved production deployment. It still carries shared
development identities and runtime assumptions that are explicit V2 blockers.
Do not use the retired Apple-container or Ansible instructions from an older
checkout; tag `v0.3.0` preserves those files for forensic recovery.

## Documentation

| Need | Canonical document |
|---|---|
| Install or remove the V2 preview | [INSTALL_V2.md](INSTALL_V2.md) |
| V2 approach and safety boundary | [docs/V2_ARCHITECTURE.md](docs/V2_ARCHITECTURE.md) |
| Current gates and status | [V2 migration roadmap](docs/architecture_migration/04-migration-roadmap.md) |
| Operator runbooks | [docs/README.md](docs/README.md) |
| Compatibility paper setup | [docs/PAPER_TRADING_SETUP.md](docs/PAPER_TRADING_SETUP.md) |
| Security | [SECURITY.md](SECURITY.md) |
| Release history | [CHANGELOG.md](CHANGELOG.md) |
| Retired V1 surface | [restore manifest](docs/archive/v1/README.md) |

## Test

```bash
python3.14 -m pytest -q
```

CI runs the unit and PostgreSQL 15 gates on Python 3.14, validates the isolated
V2 rehearsal, scans the repository, and verifies release contracts. The tag
workflow builds the published bundle and image. Exact verification evidence
belongs to the pull request and release, not to an
evergreen success claim in this README.

## License

[BTC_BOT License](LICENSE)
