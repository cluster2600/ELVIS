# Compatibility paper runtime

> This guide describes the source-only compatibility process retained for
> rollback. It is not the V2 installer, a production deployment guide, or live
> trading. `ACTIVE` remains a **NO-GO**.

## Prerequisites

- Python 3.14;
- PostgreSQL reachable through the variables used by `config/config.py`;
- a local Vault/OpenBao instance or another reviewed secret source; and
- Binance testnet credentials if a testnet market-data path is required.

Use [INSTALL_V2.md](../INSTALL_V2.md) for the packaged V2 operator preview.

## Source setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

export TRADING_MODE=paper
python main.py --mode paper
```

`TRADING_MODE=paper` selects the configured Binance testnet endpoint.
`main.py --mode paper` selects the only executable bot mode. The retained
`live` CLI value fails before bootstrap because this checkout has no validated
live-submission capability.

## Balances and resets

The compatibility code contains two different historical balance conventions:
`config/config.py` defaults to 100 USDT and no BNB seed, while
`utils/paper_trade_db.py` and the reset utilities retain an older 1,000 USDT +
1,000 BNB database convention. Until that ownership is reconciled, do not
present either as the V2 opening balance and do not infer performance from a
dashboard fallback.

The reset command is destructive to compatibility paper data:

```bash
python -m scripts.reset_paper_trading
```

Run it only against an explicitly selected disposable paper database after
stopping the bot. A process restart is not a reset. The V2 importer and
reconciliation tools never use this reset path and never invent an opening
balance.

## Safety limits

- Paper fills are simulations; they do not prove venue behaviour, slippage, or
  a live order path.
- Compatibility and V2 accounting are not interchangeable.
- Root Compose contains shared development credentials and is not approved for
  production or V2 cut-over.
- No win rate, Sharpe ratio, profit, or equivalence-to-live claim is made.
- Secrets must stay outside Git, command arguments, logs, and receipts. See
  [SECURITY.md](../SECURITY.md).

The [migration roadmap](architecture_migration/04-migration-roadmap.md) lists
the provenance, replay, composition, rollback, soak, and approval gates that
still prevent V2 activation.
