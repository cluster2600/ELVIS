# Multi-exchange compatibility boundary

> This page records an experimental compatibility feature. It is not an
> approved order-routing service, an arbitrage claim, or a live-trading guide.
> The executable bot remains paper-only and `ACTIVE` remains a **NO-GO**.

`trading.execution.exchange_manager.ExchangeManager` can aggregate configured
exchange adapters for price observation, health, and balances. Kraken and
Coinbase adapters are optional and require separately reviewed credentials.
Their presence in source does not prove sandbox coverage, production fitness,
best execution, or profitable price differences.

## API read models

When an exchange manager is registered, the compatibility API exposes price,
health, and portfolio read models. When it is absent, endpoints return
`available: false` with empty typed payloads rather than fabricated values.
Clients must check that flag.

The endpoint family is implemented under `trading/api/app.py`:

- `GET /api/exchanges`;
- `GET /api/exchanges/prices/<symbol>`;
- `GET /api/arbitrage/opportunities?symbol=BTCUSDT`;
- `GET /api/portfolio/consolidated`; and
- `GET /api/exchanges/health`.

These compatibility endpoints require configured API authentication. An
observed spread is not executable profit: it excludes or may stale across fees,
funding, inventory, transfer latency, order-book depth, venue rejection, and
fill uncertainty.

## Safety contract

- Do not configure production exchange credentials for this preview.
- Do not call routing or execution methods as a substitute for the global
  paper-only capability gate.
- Keep secrets in a reviewed store and out of `.env`, logs, documentation, and
  release bundles.
- Treat API and adapter tests as contract evidence only; they mock exchanges
  and do not validate live venue semantics.

Multi-exchange work remains compatibility context, not a V2 cut-over gate. See
the [V2 roadmap](architecture_migration/04-migration-roadmap.md).
