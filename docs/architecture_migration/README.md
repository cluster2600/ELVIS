# ELVIS architecture migration

This directory is the decision record and execution ledger for the incremental
ELVIS architecture migration. It deliberately separates observed facts, design
decisions, and future work so that a proposed architecture is never confused
with code that is already running.

## Documents

1. [ELVIS repository analysis](01-elvis-repository-analysis.md) — measured
   baseline, current component map, dependency analysis, and failure modes.
2. [Reference architecture analysis](02-reference-architectures.md) — source
   review of OpenMarket, NautilusTrader, Hummingbot, and Freqtrade.
3. [Target architecture](03-target-architecture.md) — the deliberately small
   modular-monolith design selected for ELVIS.
4. [Migration roadmap](04-migration-roadmap.md) — atomic migration slices,
   gates, rollback rules, and current status.

## Evidence snapshot

The analysis was made against these immutable revisions:

| Repository | Revision |
|---|---|
| ELVIS | [`1ffd723a05907ea9e5c2512092f5cf8505cc2725`](https://github.com/cluster2600/ELVIS/commit/1ffd723a05907ea9e5c2512092f5cf8505cc2725) |
| OpenMarket | [`0acbbff81d19d13b2ac99529fd663c3aa19963b2`](https://github.com/gregyoung14/openmarket/commit/0acbbff81d19d13b2ac99529fd663c3aa19963b2) |
| NautilusTrader | [`deff407e44fa5192aab8e1010370e85e094b5d01`](https://github.com/nautechsystems/nautilus_trader/commit/deff407e44fa5192aab8e1010370e85e094b5d01) |
| Hummingbot | [`2bfaccc48dd49e71a5b6d9b3011808e127dd00cd`](https://github.com/hummingbot/hummingbot/commit/2bfaccc48dd49e71a5b6d9b3011808e127dd00cd) |
| Freqtrade | [`c420876a1a092ec16fb959f745a65eac1329f402`](https://github.com/freqtrade/freqtrade/commit/c420876a1a092ec16fb959f745a65eac1329f402) |

The reference repositories were used read-only. No code was copied from them;
the migration adopts architectural ideas only.

## Status vocabulary

- **Observed**: verified in source or by a command recorded in the repository
  analysis.
- **Implemented**: present on this branch and covered by the stated checks.
- **Planned**: accepted direction, not yet implemented.
- **Removed**: deleted only after its replacement is active and verified.

The roadmap is the authoritative status page.
