# ELVIS V2 architecture migration

This directory is the evidence, decision record, and execution ledger for the
incremental ELVIS V2 architecture programme. V2 is a new modular-monolith
approach built around pure typed decisions, immutable journal facts, atomic
state owners, a generation-bound activation fence, and least-authority
PostgreSQL identities.

> **Programme status:** in progress on
> `codex/elvis-architecture-migration`; not released, deployed, or activated.
> The compatibility paper runtime remains authoritative and `ACTIVE` remains a
> **NO-GO**. See the concise [V2 overview](../V2_ARCHITECTURE.md) before using
> the detailed records below.

The documents deliberately separate observed facts, selected design,
implemented-but-dormant components, and future work. A proposed or tested
architecture must never be confused with code that is composed into the
running bot.

## Documents

1. [ELVIS repository analysis](01-elvis-repository-analysis.md) — immutable
   pre-V2 baseline, component map, dependency analysis, and failure modes.
2. [Reference architecture analysis](02-reference-architectures.md) — source
   review that informed V2 without copying their topology or code.
3. [Target architecture](03-target-architecture.md) — the deliberately small
   V2 modular-monolith design and its detailed contracts.
4. [Migration roadmap](04-migration-roadmap.md) — atomic migration slices,
   verification gates, rollback rules, and authoritative current status.

## V2 status at a glance

| Layer | State | Meaning |
|---|---|---|
| Typed domain, order service, feature contracts, and selected policy/risk boundaries | Implemented incrementally | Some boundaries serve the compatibility runtime |
| Durable journals, replay, account ledger, atomic owners, readiness, fence, generations, activation, and role/catalog bootstrap | Implemented and tested | Dormant; no running consumer or authority |
| Offline bootstrap, stopped-clone/fresh-target preflight, and bounded raw V1 import | Implemented locally; acceptance gates in progress | Dormant; receipts remain stale and non-authoritative |
| Dedicated production composition, V2 replay/reconciliation of imported history, shadow comparison, rollback rehearsal, soak, and approval | Pending evidence | Blocks `ACTIVE` |

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

- **Observed**: verified in source or by a recorded command.
- **Implemented**: present on this branch and covered by the stated checks; this
  does not imply deployment or runtime composition.
- **Dormant**: implemented but unreachable from the production composition
  root and without runtime authority.
- **Planned**: accepted direction, not yet implemented.
- **Removed**: deleted only after its replacement is active and verified.

The roadmap is the authoritative status page. The repository remains the source
of truth, and an explicit operator decision is required for every deployment or
cut-over action.
