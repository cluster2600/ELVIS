# ELVIS V2 architecture migration

This directory is the evidence, decision record, and execution ledger for the
incremental ELVIS V2 architecture programme. V2 is a new modular-monolith
approach built around pure typed decisions, immutable journal facts, atomic
state owners, a generation-bound activation fence, and least-authority
PostgreSQL identities.

> **Programme status:** the Python 3.14 operator preview is installable, but it
> is not a deployed or activated V2 runtime. The compatibility paper process
> remains authoritative and `ACTIVE` remains a
> **NO-GO**. The concise [V2 overview](../V2_ARCHITECTURE.md) describes the
> historical alpha.2 target; documents 05–07 below supersede its authority and
> rollback model for trajectory B/1B production work.

The documents deliberately separate observed facts, selected design,
implemented-but-dormant components, and future work. A proposed or tested
architecture must never be confused with code that is composed into the
running bot.

## Documents

1. [ELVIS repository analysis](01-elvis-repository-analysis.md) — immutable
   pre-V2 baseline, component map, dependency analysis, and failure modes.
2. [Reference architecture analysis](02-reference-architectures.md) — source
   review that informed V2 without copying their topology or code.
3. [Target architecture](03-target-architecture.md) — historical alpha.2 target
   design; useful background, but it does not authorise a trajectory-B cut-over.
4. [Migration roadmap](04-migration-roadmap.md) — historical alpha.2 delivery
   ledger. Its V1 rollback language is superseded for paper production by
   documents 05–07.
5. [Paper-production plan](05-v2-production-plan.md) — authoritative production
   delivery sequence, approved trajectory B/1B invariants, and async
   virtual-venue design.
6. [Paper-production failure register](06-v2-production-failure-register.md) —
   open P0/P1/P2 defects, regression expectations, closure gates, and rollback
   rules.
7. [Paper-production E2E matrix](07-v2-production-e2e-matrix.md) — G0 through
   G17 acceptance scenarios and immutable evidence requirements.

## V2 status at a glance

| Layer | State | Meaning |
|---|---|---|
| Typed domain, order service, feature contracts, and selected policy/risk boundaries | Implemented incrementally | Some boundaries serve the compatibility runtime |
| Durable journals, replay, account ledger, atomic owners, readiness, fence, generations, activation, and role/catalog bootstrap | Implemented and tested | Dormant; no running consumer or authority |
| Offline bootstrap, stopped-clone/fresh-target preflight, bounded raw V1 import, and read-only imported-vs-operator-hypothesis review | Implemented and packaged in the preview | Dormant; review receipts are cross-snapshot, source-unauthenticated, stale, and non-authoritative; no match or account opening |
| Signed fresh-opening approval, one-way V1 retirement, activation/candidate binding, async virtual venue with pending-order holds, dedicated production composition, V2-only recovery, backup/restore, soak, release and course evidence | Planned; every runtime-production defect remains open | Blocks `ACTIVE` |

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

Documents 05–07 are the authoritative production programme, open-defect ledger
and acceptance contract. The repository remains the source of truth, and an
explicit operator decision is required for every deployment or cut-over action.
