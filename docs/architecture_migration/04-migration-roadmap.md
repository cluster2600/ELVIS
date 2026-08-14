# ELVIS V2 alpha.2 migration ledger (historical)

> **Historical ledger, not production authority:** V2 operator tooling is
> packaged as a Python 3.14 preview. The compatibility paper process remains
> the current runtime authority.
> Installation, a receipt, or a green test does not authorise `ACTIVE`, which
> remains a **NO-GO**. Trajectory B/1B production is governed exclusively by
> [05](05-v2-production-plan.md), [06](06-v2-production-failure-register.md),
> and [07](07-v2-production-e2e-matrix.md).

The detailed command-by-command development ledger was retired from the active
tree because it repeated obsolete interpreter commands and stale test counts.
Tag `v0.3.0` preserves the pre-release history. Pull requests and CI runs are
the evidence source for individual slices.

## Delivery contract

Every migration slice must state its invariant and blast radius, add focused
tests, remain reversible, pass Python 3.14 and PostgreSQL gates appropriate to
the change, update the relevant operator runbook, and land without changing
runtime authority. Live trading is outside this programme.

## Current state

| Area | State | Runtime meaning |
|---|---|---|
| Typed signal, policy, risk, order, fill, position, and account contracts | Implemented and tested | Selected compatibility boundaries use them |
| Versioned migrations, journals, replay, account ledger, fence, generations, and activation capabilities | Implemented and tested | Dormant; no running V2 owner |
| Least-authority role/catalog bootstrap | Implemented and packaged | Offline operator action only |
| Isolated PostgreSQL 15 bootstrap rehearsal | Implemented and tested | Disposable proof; never mounts the active volume |
| Stopped-clone/fresh-target preflight | Implemented and packaged | Read-only, stale-on-return evidence; copies nothing |
| Seven-table raw legacy snapshot import | Implemented and packaged | Bounded preservation only; synthesises no V2 history |
| Imported-vs-operator-hypothesis reconciliation | Implemented and packaged | Always requires a decision; proves no provenance |
| Source-authenticated opening provenance and V2 account opening | Pending | Blocks replay and authority |
| Dedicated runtime composition and fail-closed health | Pending | Shared compatibility credentials/DDL still block authority |
| Shadow parity, stale-writer proof, rollback rehearsal, and soak | Pending | Blocks operator approval |
| V1 runtime retirement | Pending separately approved one-way cut-over | Compatibility process is the current writer, never the post-retirement rollback authority |

## Superseded production direction

This ledger ended with the alpha.2 offline toolchain. It must not be used to
select opening data, replay V1 history into the production account, restore V1
writer authority, or define the async venue. The approved production direction
is a fresh signed V2 opening, a one-way V1 writer retirement, a durable 1B
asynchronous virtual venue, V2-only rollback, and G0–G17 evidence. The exact
open work and pass/fail rules live in documents 05–07.

## Operator path available in the preview

```text
fresh PostgreSQL rehearsal
        -> offline least-authority bootstrap
        -> stopped-clone / fresh-target preflight
        -> bounded seven-table raw import
        -> read-only candidate reconciliation
        -> DECISION_REQUIRED or BLOCKED
```

Each step has a strict, secret-free intent/receipt contract and separately
revalidates mutable evidence. Connection details and passwords stay in
external libpq files. The tools never write secrets, call activation, compose a
bot, or claim a coherent source/runtime provenance chain.

Use these canonical runbooks:

- [offline bootstrap](../V2_POSTGRES_BOOTSTRAP.md);
- [isolated rehearsal](../V2_POSTGRES_REHEARSAL.md);
- [fresh-target preflight](../V2_FRESH_TARGET_CUTOVER.md);
- [bounded snapshot import](../V2_LEGACY_SNAPSHOT_IMPORT.md); and
- [reconciliation review](../V2_LEGACY_SNAPSHOT_RECONCILIATION.md).

## Verification gates

Release candidates require, at minimum:

- Python 3.14 unit and PostgreSQL 15 suites;
- isolated fresh-cluster rehearsal and disposable-resource cleanup;
- package and operator-image install smoke tests;
- static format/import checks and strict JSON/YAML validation;
- relative Markdown-link validation;
- release-artefact checksums and version consistency; and
- repository/security scanning.

Exact counts and commit hashes belong to immutable CI and release records.
Documentation intentionally avoids copying totals that become stale.

## Rollback and retention

Git tag `v0.3.0` preserves retired V1 documents and deployment experiments for
forensic inspection. Restoring files is not runtime rollback. Runtime rollback
must set kill, drain accepted V2 work, pass through `PAUSED`, verify generation,
writer fence and rollback-candidate compatibility, and activate only a V2
candidate. V1 authority is never restored after retirement.

The [V1 restore manifest](../archive/v1/README.md) gives read-only inspection
commands. No old Apple or Ansible helper is an approved deployment path.

## Definition of migration complete

Migration is complete only when the V2 process is the sole proven paper writer,
all G0–G17 production gates are closed, the entire release gate is green, and
the operator has explicitly approved cut-over. Until then, the preview remains
offline migration tooling and `ACTIVE` remains a **NO-GO**.
