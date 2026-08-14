# ELVIS documentation

This documentation is organised around the Python 3.14 V2 operator preview.
The preview is installable, but it is not an active or production-authorised
trading runtime. The compatibility paper process remains authoritative and
`ACTIVE` remains a **NO-GO**.

## Start here

1. [Install the V2 preview](../INSTALL_V2.md).
2. Read the [historical preview architecture](V2_ARCHITECTURE.md) for background.
3. Use the authoritative [production plan](architecture_migration/05-v2-production-plan.md),
   [failure register](architecture_migration/06-v2-production-failure-register.md),
   and [E2E gates](architecture_migration/07-v2-production-e2e-matrix.md).
4. Choose only the operator runbook that matches the reviewed operation:
   [fresh-opening preparation](V2_FRESH_OPENING_PLAN.md),
   [bootstrap](V2_POSTGRES_BOOTSTRAP.md),
   [isolated rehearsal](V2_POSTGRES_REHEARSAL.md),
   [fresh-target preflight](V2_FRESH_TARGET_CUTOVER.md),
   [raw snapshot import](V2_LEGACY_SNAPSHOT_IMPORT.md), or
   [reconciliation review](V2_LEGACY_SNAPSHOT_RECONCILIATION.md).

## Authority map

| Document | Authority |
|---|---|
| [V2 architecture](V2_ARCHITECTURE.md) | Historical alpha.2 preview background; superseded for production authority |
| [Target architecture](architecture_migration/03-target-architecture.md) | Historical alpha.2 component contracts |
| [Migration roadmap](architecture_migration/04-migration-roadmap.md) | Historical alpha.2 delivery ledger |
| [Production plan](architecture_migration/05-v2-production-plan.md) | Authoritative trajectory-B/1B design and delivery contract |
| [Failure register](architecture_migration/06-v2-production-failure-register.md) | Authoritative open production blockers |
| [E2E matrix](architecture_migration/07-v2-production-e2e-matrix.md) | Authoritative G0–G17 acceptance evidence |
| `V2_*` runbooks | Offline operator procedures; never activation authority |
| [Compatibility architecture](architecture.md) | Current pre-cut-over paper-runtime topology and evidence context |
| [Paper setup](PAPER_TRADING_SETUP.md) | Source-only compatibility operation |
| [V1 restore manifest](archive/v1/README.md) | Historical recovery pointer, not instructions |

Source and tests win if prose drifts. An installed package, generated receipt,
successful test, or healthy container is evidence only; none changes runtime
authority.

## Compatibility references

The following pages remain because the current paper runtime is still the
temporary writer before the one-way V2 cut-over. It is not a future rollback
authority, and none of its state may seed the trajectory-B production opening.
Each page is explicitly labelled as compatibility material:

- [runtime architecture](architecture.md);
- [system topology](ELVIS_SYSTEM_ARCHITECTURE.md);
- [component catalogue](COMPONENTS.md);
- [trading system](trading_system.md);
- [data processing](data_processing.md);
- [training](training.md); and
- [security boundaries](SECURITY_GUIDE.md).

The old Apple-container guides, unverified Ansible automation, completion
reports, and unsafe deployment recipes were removed from the active tree. Tag
`v0.3.0` preserves their exact bytes; use the
[restore manifest](archive/v1/README.md) for read-only inspection.
