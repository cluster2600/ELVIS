# ELVIS documentation

This documentation is organised around the Python 3.14 V2 operator preview.
The preview is installable, but it is not an active or production-authorised
trading runtime. The compatibility paper process remains authoritative and
`ACTIVE` remains a **NO-GO**.

## Start here

1. [Install the V2 preview](../INSTALL_V2.md).
2. Read the [V2 architecture and safety boundary](V2_ARCHITECTURE.md).
3. Check the [current migration gates](architecture_migration/04-migration-roadmap.md).
4. Choose only the operator runbook that matches the reviewed operation:
   [bootstrap](V2_POSTGRES_BOOTSTRAP.md),
   [isolated rehearsal](V2_POSTGRES_REHEARSAL.md),
   [fresh-target preflight](V2_FRESH_TARGET_CUTOVER.md),
   [raw snapshot import](V2_LEGACY_SNAPSHOT_IMPORT.md), or
   [reconciliation review](V2_LEGACY_SNAPSHOT_RECONCILIATION.md).

## Authority map

| Document | Authority |
|---|---|
| [V2 architecture](V2_ARCHITECTURE.md) | Canonical approach and safety contract |
| [Target architecture](architecture_migration/03-target-architecture.md) | Detailed component and data contracts |
| [Migration roadmap](architecture_migration/04-migration-roadmap.md) | Current status and open gates |
| `V2_*` runbooks | Offline operator procedures; never activation authority |
| [Compatibility architecture](architecture.md) | Current paper-runtime topology and rollback context |
| [Paper setup](PAPER_TRADING_SETUP.md) | Source-only compatibility operation |
| [V1 restore manifest](archive/v1/README.md) | Historical recovery pointer, not instructions |

Source and tests win if prose drifts. An installed package, generated receipt,
successful test, or healthy container is evidence only; none changes runtime
authority.

## Compatibility references

The following pages remain because the current paper runtime is still the
rollback authority. Each is explicitly labelled as compatibility material:

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
