# Deployment status

ELVIS V2 is available only as an **operator preview**. Install it with the
pinned operator image and release bundle using the exact procedure in
[INSTALL_V2.md](../INSTALL_V2.md). That installation exposes offline migration
tools; it does not deploy or activate the V2 paper runtime.

## Current boundaries

- Python 3.14 is the only supported interpreter.
- Paper trading is the only executable bot mode.
- `deploy/v2/compose.bootstrap.yml` is an isolated PostgreSQL rehearsal, not a
  production stack.
- Root `docker-compose.yml` is retained compatibility evidence with shared
  development credentials and runtime assumptions; it is not the V2 install
  path and is not production-approved.
- The former Ansible tree and Apple-container helpers were not verified,
  contained unsafe defaults, and have been removed. Tag `v0.3.0` is the
  forensic restore point; do not restore those scripts as deployment advice.

## Production gate

No production deployment is authorised until all of these are evidenced:

1. authenticated source/runtime opening provenance;
2. deterministic V2 account opening and replay;
3. dedicated runtime identities, external SCRAM secrets, restrictive HBA, and
   network policy;
4. removal of runtime DDL and migration authority;
5. fail-closed startup and health on exact catalog, identity, generation, and
   runtime authority;
6. side-effect-free shadow comparison and stale-writer proof;
7. pause/rollback rehearsal and soak; and
8. explicit operator approval.

The [migration roadmap](architecture_migration/04-migration-roadmap.md) is the
authority for these gates. A release bundle, image, receipt, passing CI run, or healthy
database is not deployment or cut-over proof. `ACTIVE` remains a **NO-GO**.
