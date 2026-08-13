# ELVIS V2 isolated PostgreSQL rehearsal

This runbook defines M9b.14c3c1: a disposable, fresh PostgreSQL 15 rehearsal
for the dormant ELVIS V2 bootstrap. It is deliberately separate from the
current runtime Compose stack and its persistent volume.

> **No deployment or cut-over is authorised.** This composition contains no
> bot, trainer, activation service, runtime startup hook, or application health
> signal. `ACTIVE` remains a **NO-GO**.

## Rehearsal boundary

```mermaid
flowchart LR
    OP["operator"] --> EXT["external 0600 secrets and libpq files"]
    EXT --> BOOT["one-shot bootstrap container"]
    BOOT --> PG["fresh isolated PostgreSQL 15"]
    PG --> RECEIPT["secret-free receipt"]
    OLD["existing ELVIS volume"] -. "never mounted" .-> PG
    BOT["bot and trainer"] -. "not composed" .-> PG
    ACTIVE["activation"] -. "not invoked" .-> PG
```

Graph artefacts: [Mermaid source](../diagrams/v2-c3c1-rehearsal-boundary.mmd),
[SVG](../diagrams/v2-c3c1-rehearsal-boundary.svg),
[PNG](../diagrams/v2-c3c1-rehearsal-boundary.png), and
[editable Excalidraw](../diagrams/v2-c3c1-rehearsal-boundary.excalidraw).

The composition uses only `deploy/v2/compose.bootstrap.yml`. It does not extend
or override the root `docker-compose.yml`. PostgreSQL has no host-published
port, runs on an internal subnet, uses a pinned PostgreSQL 15 image digest, and
loads an explicit SCRAM-only HBA allowlist followed by IPv4 and IPv6 rejects.

## External operator material

Create a directory outside Git with mode `0700`. Every file in it must use mode
`0600`:

- `postgres_admin_password` for first initialization;
- `pg_service.conf`, based on `deploy/v2/pg_service.conf.example`;
- `pgpass`, containing the admin and later six role credentials;
- `bootstrap.json`, copied first from the stage manifest and later from the
  complete manifest.

Do not place passwords, a populated pgpass, or endpoint overrides in the
repository. The committed JSON and service files are non-secret examples only.

Use one unique, rehearsal-only project name and the external directory on every
command:

```bash
export ELVIS_V2_OPERATOR_DIR=/absolute/path/to/elvis-v2-rehearsal-operator
export ELVIS_V2_REHEARSAL_PROJECT=elvis-v2-rehearsal-review
export ELVIS_V2_OPERATOR_UID="$(id -u)"
export ELVIS_V2_OPERATOR_GID="$(id -g)"

docker compose \
  --project-name "$ELVIS_V2_REHEARSAL_PROJECT" \
  --file deploy/v2/compose.bootstrap.yml \
  --profile v2-rehearsal \
  --profile v2-operator \
  config --quiet
```

The operator directory path must be absolute and must not be a repository
directory. The explicit UID/GID makes the read-only operator container run as
the owner of those mode-0600 files; never replace them with root or relax the
file modes. Replace the example project name with a collision-free name. The
fixed internal subnet deliberately permits only one rehearsal at a time; stop
and clean the first project before starting another.

## Fresh state flow

```mermaid
flowchart TD
    EMPTY["empty labelled volume"] --> INIT["PostgreSQL 15<br/>SCRAM initialized"]
    INIT --> STAGE["bootstrap<br/>stage pass"]
    STAGE --> CREDS["credentials<br/>required"]
    CREDS --> PROVISION["external credential<br/>provisioning"]
    PROVISION --> COMPLETE["bootstrap<br/>complete"]
    COMPLETE --> REPEAT["repeat<br/>complete"]
    EMPTY -->|"pre-existing data"| REJECT["reject"]
    INIT -->|"typed error"| STOP["stop and inspect"]
    STAGE -->|"typed error"| STOP
    PROVISION -->|"failed proof"| STOP
```

Graph artefacts: [Mermaid source](../diagrams/v2-c3c1-fresh-state-flow.mmd),
[SVG](../diagrams/v2-c3c1-fresh-state-flow.svg),
[PNG](../diagrams/v2-c3c1-fresh-state-flow.png), and
[editable Excalidraw](../diagrams/v2-c3c1-fresh-state-flow.excalidraw).

### Procedure

1. Confirm that the project name and volume belong only to this rehearsal.
2. Render the isolated file with `docker compose ... config --quiet`.
3. Start only the `postgres` service under the `v2-rehearsal` profile.
4. Verify PostgreSQL 15, `password_encryption=scram-sha-256`, and zero HBA
   parser errors.
5. Run the one-shot CLI with `bootstrap-stage-v1.example.json`; require exit
   `10`, `CREDENTIALS_REQUIRED`, no `np` schema, and seven staged roles.
6. Provision the six passwords externally using parameterized SQL over the
   admin channel. No production credential writer is included in this slice.
7. Add the six entries to the external libpq files and use
   `bootstrap-complete-v1.example.json`.
8. Run the CLI; require exit `0`, migrations 1 through 6, and `COMPLETE`.
9. Repeat once and require the same terminal result.
10. Prove every role with its own credential and reject crossed credentials.
11. Preserve only secret-free receipts and test evidence, then stop the
    rehearsal services.

The full command syntax, receipt taxonomy, and commit-unknown recovery remain
in the [bootstrap runbook](V2_POSTGRES_BOOTSTRAP.md).

Using the environment from the command block above, the explicit service
operations are:

```bash
docker compose \
  --project-name "$ELVIS_V2_REHEARSAL_PROJECT" \
  --file deploy/v2/compose.bootstrap.yml \
  --profile v2-rehearsal \
  --profile v2-operator \
  up --detach --wait postgres

docker compose \
  --project-name "$ELVIS_V2_REHEARSAL_PROJECT" \
  --file deploy/v2/compose.bootstrap.yml \
  --profile v2-rehearsal \
  --profile v2-operator \
  run --rm --no-deps bootstrap \
  --config /run/operator/bootstrap.json \
  --apply \
  --confirm-exclusive-ddl-role-window
```

## Existing-volume decision

```mermaid
flowchart TD
    SOURCE["stop writers on source"] --> CLONE["verified physical clone"]
    CLONE --> INVENTORY["catalog and ownership inventory"]
    INVENTORY --> OWNER{"shared owner owns PL/pgSQL or catalog?"}
    OWNER -->|"yes"| DEFER["reject and design c3c2 remediation"]
    OWNER -->|"no"| ADOPTION["separate adoption rehearsal"]
    ADOPTION --> REVIEW["rollback and evidence review"]
    DEFER --> REVIEW
```

Graph artefacts:
[Mermaid source](../diagrams/v2-c3c1-existing-volume-decision.mmd),
[SVG](../diagrams/v2-c3c1-existing-volume-decision.svg),
[PNG](../diagrams/v2-c3c1-existing-volume-decision.png), and
[editable Excalidraw](../diagrams/v2-c3c1-existing-volume-decision.excalidraw).

The current ELVIS volume was initialized by the shared runtime superuser. The
official image's initialization variables and HBA defaults apply only to an
empty data directory, so c3c1 must never mount, rename, delete, or demote that
volume. The next slice must use a stopped, verified clone and make an
explicitly reviewed ownership-remediation or fresh-target decision.

M9b.14c3c2 now chooses the fresh-target branch without copying data. Its
[read-only cut-over preflight](V2_FRESH_TARGET_CUTOVER.md) inspects a stopped
source clone and a separately bootstrapped target, requires distinct cluster
system identifiers, and emits only stale-on-return evidence for the next
bounded importer design. The active source volume remains outside both c3c1
and c3c2.

## Rollback and cleanup

Rollback for this slice means removing only the fresh rehearsal project after
capturing secret-free evidence. Verify the exact Compose project and volume
label before any volume deletion. Never use the root project name or its
volume. Credential rotation is not rollback; if a credential result is
uncertain, rotate to a new set in a separately approved operation.

After verifying the project name literally, cleanup is:

```bash
docker compose \
  --project-name "$ELVIS_V2_REHEARSAL_PROJECT" \
  --file deploy/v2/compose.bootstrap.yml \
  --profile v2-rehearsal \
  --profile v2-operator \
  down --volumes --remove-orphans --rmi local
```

## Remaining blockers

- the current bot still performs runtime DDL;
- bot and trainer still share a privileged credential in legacy manifests;
- active Compose, Ansible, and Apple launch paths are not migrated;
- runtime startup and health are not fail-closed on the V2 catalog;
- replay, reconciliation, rollback rehearsal, soak, and operator activation
  approval remain pending.
- the bounded fresh-target importer and its parity proof remain a later slice;
  c3c2 performs inspection only.

## Verification status

The frozen local evidence passed 6 contract tests under both Python 3.10 and
3.14 and 2 opt-in Docker/PostgreSQL 15 scenarios under Python 3.14. The
rehearsal proved the full `10 -> 0 -> 0` flow, six separate SCRAM identities,
the HBA catalog, marker-preserving restart, non-mutating rejection of an
unmarked non-empty volume, secret redaction, and complete Compose cleanup.
Static formatting, compilation, Compose rendering, relative links, and all
three Mermaid source/render sets were green. Pull-request CI and the broader
regression suites remain the acceptance record; none of this authorises
deployment or `ACTIVE`.
