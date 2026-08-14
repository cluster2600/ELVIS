# Install the ELVIS V2 operator preview

`v2.0.0-alpha.2` is a **paper/migration operator preview** for Python 3.14.
It packages the bounded PostgreSQL bootstrap, cut-over inspection, snapshot
import, and reconciliation tools in one multi-architecture container.

> **ACTIVE: NO-GO.** This release does not contain a V2 trading-runtime
> launcher, does not enable live trading, and does not authorize a cut-over.
> Use it only with paper data, stopped source clones, and fresh or disposable
> targets. Every operator remains responsible for reviewing the emitted JSON
> evidence before a separate activation decision.

## Requirements

- Docker Engine or Docker Desktop with Docker Compose v2
- Linux `amd64` or `arm64`
- anonymous network access to the public `ghcr.io/cluster2600/elvis-v2-operator`
- the GitHub CLI for the shortest verified download path

No Python installation and no wheel are required. The image itself uses only
Python 3.14.

## Download and verify

```bash
TAG=v2.0.0-alpha.2
mkdir elvis-v2-operator-preview
cd elvis-v2-operator-preview

gh release download "$TAG" \
  --repo cluster2600/ELVIS \
  --pattern 'elvis-v2-operator-*.tar.gz' \
  --pattern 'elvis-v2-operator-*.spdx.json' \
  --pattern 'IMAGE_DIGEST.txt' \
  --pattern 'SHA256SUMS' \
  --pattern '*.intoto.jsonl'

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum --check SHA256SUMS
else
  shasum -a 256 --check SHA256SUMS
fi

gh attestation verify "elvis-v2-operator-${TAG}.tar.gz" \
  --repo cluster2600/ELVIS

tar -xzf "elvis-v2-operator-${TAG}.tar.gz"
cd "elvis-v2-operator-${TAG}"
```

Only extract the bundle after both checksum and GitHub attestation verification
succeed. The `IMAGE_DIGEST.txt` inside that authenticated bundle records the
immutable multi-architecture image reference. The release also includes
separate SPDX JSON SBOMs for Linux `amd64` and `arm64`, plus a downloadable
provenance bundle (`*.intoto.jsonl`).

## Pull and smoke-test a clean install

```bash
cp v2-preview.env.example .env
mkdir -p operator

# Prefer the immutable digest over the human-readable prerelease tag.
export ELVIS_V2_OPERATOR_IMAGE="$(cat IMAGE_DIGEST.txt)"

docker compose --env-file .env -f compose.preview.yml config --quiet
docker compose --env-file .env -f compose.preview.yml pull
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator --help
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator --version
```

The help output must list exactly these operator commands:

- `bootstrap`
- `cutover-preflight`
- `import-snapshot`
- `reconcile-snapshot`

There is deliberately no `run`, `trade`, `live`, or `activate` command.

The release gate pulls both image architectures and the clean Compose bundle
without registry credentials. A private package or authenticated-only pull
cannot produce this prerelease.

## Prepare operator inputs

The Compose file mounts `./operator` read-only at `/run/operator`. Copy only the
examples needed for the operation and edit every database name, role, and host
for the isolated environment:

```bash
cp operator-examples/pg_service.preview.conf.example operator/pg_service.conf
cp /absolute/path/to/your-postgresql-ca.crt operator/ca.crt
cp operator-examples/bootstrap-stage-v1.example.json operator/bootstrap.json
cp operator-examples/cutover-preflight-v1.example.json \
  operator/cutover-preflight.json
cp operator-examples/legacy-snapshot-import-v1.example.json \
  operator/import-snapshot.json
cp operator-examples/legacy-snapshot-reconciliation-v1.example.json \
  operator/reconcile-snapshot.json

umask 077
touch operator/pgpass
```

Start with `bootstrap-stage-v1.example.json` when only the administrator
service is available. Replace `operator/bootstrap.json` with the bundled
`bootstrap-complete-v1.example.json` only after the individually scoped role
credentials and matching service entries have been prepared.

The service template uses non-resolving `.example.invalid` hosts as a safety
barrier and requires `sslmode=verify-full` against the operator-controlled
`operator/ca.crt`. Replace every service that you intend to use with a DNS name
present in that server certificate; do not weaken the TLS mode or remove the
barrier from unused services. Add PostgreSQL password entries to
`operator/pgpass` using the standard `host:port:database:user:password` format.
Do not commit, paste into logs, or package this file. `pg_service.conf` service
names must match the selected JSON input. A host PostgreSQL server may be
addressed as `host.docker.internal` only if that name is present in its
certificate; a remote isolated database must use its certificate-verified DNS
name. Plaintext or certificate-unverified PostgreSQL endpoints are rejected by
the bundled template.

The container defaults to unprivileged UID/GID `65532`. For private files owned
by your local account, export your own IDs before each Compose invocation:

```bash
export ELVIS_V2_OPERATOR_UID="$(id -u)"
export ELVIS_V2_OPERATOR_GID="$(id -g)"
```

## Run bounded operator commands

First inspect each command's exact contract:

```bash
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator bootstrap --help
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator cutover-preflight --help
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator import-snapshot --help
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator reconcile-snapshot --help
```

Examples below show the required explicit confirmations; they are not approval
to use a production source or target.

Stage a fresh paper target:

```bash
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator bootstrap \
  --config /run/operator/bootstrap.json \
  --apply \
  --confirm-exclusive-ddl-role-window \
  > operator/bootstrap-receipt.json
```

Inspect a stopped source clone and fresh target:

```bash
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator cutover-preflight \
  --config /run/operator/cutover-preflight.json \
  --inspect \
  --confirm-stopped-source-clone \
  --confirm-exclusive-database-window \
  > operator/preflight-receipt.json
```

Import into a disposable target only:

```bash
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator import-snapshot \
  --config /run/operator/import-snapshot.json \
  --preflight-receipt /run/operator/preflight-receipt.json \
  --import-snapshot \
  --confirm-stopped-source-clone \
  --confirm-exclusive-database-window \
  --confirm-disposable-target \
  > operator/import-receipt.json
```

Perform the read-only reconciliation assessment:

```bash
docker compose --env-file .env -f compose.preview.yml \
  run --rm --no-deps operator reconcile-snapshot \
  --config /run/operator/reconcile-snapshot.json \
  --import-receipt /run/operator/import-receipt.json \
  --assess \
  --confirm-reviewed-database-window \
  --confirm-disposable-target \
  > operator/reconciliation-receipt.json
```

Some safe, incomplete outcomes intentionally use non-zero exit codes while
still emitting a JSON receipt (for example, blocked preflight or decision
required). Treat both the exit status and the receipt as evidence. Neither a
successful command nor a `READY` receipt changes **ACTIVE: NO-GO**.

## Uninstall

The preview has no daemon and creates no named volume:

```bash
docker compose --env-file .env -f compose.preview.yml down --remove-orphans
```

Remove the downloaded directory and local image only when their evidence files
are no longer needed. Never delete the source snapshot or receipts as part of a
routine uninstall.
