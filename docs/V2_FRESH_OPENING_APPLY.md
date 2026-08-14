# ELVIS V2 durable fresh-opening apply

> **Durable, one-shot source command.** This procedure can create exactly one
> fresh trajectory-B paper account, or resolve its exact target-local replay.
> It does not retire V1, select a sole writer, activate a runtime or authorise
> trading. A successful receipt remains `LEGACY/0/S0` with
> `runtime_activation_authorized=false` and `trading_authorized=false`.

## Availability and boundary

Run this command only from the reviewed PR3 source checkout with Python 3.14:

```bash
python3.14 -m scripts.v2_opening_apply --help
```

The historical `v2.0.0-alpha.2` operator image, four-command dispatcher and
release smoke tests do not contain `opening-apply`. Do not copy this module into
that immutable image or describe the source command as a released installation.

The command composes the frozen application service with the public
`PostgresFreshOpeningProvisioning(connection_factory)` adapter. That adapter
owns the database fence, replay-first nonce inspection, current authority check
at database time, physical-target admission and one atomic transaction. The CLI
does not issue SQL itself. Its connection is an independently authenticated,
short-lived database-owner superuser session used only by the offline control
plane. It is never a runtime credential.

The managed `opening` role is only an admission-bound provenance anchor. It is
`NOLOGIN`, has a null password, has no membership, HBA or libpq service entry,
and receives no database, schema, table, sequence or function privilege. The
database-owner control plane is already the PostgreSQL trust root; a compromised
database owner is outside this slice's threat model.

## Required frozen evidence

Complete the controlled key-generation and approval ceremonies in the
[read-only fresh-opening plan](V2_FRESH_OPENING_PLAN.md) first. The exact intent,
detached approval, trust policy and independently authenticated policy/key pins
are all mandatory for apply. Never regenerate a nonce or edit signed bytes to
make a failed command pass.

The physical-target document is separate from the business approval because a
restore changes physical identity without rewriting the approved logical
target. The pin flow deliberately uses two non-circular records:

1. before bootstrap, an operator-approved pin-authority record binds the exact
   candidate, logical target and deployment incarnation; it does **not** contain
   the terminal catalog digest;
2. after terminal bootstrap, a target document binds that record's SHA-256 to
   the observed database name, `system_identifier`, control-plane role, inert
   opening anchor and terminal catalog digest.

The invoking operator must obtain the canonical SHA-256 of the second document
through an independent, access-controlled approval channel. Hashing the same
local file immediately before invocation is not independent authentication.
This is an operator/control-plane trust ceremony, not an Ed25519 proof performed
by PostgreSQL.

Copy the deliberately invalid
[`fresh-opening-target-v1.template.json`](examples/v2/fresh-opening-target-v1.template.json)
outside the checkout and replace every sentinel from a frozen, authenticated
target-admission record:

```bash
install -d -m 0700 /absolute/private/path/elvis-opening-apply
install -m 0600 \
  docs/examples/v2/fresh-opening-target-v1.template.json \
  /absolute/private/path/elvis-opening-apply/target.json
```

The document has exactly these version-1 fields:

- expected database name and decimal PostgreSQL `system_identifier`;
- the database-owner control-plane role and inert `NOLOGIN` opening anchor;
- deployment-incarnation identifier;
- terminal migration/catalog SHA-256; and
- SHA-256 of the authenticated record that supplied those pins.

The repository template uses system identifier `0` and zero fingerprints, so
it is valid JSON but always invalid input. It is never target-admission proof.

Configure a dedicated, short-lived libpq service for the database-owner
control plane outside Git. Pass only its lowercase service name as
`--admin-service`; never place a DSN, password,
`PGPASSWORD`, CA private material or connection URI in an argument, JSON file,
receipt, screenshot or video. The service must use the independently reviewed
TLS and `verify-full` boundary from the PostgreSQL bootstrap runbook.

Before apply, independently verify that:

- migration `0007` is the exact terminal migration and bootstrap version 2 is
  complete;
- the dedicated physical target and deployment incarnation match the frozen
  record;
- the target has no prior V2 economic state except a possible exact replay of
  these bytes;
- the inert opening anchor cannot log in or invoke any capability;
- the admin service is not present in runtime, course or deployment assets and
  no unreviewed database-owner session exists;
- an exclusive opening window is established on the dedicated fresh target;
  and
- the signed approval is still current for an absent/new mutation.

## Apply once

Use the same intent, approval, policy and out-of-band pins that passed the
read-only planner:

```bash
python3.14 -m scripts.v2_opening_apply \
  --intent /absolute/private/path/elvis-opening-plan/intent.json \
  --approval /absolute/private/path/elvis-opening-plan/approval.json \
  --trust-policy /absolute/private/path/elvis-opening-plan/trust-policy.json \
  --target /absolute/private/path/elvis-opening-apply/target.json \
  --pinned-target-document-sha256 <authenticated-lowercase-sha256> \
  --pinned-trust-policy-sha256 <authenticated-lowercase-sha256> \
  --pinned-signer-public-key-sha256 <authenticated-lowercase-sha256> \
  --admin-service elvis_target_opening_admin \
  --apply-opening \
  --confirm-dedicated-fresh-target \
  --confirm-exclusive-opening-window \
  > /absolute/private/path/elvis-opening-apply/result.json
```

The three confirmation flags record explicit invocation intent and are checked
before any input file is opened or libpq service is resolved. They are not
authentication and cannot replace the detached signature, out-of-band pins,
controlled ceremonies or database admission checks. Redirection is performed
by the invoking shell. ELVIS writes only to the selected PostgreSQL transaction
and standard output.

For an absent opening, the trusted Python control plane verifies Ed25519, the
pinned policy and key, current policy `revoked` state and approval lifetime at
the database fence time. PostgreSQL independently enforces physical identity,
terminal catalog, target emptiness, nonce/conflict rules, document relations,
atomicity and the final approval-expiry check. PostgreSQL does not verify
Ed25519 or query an external live revocation service. Exact durable replay is
resolved before the current-authority check.

## Result and exit contract

When standard output is writable, every non-help invocation emits exactly one
compact JSON line. It never emits a raw public key, signature, nonce, path,
libpq service, database/role name, `system_identifier`, deployment incarnation,
DSN, password or exception.

| `result` | Exit | `side_effect_state` | Meaning |
|---|---:|---|---|
| `CREATED` | 0 | `COMMITTED` | Exact evidence, nonce, opening and physical receipt committed atomically |
| `REPLAYED` | 0 | `COMMITTED` | Byte-exact committed receipt returned before current authority evaluation |
| `BLOCKED` | 10 | `NONE` | Current authority or physical-target admission did not permit an absent mutation |
| `CONFLICT` | 20 | `NONE` | Nonce namespace or target opening exists with different exact content |
| `COMMIT_UNKNOWN` | 21 | `UNKNOWN` | Commit acknowledgement was lost and exact durable state is not yet resolved |
| `INVALID_INPUT` | 2 | `NONE` | Invocation, local file or typed document is malformed or unsafe |
| `INTERNAL_ERROR` | 70 | `UNKNOWN` | Unexpected failure; database contact or commit cannot safely be ruled out |

The only relayed reason codes are the frozen uppercase codes from the
application/adapter contract: `FRESH_OPENING_CREATED`,
`EXACT_DURABLE_REPLAY`, the PR2 `BLOCKED_*` codes,
`TARGET_ADMISSION_BLOCKED`, `FRESH_OPENING_NONCE_CONFLICT`,
`FRESH_OPENING_TARGET_CONFLICT` and `FRESH_OPENING_COMMIT_UNKNOWN`. Any other
adapter text becomes a generic `INTERNAL_ERROR`; raw errors are never relayed.

A committed receipt exposes only evidence/receipt SHA-256 values, migration
head and the unchanged authority state. `CREATED` means the economic opening is
durable; it does not mean ready, provisioned runtime, V1 retired, `PAUSED`,
`ACTIVE`, canary passed or production released.

## Exact replay and conflict rules

The registry is target-local over
`(trust_domain, signer_key_id, nonce)`. `logical_target` is inside the signed
intent digest. Under that namespace:

- the exact already-committed intent, approval, policy and opening returns the
  same durable receipt as `REPLAYED`;
- another logical target or any changed exact content returns `CONFLICT`;
- another trust domain or signer key is a separate namespace on that physical
  target; and
- another physical database has an independent registry and never proves
  global nonce uniqueness.

Replay comparison happens under the database fence before current approval
freshness or revocation is evaluated. Therefore an exact committed replay
remains resolvable after approval expiry or signer revocation, while an absent
or changed mutation remains blocked.

## `COMMIT_UNKNOWN` recovery

Preserve the complete command inputs and the one-line result. Do not edit the
intent, generate a nonce, change target pins, switch databases or retry with a
different approval. Re-run only the identical command against the same physical
target so the adapter can perform exact durable readback:

- committed exact bytes must resolve to `REPLAYED` with the same receipt hash;
- a proven absent transaction may proceed only if current authority still
  permits the original mutation; and
- conflicting durable bytes remain `CONFLICT` and require containment, not
  deletion or overwrite.

If exact readback cannot resolve the result, freeze the target, preserve its
database and logs, and escalate. Never `UPDATE` or `DELETE` the immutable
opening, nonce, evidence or receipt. Recovery is a forward correction or a new
reviewed fresh target, not an in-place rewrite.

If standard-output delivery fails after database contact, the process exits
`70` without attempting a second write. The opening may already be committed;
treat the result as unknown and re-run only the byte-exact command above to
obtain the durable `REPLAYED` receipt or another typed outcome.

## Verification

The pure CLI contract needs no PostgreSQL server:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.14 -m pytest -q -p no:cacheprovider \
  tests/test_v2_opening_apply.py
```

Run the adapter's separately gated PostgreSQL 15 integration suite before any
rehearsal. It must prove that the opening anchor cannot log in or execute the
opening functions, that a non-owner invocation fails before taking locks, and
that admin create/replay/conflict/revoked/expired/tamper/commit-unknown paths are
fail-closed. Unit tests use an injected provisioning factory and never contact
a database, network, Docker, Kali or exchange.
