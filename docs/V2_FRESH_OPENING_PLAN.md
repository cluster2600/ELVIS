# ELVIS V2 fresh-opening plan

> **Read-only source tool.** This procedure validates a proposed trajectory-B
> paper-account opening and its detached approval. It does not contact
> PostgreSQL, reserve a nonce, create or provision an account, change writer
> authority, or make a runtime `ACTIVE`.

## Scope

The planner is the PR2 boundary from the
[paper-production plan](architecture_migration/05-v2-production-plan.md). It
prepares one explicit fresh V2 paper opening with no V1 balance, P&L, order,
fill, fee, position, settlement, or journal continuity.

The planner can prove only that:

- the three local JSON documents have their exact version-1 shapes;
- the business intent is canonical and internally consistent;
- the prospective account opening is derived by the existing V2 opening
  codec;
- the detached Ed25519 signature matches the pinned public key and trust
  policy;
- the approval is valid at the one UTC evaluation time; and
- the resulting intent, approval, trust-policy and opening-payload digests are
  deterministic.

It cannot prove that a physical database is empty or admitted. It cannot know
whether a nonce has already been used in a target-local durable registry.
Those checks and the atomic `CREATED`/`REPLAYED`/`CONFLICT` transaction belong
to PR3.

## Availability

Run this command from a source checkout with Python 3.14:

```bash
python3.14 -m scripts.v2_opening_plan --help
```

The public `v2.0.0-alpha.2` operator image remains byte-for-byte historical and
does not contain this command. Do not expect `opening-plan` in that image's
four-command dispatcher. Packaging this source tool requires a separately
versioned future candidate.

## Input boundary

The command reads at most 64 KiB from each named path. Each path must resolve
directly to one regular file opened with `O_NOFOLLOW`; symlinks, FIFOs,
devices, duplicate JSON keys (including escaped equivalents), non-finite JSON
constants, invalid UTF-8, unknown fields, and oversized files are rejected.

The repository templates under [`docs/examples/v2`](examples/v2) are shape
guides only. Their `EXAMPLE_INVALID_*` sentinels and zeroed cryptographic values
are deliberately non-operational and must be replaced through the approved
business and trust workflow. A template is never an approval.

Three documents are kept separate:

1. **Fresh-opening intent** — the stable business fact: logical target,
   execution scope, account, positive owner generation, opening codec/version,
   one positive unreserved collateral amount, margin quantum, trajectory-B
   policy, operator and approver identities, UTC review window, trust domain,
   signer key ID and nonce. It deliberately excludes a PostgreSQL
   `system_identifier`, database incarnation and runtime candidate.
2. **Detached approval** — Ed25519 only, binding the exact canonical intent
   digest to one signature. PEM, JWK, JWT, embedded keys and alternate
   algorithms are not accepted.
3. **Trust policy** — the current public Ed25519 trust anchor, authorised
   approval identity, key status and maximum approval duration. It contains no
   private key.

The expected canonical trust-policy SHA-256 and expected raw public-key
SHA-256 are supplied independently on the command line. They must come from a
previously frozen, authenticated control-plane record. Computing either pin
from the same untrusted input during this invocation defeats the trust
boundary.

The raw Ed25519 signer key must come from a controlled key-generation ceremony,
and its SHA-256 fingerprint must be frozen and authenticated out of band. A
controlled approval ceremony is also required to freeze the intent, approval
window and detached signature before this planner is invoked.

The private signing key must never enter Git, CI, this repository, the operator
image, command arguments, receipts, screenshots, videos, or application logs.

## Prepare the files

Copy the deliberately blocked templates outside the checkout and replace every
sentinel from authenticated inputs:

```bash
install -d -m 0700 /absolute/private/path/elvis-opening-plan
install -m 0600 \
  docs/examples/v2/fresh-opening-intent-v1.template.json \
  /absolute/private/path/elvis-opening-plan/intent.json
install -m 0600 \
  docs/examples/v2/fresh-opening-approval-v1.template.json \
  /absolute/private/path/elvis-opening-plan/approval.json
install -m 0600 \
  docs/examples/v2/fresh-opening-trust-policy-v1.template.json \
  /absolute/private/path/elvis-opening-plan/trust-policy.json
```

Do not copy a V1 balance or the historical alpha.2 reconciliation hypothesis.
No environment variable or repository example may choose the collateral asset,
amount, quantum, policy, logical target, identity, expiry, key or nonce.

The opening amount and margin quantum are canonical fixed-point Decimal text.
They are positive, finite, contain no exponent or sign, and the amount must be
an exact multiple of the quantum. The opening has exactly one balance, for the
declared collateral asset, with reserved amount exactly zero.

Have the authorised external signer sign the canonical intent using Ed25519.
The ELVIS planner exposes no private-key or signing operation. Record only the
lowercase hexadecimal detached signature in the approval document.

## Run the read-only preparation

```bash
python3.14 -m scripts.v2_opening_plan \
  --intent /absolute/private/path/elvis-opening-plan/intent.json \
  --approval /absolute/private/path/elvis-opening-plan/approval.json \
  --trust-policy /absolute/private/path/elvis-opening-plan/trust-policy.json \
  --pinned-trust-policy-sha256 <authenticated-lowercase-sha256> \
  --pinned-signer-public-key-sha256 <authenticated-lowercase-sha256> \
  > /absolute/private/path/elvis-opening-plan/preparation.json
```

The command never reads `PGSERVICE`, a DSN, a secret environment variable, or
a default opening amount. It performs no socket, subprocess, SQL or filesystem
write. Redirection above is performed by the invoking shell, not by ELVIS.

## Result contract

Every non-help invocation emits exactly one compact, secret-free JSON object on
standard output. Errors never echo a path, raw key, detached signature, nonce,
input document or exception.

| `result` | Exit | Meaning |
|---|---:|---|
| `PREPARED` | 0 | Canonical bytes, signature, pins and freshness verified read-only |
| `BLOCKED` | 10 | A well-shaped plan lacks current authenticated authority |
| `INVALID_INPUT` | 2 | Invocation, file or document is malformed or unsafe |
| `INTERNAL_ERROR` | 70 | Unexpected local failure; no authority may be inferred |

`PREPARED` is not `CREATED`, `REPLAYED`, `COMMITTED`, readiness, provisioning
or activation evidence. Every result permanently reports:

```json
{
  "side_effect_state": "NONE",
  "database_contact": false,
  "nonce_registry_checked": false,
  "target_local_replay_authority": "UNAVAILABLE_UNTIL_PR3",
  "account_opening_authorized": false,
  "account_provisioning_authorized": false,
  "runtime_activation_authorized": false,
  "trading_authorized": false
}
```

The safe output includes only the canonical intent, approval, trust-policy,
signer-public-key and prospective opening-payload SHA-256 values plus stable
status/reason fields. It never includes raw key, signature, nonce, file path,
PostgreSQL identity or private infrastructure detail.

Stable `primary_reason_code` values distinguish missing authority or approval,
policy/trust-domain mismatch, unknown or revoked signer, approver or approval
binding mismatch, an approval that is not yet valid or has expired, and an
invalid signature. A malformed policy, raw key, signature encoding or weak/
small-order Ed25519 point is instead `INVALID_INPUT`; it is never downgraded to
an authority decision.

## Replay and recovery boundary

Running the planner twice over the same documents recomputes the same digests.
That is deterministic verification only; the second run must still say
`PREPARED`, never `REPLAYED`.

Do not react to a timeout or local error by generating a new nonce or editing a
signed intent. PR2 has no commit acknowledgement to recover because it performs
no transaction. PR3 will own the target-local registry over
`(trust_domain, signer_key_id, nonce)`, physical-target admission and exact
readback after an ambiguous commit.

## Verification

The focused Python 3.14 checks are:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.14 -m pytest -q -p no:cacheprovider \
  tests/test_fresh_opening.py \
  tests/test_v2_opening_plan.py \
  tests/test_paper_account_journal_codec.py
```

No PostgreSQL, Docker, Kali, runtime, exchange, network or private signing key
is required for this PR2 verification.
