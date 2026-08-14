<!-- /autoplan restore point: /Users/maxime/.gstack/projects/cluster2600-ELVIS/agent-v2-production-autoplan-restore-20260814-070533.md -->

# ELVIS V2 paper-production plan

> **Draft for review.** This plan starts from `v2.0.0-alpha.2`, which packages
> offline migration/operator tooling but does not compose an authoritative V2
> trading runtime. No successful test, receipt, installation, merge, or release
> changes runtime authority by itself.

## Objective

Deliver an installable, observable, reversible ELVIS V2 paper-trading runtime
that is the sole proven writer on its target database, then publish the code,
operator documentation, verification evidence, installation screenshots, and
a course-ready installation video through GitHub.

The production target in this programme is **paper trading only**. Exchange
credentials capable of real-money trading, executable live mode, and any
financial-capital decision remain outside this plan and require a separate
explicit approval.

## Starting point

The current release already supplies:

- Python 3.14-only source and CI contracts;
- typed signal, policy, risk, order, fill, position, and account boundaries;
- versioned PostgreSQL journal, account, fence, generation, and activation
  capabilities;
- an offline least-authority bootstrap;
- isolated PostgreSQL 15 rehearsals;
- stopped-source/fresh-target preflight;
- bounded seven-table legacy snapshot import; and
- a read-only, non-authoritative opening-candidate reconciliation review.

The compatibility paper process remains authoritative. The operator image does
not contain or start a V2 trading runtime. The root Compose file is legacy
evidence, not a V2 production installer.

The existing pure order reducer is useful 1B substrate: it already distinguishes
acknowledgement from fills and validates partial/multiple fills, cancellation
races, exact duplicates and overfill. The durable boundary is not 1B: the
current plan, owners and SQL manifest require one terminal ACK-plus-full-fill
batch, and a standalone fill append can advance position without account
settlement. Those paths remain historical/test primitives and are excluded from
the production dependency closure rather than weakened in place.

## Architecture decision — trajectory B approved

The product owner selected **B** on 2026-08-14: create a fresh V2 paper account
with an explicit opening, retain V1 as operationally read-only evidence, and
make V2 the only possible writer. The programme deliberately provides **no**
historical P&L, balance, position, event, or accounting continuity from V1.

The product owner then selected **1B**: paper production must include a real
asynchronous virtual-venue lifecycle. The release cannot reduce execution to an
immediate full fill. It must durably support acknowledgement, rejection,
cancellation, zero fill, partial and multiple fills, duplicate/out-of-order
venue messages, restart recovery, and deterministic reconciliation before it
may be called production.

The opening is a new V2 business fact, not a reconstruction of V1. Its
collateral asset, amount, quantum, account identity, approving operator, exact
intent, and policy digest must be supplied explicitly and bound to an immutable
receipt. No code, manifest, example, environment variable, or compatibility
balance may silently choose the amount. Until that intent is supplied and
approved, the account remains unopened and activation remains unavailable.

V1 data may be queried only through bounded, read-only comparison tooling. It
may help an operator understand decision differences, but it cannot seed the V2
opening, prove parity, regain writer authority, or serve as an automatic
rollback target.

## Non-negotiable invariants

1. Only one paper writer may be authoritative at a time.
2. Returning from `ACTIVE` to `LEGACY` is forbidden. Rollback passes through `PAUSED`, restores
   verified V2 durable state or redeploys a known-good V2 image, and never
   restores V1 writer access.
3. Missing, stale, ambiguous, unreadable, or unauthenticated evidence fails
   closed and prevents submission.
4. Runtime identities cannot create roles, migrate schema, grant privileges,
   perform DDL, or activate themselves.
5. Secrets never enter Git, release assets, command arguments, receipts,
   screenshots, videos, CI logs, or application logs.
6. The fresh-opening intent, approval, policy, and resulting account opening
   are durable and bound to exact digests. Imported or compatibility balances
   are not opening evidence.
7. Shadow execution is side-effect free: no order submission and no mutation
   of cooldowns, positions, accounts, feedback, or persistence.
8. V1 remains stopped and operationally read-only once its writer is revoked.
   It is evidence, not a rollback authority. “Immutable archive” is claimed
   only after independent integrity verification and WORM/retention controls
   exist.
9. Every destructive or authority-changing operator action requires an exact
   intent, explicit confirmation, a preflight, a receipt, and a documented
   recovery path.
10. Production means observed endpoint state, not a successful command or a
    generated artefact.
11. `PaperAccountState.ACTIVE` describes only an economically solvent account.
    It never grants runtime authority. Submission additionally requires
    `runtime_mode=ACTIVE`, `admission_valid=true`, `market_ready=true`, a clear
    kill switch, and `trading_authorized=true`, all derived from current
    durable evidence.
12. The logical opening intent binds stable business identity and policy, not a
    physical PostgreSQL `system_identifier`. A separate admission/provisioning
    receipt binds the exact target cluster, migration head, catalog and
    transition so a verified restore never rewrites the approved business
    intent.
13. `authority_transition_sequence` advances gaplessly only for a committed
    authority transition under lock. Rejected, stale or conflicting attempts
    use a separate audit-attempt ID and never mutate the authority tail.
14. Each activation epoch binds the exact executable candidate through the
    immutable pre-CAS activation-intent manifest and post-CAS activation receipt:
    source commit, immutable image, dependency lock, rendered configuration,
    model/features, strategy/risk policy, opening/provisioning, prior accepted
    authority receipt, rollback candidate, approval, activation ID and
    deployment/database/venue incarnations.
15. `SHADOW` is an offline evaluation mode with zero database authority. It is
    not a writer-authority state.
16. Command admission atomically commits the immutable execution candidate,
    an account-global worst-case collateral/fee hold, the order identity and
    one submit outbox message before any virtual-venue work can begin.
17. Venue causality is a gapless global sequence per execution scope,
    activation epoch and venue incarnation. A per-order ordinal validates each
    order history, but never overrides the global economic ordering. Delivery
    order is not causal order.
18. Venue-input causality is a separate gapless global sequence under the same
    scope, activation epoch and venue incarnation. Inputs include submit,
    cancel, immutable market-step and virtual-timer actions. Exactly one fenced
    venue source owner consumes `next_input` and atomically appends only the
    immediate deterministic source-event batch before advancing the input tail.
    A future partial fill requires a later durable input; future fills are never
    pre-appended when the submit command is consumed.
19. Every venue event advances lifecycle, finality, hold, application receipt
    and applied venue tail in one transaction. A confirmed fill additionally
    advances position, account journal and postings in that transaction. A
    rejection or terminal cancellation releases its hold there. No production
    path may append an event to only one projection.
20. Exact duplicate messages are no-ops. A reused identity/sequence with
    different canonical bytes, an overfill, invalid correlation or unsupported
    economic fact is quarantined, leaves the applied tail blocked and forces
    `trading_authorized=false`; it is never skipped as a dead letter.
21. A kill switch blocks new risk but cannot block ingestion, settlement,
    reconciliation or idempotent cancellation of already accepted work. A
    planned pause sets the kill switch, reaches venue finality with zero
    commands/events/gaps/holds/non-terminal orders, and only then commits the
    `ACTIVE -> PAUSED` transition.
22. `writer_fence` is the committed `authority_transition_sequence` of the
    active epoch. Every submission, venue-source and projector transaction
    locks and rechecks the authority row, generation, candidate and
    incarnations until commit. An authority transition takes the conflicting
    lock and therefore cannot pass an older in-flight writer.

## Success criteria

The programme is complete only when all of the following are true on one exact
release commit and immutable image digest:

- one explicit fresh-opening intent and approval are authenticated and durably
  linked to exactly one V2 paper account;
- provisioning is idempotent with exact `CREATED` or `REPLAYED` outcomes, and
  no V1 rows or inferred history enter the V2 journal or balances;
- migration, schema owner, activation, runtime owner, readiness, and read-only
  identities are distinct and admitted through restrictive SCRAM/HBA/network
  policy;
- the runtime has no DDL, GRANT, role-management, migration, activation, or
  secret-fallback path;
- liveness, admission validity, market readiness and trading authorisation are
  distinct; `trading_authorized` is true only when runtime mode is `ACTIVE`,
  exact candidate/admission evidence is current, the market/venue worker is
  ready, and the kill switch is clear;
- startup, readiness, health, and order submission fail closed on database,
  catalog, role, system-identifier, generation, fence, or provenance drift;
- V2 decisions pass reviewed specifications, golden vectors and deterministic
  virtual-venue scenarios. An optional V1 comparison is informational only,
  non-blocking and side-effect free;
- stale compatibility writers are proved unable to commit after the fence;
- the fresh economic opening commits while authority remains `LEGACY/0/S0`;
  proven V1 retirement commits `LEGACY/0/S0 -> PAUSED/0/S1`; initial
  activation commits `PAUSED/0/S1 -> ACTIVE/1/S2`; V2-only
  `ACTIVE/N/S -> PAUSED/N/(S+1) -> ACTIVE/(N+1)/(S+2)` transitions are
  rehearsed, including
  lost acknowledgement, ambiguous commit, restart, and DB failure;
- `runtime_generation` changes only on activation, while the separate
  monotonic `authority_transition_sequence` records only the committed
  `LEGACY -> PAUSED` V1-retirement transition, pauses and activations; rejected
  attempts remain immutable audit evidence without consuming a transition
  sequence;
- the asynchronous virtual venue proves accepted/rejected/cancelled/zero-fill/
  partial-fill/multi-fill flows, duplicate and out-of-order delivery,
  acknowledgement loss, crash/restart and exact replay without duplicate
  economic effects;
- every accepted order reserves its bounded worst-case exposure before venue
  evaluation; fills convert that hold into exact margin/fees and terminal
  finality releases the remainder, so effective available collateral never
  becomes negative;
- the global venue source/applied tails converge, every per-order history is
  contiguous, the venue-input source/applied tails converge, and quarantine,
  delivery gaps, unknown finality, pending leases, inbox/outbox backlog,
  non-terminal orders and active holds are all zero at every pause,
  restore-admission and soak exit;
- backup and restore are rehearsed from an independently verified backup;
- a time-bounded paper soak meets explicit reliability, correctness,
  observability, and resource thresholds;
- an operator records the final paper cut-over approval;
- the Kali production host runs the pinned V2 runtime with healthy dependencies
  and no regression to its existing unrelated stacks;
- public GitHub PRs, CI, release assets, checksums, SBOMs, attestations,
  installation docs, screenshots, and the course video are verified at their
  destination; and
- the V1 runtime and credential are archived or removed only after the V2 soak,
  while V1 database evidence remains operationally read-only and is called
  immutable only after independent integrity and WORM/retention proof.

Acceptance labels are exact: `RELEASED` means public immutable artefacts exist;
`DEPLOYED_PAUSED/PRODUCTION_READY` means the target is installed and admitted
but cannot submit; only an observed, separately approved canary in
`ACTIVE_PAPER_PRODUCTION` is production.

## Bounded 1B product contract

The first production venue is intentionally smaller than a general exchange:

- deterministic `MARKET` orders for reviewed linear instruments only;
- one collateral asset and fees only in that asset;
- one non-terminal order per position key, with account-global capacity holds;
- immutable stored market snapshots and execution terms; no mutable lookup
  after candidate admission and no RNG;
- PostgreSQL 15 as the durable single-account venue-input/event queue; notification
  may wake workers but is never authority; and
- no live venue adapter, live credential, limit-order book maintenance,
  cross-collateral fee conversion or high-frequency claim.

Distinct identities must never be overloaded:

- the immutable release candidate binds executable, configuration-schema/
  template/default and policy bytes;
- a deployment-target record binds the physical database/catalog/roles and
  exact rendered secret-free configuration digest;
- the activation-record family contains one immutable pre-CAS activation-intent
  manifest and one immutable post-CAS activation receipt. The manifest binds
  the candidate, target, opening/provisioning, prior accepted authority receipt,
  rollback candidate, approval/kill-clear intent, expected generation/
  transition/fence, deployment/database/venue incarnations and expected
  zero-work high-water marks. The receipt binds the committed result back to
  that exact manifest without rewriting it; and
- each gate receipt records mutable before/after authority, readiness, tails,
  holds, leases and quarantine without rewriting any earlier record.

| Identity | Meaning | Advances on |
|---|---|---|
| `authority_transition_sequence` | Gapless accepted authority history and active writer fence | committed V1 retirement, pause or activation |
| `runtime_generation` | Positive activation epoch | accepted `PAUSED -> ACTIVE` only |
| `runtime_candidate_sha256` | Exact source/image/config/model/policy closure | never mutated |
| `activation_intent_manifest_sha256` | Exact pre-CAS activation proposal and approval closure | never mutated |
| `activation_receipt_sha256` | Exact post-CAS committed result/readback bound to one intent manifest | each accepted activation; never mutated |
| `deployment_incarnation_id` | Approved logical deployment | new approved deployment |
| `venue_incarnation_id` | Durable deterministic venue bound to activation | new activation after complete drain |
| `worker_instance_id` / `lease_epoch` | Ephemeral worker and its fencing claim | process start / claim or reclaim |
| `venue_input_sequence` | Global submit/cancel/market-step/timer causal order | each canonical durable input |
| `venue_input_applied` | Contiguous prefix consumed by the sole fenced venue source owner | atomic immediate source-event batch append |
| `venue_event_sequence` | Global causal economic order | each canonical source event |
| `venue_event_applied` | Contiguous prefix committed by the projector | each atomic event application |
| `venue_order_sequence` | One order's validation ordinal | each canonical source event for that order |
| `position_version` | Applied lifecycle/position prefix | each applied lifecycle event |
| `account_version` | Economic settlement prefix | each applied fill |
| `risk_sequence` | Pending-order capacity history | hold placement, reduction or release |

All descendant commands/events/holds/settlements bind the activation candidate,
generation, writer fence, deployment and venue incarnations. Gapless counters
are allocated under a locked row in the inserting transaction; PostgreSQL
sequences are not used where rollback would create a causal hole.

### Canonical activation record pair

`activation-intent-manifest-v1` exists before the authority CAS and has exactly
these canonical fields:

```text
schema_version
activation_id
idempotency_key
execution_scope
account_key
release_candidate_sha256
deployment_target_sha256
rendered_config_sha256
opening_receipt_sha256
provisioning_receipt_sha256
prior_authority_transition_receipt_sha256
paused_backup_restore_receipt_sha256
rollback_candidate_manifest_sha256
activation_approval_sha256
kill_clear_intent_sha256
trust_policy_sha256
approver_key_id
approval_expires_at
approval_nonce
expected_current_runtime_mode
expected_current_runtime_generation
expected_current_authority_transition_sequence
expected_current_writer_fence
expected_next_runtime_mode
expected_next_runtime_generation
expected_next_authority_transition_sequence
expected_next_writer_fence
deployment_incarnation_id
database_incarnation_id
venue_incarnation_id
expected_zero_work_high_watermarks
```

For the initial `PAUSED/0/S1 -> ACTIVE/1/S2` transition,
`prior_authority_transition_receipt_sha256` is the exact accepted
`RETIRE_V1_WRITER` receipt. For every later activation it is the exact accepted
pause receipt. `paused_backup_restore_receipt_sha256` is the exact receipt for
the quiescent `PAUSED` backup and isolated restore that qualifies the database
incarnation. It may be canonical JSON `null` only for an explicitly disposable
activation whose gate makes no recovery or production claim; it is mandatory
for the initial production activation and every reactivation following a
quiescent backup. `rollback_candidate_manifest_sha256` always names a complete
safe-redeploy manifest. For the first V2 activation it names the same release
candidate, independently proved redeployable against the exact migration head,
codecs, economic versions and rendered configuration; later activations may
instead name an older, fully qualified compatible V2 candidate. It is never a
floating tag, a bare image digest, or an absent hypothetical predecessor.
`kill_clear_intent_sha256` is mandatory and is signed as part of the same
next-epoch approval; clearing kill is not a standalone mutation.
The approver signs canonical `activation-intent-core-v1` bytes containing every
field above except `activation_approval_sha256`; the completed manifest then
adds the detached approval-envelope digest. The approval never signs a digest
that recursively contains itself.

The CAS locks authority and asynchronous-work tails, verifies every expected
field, clears kill, commits the next activation and persists the canonical
`activation-receipt-v1` bytes in the same transaction. Only after commit may the
operator emit those already durable bytes, which contain exactly:

```text
schema_version
receipt_id
activation_intent_manifest_sha256
result
side_effect_state
committed_activation_id
committed_authority_transition_id
previous_runtime_mode
previous_runtime_generation
previous_authority_transition_sequence
previous_writer_fence
committed_runtime_mode
committed_runtime_generation
committed_authority_transition_sequence
committed_writer_fence
deployment_incarnation_id
database_incarnation_id
venue_incarnation_id
committed_at
durable_readback_sha256
```

An activation receipt has `result=COMMITTED` and must name the manifest digest
and an exact durable authority/state readback that excludes the receipt's own
digest. Its canonical core, committed timestamp, readback digest and receipt
digest are therefore recoverable even if the process dies after commit but
before acknowledgement. Exact replay returns that byte-identical durable
receipt with operator result `REPLAYED`; it creates no second receipt.
`CONFLICT` and unresolved `COMMIT_UNKNOWN` are operator/audit-attempt results and
create no activation receipt. A timeout never manufactures one:
`COMMIT_UNKNOWN` remains non-authorising until readback resolves to the one
committed receipt or proves that no mutation occurred.

### Canonical operator contract

`operator-status-v1` has one ordered representation in human output and JSON:

```text
RESULT
TRADING_AUTHORIZED
RUNTIME_MODE
LIVENESS
ADMISSION_VALID
MARKET_READY
VENUE_READY
ECONOMIC_ACCOUNT_STATE
RUNTIME_GENERATION
AUTHORITY_TRANSITION_SEQUENCE
WRITER_FENCE
KILL_SWITCH
PRIMARY_REASON_CODE
BLOCKERS
SIDE_EFFECT_STATE
SAFE_ACTION_CODE
RECEIPT_SHA256
```

`SIDE_EFFECT_STATE` is exactly `NONE`, `COMMITTED`, or `UNKNOWN`.
`operator-result-v1` uses `VERIFIED`, `CREATED`, `REPLAYED`, `COMMITTED`,
`BLOCKED`, `CONFLICT`, `COMMIT_UNKNOWN`, `INVALID_INPUT`, or `INTERNAL_ERROR`.
Exit codes are respectively `0` for the first four, `10`, `20`, `21`, `2`, and
`70`. A `status` observation exits zero when the observation itself is valid;
`TRADING_AUTHORIZED=false` remains explicit data, never a transport error.
`SAFE_ACTION_CODE` is informative and is never run automatically.

Primary blockers use the first matching code in this precedence order:
`SHUTTING_DOWN`, `OBSERVABILITY_UNTRUSTWORTHY`, `DATABASE_UNAVAILABLE`,
`ADMISSION_UNREADABLE`, `ADMISSION_STALE`, `CANDIDATE_MISMATCH`,
`STALE_RUNTIME_GENERATION`, `STALE_WRITER_FENCE`, `KILL_SWITCH_UNREADABLE`,
`VENUE_WORKER_NOT_READY`, `VENUE_EVENT_CONFLICT`, `VENUE_EVENT_GAP`,
`VENUE_FINALITY_UNKNOWN`,
`ORDER_RECONCILIATION_REQUIRED`,
`SETTLEMENT_RECONCILIATION_REQUIRED`, `VENUE_OUTBOX_BACKLOG`,
`RESOURCE_BUDGET_EXCEEDED`, `RUNTIME_PAUSED`, `APPROVAL_REQUIRED`,
`RUNTIME_LEGACY`, `KILL_SWITCH_SET`,
`MARKET_DATA_INVALID`, `MARKET_DATA_STALE`, `RESERVATION_INSUFFICIENT`, then
`READY`. `BLOCKERS` retains every
simultaneous code in that same stable order.

`SAFE_ACTION_CODE` is exactly one of `NONE`, `REESTABLISH_OBSERVABILITY`,
`RESTORE_DATABASE_CONNECTIVITY`, `REVALIDATE_ADMISSION`,
`REVALIDATE_KILL_SWITCH`, `SET_KILL_AND_DRAIN`, `RECONCILE_VENUE`,
`RECONCILE_ORDER`, `RECONCILE_SETTLEMENT`, `WAIT_FOR_VALID_MARKET`,
`REDUCE_OR_REJECT_ORDER`, `STOP_PRODUCERS_AND_RECOVER`,
`SUPPLY_SIGNED_APPROVAL`, `RETIRE_V1_WRITER`, or
`SUPPLY_ACTIVATION_APPROVAL`. The mapping is deterministic:

- shutdown and `READY` use `NONE`;
- observability and database blockers use their corresponding
  re-establish/restore codes;
- admission/candidate/generation/fence blockers use `REVALIDATE_ADMISSION`;
- kill unreadable uses `REVALIDATE_KILL_SWITCH`; `RUNTIME_PAUSED` uses
  `SUPPLY_ACTIVATION_APPROVAL`, which means a fresh next-epoch approval with its
  coupled kill-clear intent; only an `ACTIVE` kill-set runtime uses
  `SET_KILL_AND_DRAIN`;
- venue conflict/gap/finality/backlog/worker blockers use `RECONCILE_VENUE`;
- order and settlement blockers use their matching reconciliation code;
- market blockers use `WAIT_FOR_VALID_MARKET`;
- reservation and resource blockers use `REDUCE_OR_REJECT_ORDER` and
  `STOP_PRODUCERS_AND_RECOVER` respectively;
- approval and legacy blockers use `SUPPLY_SIGNED_APPROVAL` and
  `RETIRE_V1_WRITER` respectively.

After higher-priority authority, venue-worker, causal, reconciliation and
resource blockers, `RUNTIME_PAUSED` deliberately precedes `KILL_SWITCH_SET`: a
drained paused runtime is expected to remain kill-set. Its next safe action is
never another kill/drain loop or a standalone clear; it is to supply the exact
fresh activation approval whose pre-CAS manifest couples kill-clear to the next
activation transaction. A broken venue worker therefore remains visible even
while kill is set, because it can prevent accepted work from reaching finality.

Every mutating command follows exact preflight, canonical intent summary,
explicit confirmation, idempotency key, commit/readback resolution and receipt:

| Command | Identity | Required state | Mutation / replay rule |
|---|---|---|---|
| `verify` / `preflight` | audit/readiness | any | read-only; digest-bound report |
| `bootstrap` | migration/schema owner | fresh target | forward catalog only; exact rerun is `REPLAYED` |
| `opening-plan` | offline verifier | unopened | read-only signed intent |
| `opening-apply` | opening owner | `LEGACY/0/S0`, empty | account + provenance + nonce atomically; exact replay only |
| `retire-v1` | offline activation | `LEGACY/0/S0` | distinct signed `RETIRE_V1_WRITER`; CAS to `PAUSED/0/S1` |
| `stage` / `status` | runtime/readiness | non-authoritative | no authority mutation |
| `activate` | offline activation | `PAUSED`, zero async work | exact pre-CAS activation-intent manifest; atomic kill-clear/next-generation/fence CAS; post-CAS activation receipt |
| `kill-set` | independent operator | `ACTIVE` | idempotent set; runtime can set but never clear |
| `drain` | drain/reconciliation | `ACTIVE`, kill set | no new risk; ingest/settle/cancel/reconcile only |
| `pause` | offline activation | `ACTIVE`, kill set, drained | same-generation next-fence CAS |
| `kill-clear` | offline activation | admitted `PAUSED` | intent only; no standalone mutation; digest is coupled to the fresh activation approval and consumed by `activate` |
| `code-rollback` | operator + activation | drained `PAUSED` | exact rollback-candidate manifest, then next activation |
| `restore-isolated` | backup/restore | isolated target | preserves backed-up mode/generation/transition/fence exactly; records a new DB incarnation only; creates no authority transition and remains non-authoritative |

`account_owner_generation` in opening commands is the account-stream owner
generation and is not `runtime_generation`. A rollback candidate is not merely
an image digest: its manifest proves compatibility with the exact migration
head, codecs, venue/account economic versions and rendered configuration.

## Delivery strategy

Ship small, ordered pull requests. Each PR is dormant or fail-closed until the
next gate and cannot silently grant authority.

### PR 1 — Production plan and evidence contract

- Freeze the architecture, operator journey, threat model, failure registry,
  test matrix, deployment topology, observability thresholds, rollback policy,
  and media provenance contract.
- Add executable documentation/link/config guards.
- Publish no new runtime or authority.

### PR 2 — Fresh-opening intent and approval contract

- Add a canonical, secret-free intent format that requires execution scope,
  account key, positive `owner_generation`, opening-codec/version, collateral
  asset, exact Decimal amount, margin quantum, opening policy, logical target,
  operator identity, approval identity, expiry, trust domain/key ID and nonce.
- Require a detached Ed25519 signature verified against a pinned public key and
  trust-policy digest; a command-line confirmation is not authentication.
- Derive the prospective opening payload and hashes without touching a database.
- Require exactly one positive, unreserved collateral balance and reject missing,
  zero/negative, unquantized, stale, internally duplicated, ambiguous, or
  example/default values. Keep this slice read-only and non-authorising.
- Do not claim global or cross-database nonce replay protection in this offline
  slice. Report target-local replay authority as unavailable until the next
  transaction checks that target's durable nonce registry under lock.
- Bind a stable logical target name and account identity, but deliberately omit
  the physical PostgreSQL `system_identifier`; physical target admission is a
  later receipt so backup restoration does not invalidate or rewrite the
  business approval.
- The private signing key never enters Git, CI, the operator image, receipts,
  screenshots or video. An absent, revoked or unconfigured trust anchor yields
  `BLOCKED_AUTHORITY_UNCONFIGURED`, never an approved intent.

### PR 3 — Forward schema migration and idempotent fresh V2 opening

- Add a forward-only migration that durably binds the approved fresh-opening
  intent, signature, policy and account opening without changing existing
  opening-codec golden hashes. Add a locked **target-local** nonce registry with
  uniqueness on `(trust_domain, signer_key_id, nonce)`. Within one physical
  PostgreSQL target, the signed intent digest stores the logical target, so
  reusing a nonce for different content or another logical target under that
  same trust/key namespace is `CONFLICT`. A different trust domain or signer
  key is a separate namespace within that registry and may use the same raw
  nonce. This registry makes no cross-database uniqueness claim: another
  isolated physical target can consume the same raw nonce only under its own
  provisioning receipt and remains non-authoritative. Sole-writer admission
  and activation later select exactly one physical database incarnation.
- Provision exactly one opening with `CREATED` or exact `REPLAYED` outcomes in
  one target transaction; any same-identity/different-content attempt is
  `CONFLICT`.
- Revalidate target database, system identifier, terminal catalog, roles,
  generation, empty V2 state, and approval under protective locks before write.
- Emit a physical admission/provisioning receipt that binds the approved intent
  digest to the observed PostgreSQL `system_identifier`, migration head,
  terminal catalog fingerprint and deployment incarnation.
- Insert no imported V1 row, trade, order, fill, fee, P&L, position, or balance.
  A wrong opening requires a fresh target or forward correction, never
  `UPDATE`/`DELETE` of immutable opening facts.
- Leave writer authority exactly `LEGACY/0/S0` and every activation-authority
  flag false. A commit-unknown result is resolved by exact nonce/intent/opening
  readback even after approval expiry; expiry or revocation blocks only absent
  or new mutation.

### PR 4 — B-only authority ledger and V1-retirement capability

- Add an append-only, gapless authority-transition ledger distinct from
  activation epochs and an immutable audit-attempt ledger for rejected calls.
- Keep `paper_runtime_generations` as positive activation epochs. Permit the
  first activation from `PAUSED/0`, forbid direct `LEGACY -> ACTIVE`, and never
  append a generation for pause.
- Add dormant, least-authority operations that can drain V1 sessions, revoke
  its writer credential, retain evidence digests, prove reconnect/commit
  failure, and then CAS `LEGACY/0/S0 -> PAUSED/0/S1`. The operation requires a
  distinct signed `RETIRE_V1_WRITER` approval; an opening or activation
  approval cannot substitute for it.
- Prove the operations only on disposable targets in this PR. Do not stop the
  real V1 writer or perform the real transition until qualification phase R2
  has a frozen candidate, backup, target identity and explicit cut-over
  approval.
- No later transition to `LEGACY` exists, and `SHADOW` is removed from database
  writer-authority semantics.
- Introduce a deployment incarnation plus the canonical pre-CAS activation-
  intent-manifest and post-CAS activation-receipt schemas above. The initial
  manifest binds the accepted V1-retirement receipt; every later manifest binds
  the accepted pause receipt. Production activations also bind the applicable
  quiescent `PAUSED` backup/restore receipt. Each binds the release-candidate,
  deployment-target and `rollback_candidate_manifest_sha256` digests to exact
  executable authority without a self-referential post-commit record. Persist
  the canonical post-CAS receipt core in the same transaction as the authority
  ledger and singleton so lost acknowledgement can return identical bytes.
- Define `writer_fence` as the committed transition sequence of the active
  epoch. Economic writers hold/recheck the authority lock through commit;
  retirement, pause and activation take the conflicting lock.

### PR 5 — Asynchronous virtual-venue domain, fixtures, and capacity model

- Retire the terminal `PaperSubmissionPlan` from the production design and add
  versioned submit/cancel commands; accepted/rejected/no-fill/fill/
  cancellation-accepted/cancellation-rejected events; immutable execution
  terms; market snapshots; and pending-order capacity holds.
- Support zero fill, partial and multiple fills, fees, slippage and price
  improvement without exceeding instruction quantity or spending unreserved
  collateral.
- Bound the first production venue to MARKET orders, linear instruments, fees
  in the single collateral asset, and one non-terminal order per position key.
  Account-global holds still serialize scarce collateral across positions.
- Define the complete order state machine and illegal transitions, including a
  queued cancel before acknowledgement, cancel/fill races and causally earlier
  fills delivered after cancellation.
- Use a versioned `venue-scenario-v1` fixture: immutable economic source events
  and expected-state digests are separate from duplicate/delay/drop/reorder
  delivery faults. Use virtual time and deterministic IDs; no wall clock,
  network venue or global RNG enters the pure model.
- Keep strategy/risk acceptance separate from venue outcome; a venue rejection
  is not a policy rejection.

### PR 6 — Durable venue inbox/outbox, worker, replay, and accounting

- Persist submission commands in a transactional outbox and venue messages in
  an idempotent inbox before applying them to order, position and account
  streams.
- In the command-admission transaction, lock the account risk stream, replay
  economic balance plus active holds, and atomically store the candidate,
  worst-case price/slippage/fee-capped hold, async order and submit outbox.
- Enforce separate gapless global venue-input and causal-event sequences per
  execution scope, activation epoch and venue incarnation, plus a per-order
  ordinal, immutable payload hash and terminal-finality watermark. Venue inputs
  are submit/cancel commands or immutable market-step/virtual-timer actions.
  Exactly one fenced source owner consumes `next_input` and atomically appends
  only its immediate deterministic event batch; later partial fills require
  later durable inputs. Buffer a delivery gap and apply only
  `last_applied + 1`; never treat arrival or worker scheduling as economic
  order.
- Apply every venue event in one fenced transaction across lifecycle,
  finality, capacity hold, application receipt and applied tail. Fills also
  update position and progressive account journal/postings; rejection and
  terminal cancellation release capacity. Commit cancel intent and its outbox
  exactly once. Keep the old complete-batch tables read-only and make the
  standalone fill append API unreachable from the production closure.
- Resolve worker crash, process restart and lost acknowledgement at every
  durable boundary by exact readback. Add fenced leases, bounded backpressure
  and retry; conflicting economic messages are quarantined and block the stream
  rather than being skipped through a dead-letter queue.
- Add forward migrations, least-privilege ACLs, retention/resource bounds and
  full recovery for awaiting-acceptance, open, partial and cancel-pending
  orders. Readiness covers every new relation, function, role, tail and lease.

### PR 7 — Dedicated V2 runtime, admission, kill switch, and health

- Add a new V2-only entrypoint and dependency-closure guard with no import or
  runtime reachability to `main.py`, the legacy executor or V1 DB helper.
- Compose the typed signal, policy, risk, asynchronous venue, journal, position
  and account services; remove runtime DDL, GRANT, migrations, credential
  defaults and balance fallbacks.
- Verify source commit, immutable image, dependency lock, rendered config,
  model/features, strategy/risk policy, opening, activation ID and incarnation
  on activation and before every economic write.
- Separate `liveness`, `admission_valid`, `market_ready` and
  `trading_authorized`; no API/thread/market client/worker starts before staged
  admission, and no submission occurs until exact `ACTIVE` authority is
  observed.
- Consume the durable kill switch in the final fenced submission/venue path,
  prove no new-risk bypass, and keep inbox ingestion, fill settlement,
  reconciliation and idempotent cancel/drain alive while it is set.
- Publish one canonical human/JSON status envelope with separately derived
  `liveness`, `admission_valid`, `market_ready`, `venue_ready`,
  `trading_authorized`, economic account state, runtime mode, activation epoch,
  transition sequence, kill-switch state, reason code, side effects, next safe
  action and receipt digest.
- Expose authenticated operator commands for preflight, pause, activation and
  code rollback; no public runtime endpoint changes authority.

### PR 8 — Deployment, observability, code rollback, backup, and DR

- Add restrictive V2-only Compose/deployment assets using external secrets,
  immutable digests, unique resources and one authoritative dashboard.
- Normal code rollback preserves the current database: set the kill switch
  while still `ACTIVE`, keep fenced ingest/settlement/cancel/reconcile alive,
  drain to finality and zero async work, then commit `PAUSED`, deploy the
  approved old image, and activate the next epoch with a new activation
  intent manifest and its resulting activation receipt.
- For the first release, qualify a same-candidate safe-redeploy manifest before
  activation so `rollback_candidate_manifest_sha256` is real even when no older
  V2 production candidate exists. Later releases may qualify an older V2
  candidate, but compatibility with the current durable schema and economic
  codecs must be reproved.
- Treat database restore as separate disaster recovery: restore into isolation,
  create a new incarnation, rotate credentials, validate historical maxima and
  require a new approval before any activation.
- Add encrypted PostgreSQL backup, off-host retention/WORM, integrity,
  recovery-time and recovery-point contracts and rehearse them on a disposable
  target. The first release accepts only quiescent DR backups taken after
  kill-set drain/finality and the pause CAS; continuous recovery of in-flight
  async work is not claimed.
- Add secret-free structured logs, metrics, alerts, and one authoritative V2
  dashboard for economic state, authority mode, readiness dimensions,
  generation, transition sequence, opening, venue backlog/gaps, kill switch,
  submission and recovery.
- Keep metric labels bounded and expose source/applied venue tails, outbox/inbox
  backlog, gap age, unresolved-order age, settlement lag, holds, quarantine,
  stale-writer rejections and duplicate economic effects (which must remain
  exactly zero).
- Keep backup identities unable to submit, activate, or mutate runtime state.

### PR 9 — Golden specifications and informational V1 comparison

- Run V2 against reviewed golden signal/risk/venue/accounting vectors and
  side-effect-free shadow fixtures.
- Keep any stopped-V1 decision comparison optional, read-only, informational
  and non-blocking. Never compare balances, P&L, positions or journals as if the
  fresh account had continuity.
- Publish discrepancies as bounded evidence without granting V1 oracle status.

### Release phase R1 — Build the immutable production candidate once

- Build runtime and operator images from minimal closures for each supported
  architecture with SBOM, provenance, vulnerability gates, checksums and
  attestations.
- Assign immutable candidate digests before E2E. Never rebuild between soak,
  promotion and public destination verification.
- Stage a private/restricted bundle and install it anonymously only after it is
  promoted; G0 verifies source/build inputs before public publication rather
  than requiring a future public asset.

### Qualification phase R2 — End-to-end rehearsal, Kali deployment, and paper soak

- Execute fresh install, bootstrap, signed opening, retirement, async venue,
  activation, restart, failure, rollback and DR rehearsals on disposable PG15.
- Only after the immutable candidate and disposable rehearsals pass, repeat G1,
  G2 and G3 on the new production target and execute the separately approved
  real opening while authority remains `LEGACY/0/S0`. Then take that target's
  encrypted opening-only backup, restore it in isolation and verify the exact
  `LEGACY/0/S0` mode, generation, transition sequence and fence without creating
  an authority transition.
- After that opening-backup receipt exists, stop new V1 work without yet
  revoking its identity, drain its in-flight work and sessions, and complete
  the independent V1 archive integrity/WORM/isolated-restore/read-only-
  comparator proof. Only then obtain the distinct signed
  `RETIRE_V1_WRITER` approval naming those exact receipts plus the V1 process,
  login and session identities. Permanently revoke the writer and commit the
  one-way retirement to `PAUSED/0/S1`. If approval is withheld before that
  commit, V2 remains unauthorised; this pre-retirement abort is not a future
  V1 rollback right.
- Immediately take a new quiescent backup at `PAUSED/0/S1`, restore it in
  isolation and verify that the restored authority remains `PAUSED/0/S1` with
  only a new database-incarnation receipt. The earlier opening-only `LEGACY`
  backup is now expired as a rollback point. Repeat the install and paused
  admission on Kali without joining or restarting unrelated stacks. After
  exact retirement and `PAUSED` backup-receipt readback, obtain the separate
  activation approval and its coupled kill-clear intent. Its pre-CAS manifest
  must name both receipts, the opening, release-candidate and deployment-target
  digests, database/venue incarnations and
  `rollback_candidate_manifest_sha256`. Activate and execute the exact
  production paper canary once, then retain its causal/economic receipt and the
  post-CAS activation receipt.
- Run deterministic fault injection and reviewed 24h/72h paper soak; publish
  exact commit/image/database fingerprints and sanitized evidence.

### Promotion phase R3 — Bit-for-bit GitHub promotion and destination verification

- Promote the exact candidate digests without rebuild or moving an existing
  tag. Publish PRs, release assets, checksums, SBOMs, provenance, attestations
  and sanitized gate evidence.
- Download anonymously, verify every destination asset, and reinstall from the
  public bundle only on fresh disposable twins to prove exact equality to the
  soaked candidate. The production target is readback-only during destination
  verification: observe its digest, activation-intent manifest, activation
  receipt and status without reinstalling, restarting or mutating it.
- Do not call the release production unless the observed Kali endpoint is
  `ACTIVE_PAPER_PRODUCTION`; otherwise label it
  `DEPLOYED_PAUSED/PRODUCTION_READY`.

### Media phase R4 — Separate course capture and media publication

- Capture the verified clean-install journey at 16:9 with readable terminal
  text and secret-safe fixtures.
- Produce versioned screenshots, narration, captions/transcript, media
  provenance, storyboard, and a Studio preview.
- Render the final course video only after explicit preview approval.
- Publish media only after release/destination verification; a media failure
  cannot retroactively invalidate safe runtime bytes, but must block course
  publication until corrected.

## End-to-end verification matrix

| Layer | Required evidence |
|---|---|
| Pure/domain | property, boundary, serialization, version, fresh-opening hash/signature, Decimal quantum, venue fixture compilation, hold conservation, delivery permutations, and malformed-input tests |
| Application | async venue-input/event state-machine properties, global/per-order ordering, finality, idempotency, lost acknowledgement, cancellation/fill races, concurrent owner, and fail-closed port tests |
| PostgreSQL | PG15 integration, role/ACL/HBA, system ID, candidate/hold/outbox/inbox/projector transactions, global sequences, leases, quarantine, triggers, generations/transitions, crash/restart, backup/restore |
| Runtime | staged admission before effects, no legacy reachability/DDL/fallbacks, DB loss, stale candidate/generation, kill switch, blocked submission, health dimensions |
| Venue | ack/reject/cancel, zero/partial/multi-fill, duplicate/out-of-order/gap/conflict, restart/replay, deterministic time and resource bounds |
| Shadow | zero effects, deterministic golden inputs, optional informational V1 discrepancy, and explicit no-accounting-parity assertions |
| Deployment | clean-directory install, immutable digest, amd64/arm64, TLS, external secrets, resource limits, upgrade/rollback |
| Security | secret scan, dependency/image CVE gates, SBOM/provenance/attestation, least privilege, API authentication |
| Observability | metrics/log reason parity, dashboard/alert contract, recovery visibility, no sensitive data |
| Operations | preflight, backup, restore, fresh opening, pause, activation, V2-only rollback, stale V1 writer, ambiguous commit, runbook dry-run |
| Host | existing-service inventory unchanged, isolated project/network/volumes, restart counts, health, cleanup |
| Documentation | commands executed from clean bundle, links/configs validated, claims tied to immutable evidence |
| Media | clean capture, readable frames, redaction review, transcript, provenance, preview approval, final render QA |

## Production topology selected for the first paper release

```text
operator workstation / CI
  |-- signed intents, receipts, attestations, release assets
  |-- offline migration identity
  `-- offline activation identity

Kali production host
  |-- elvis-v2-runtime (paper only; unprivileged; immutable image)
  |     |-- readiness identity (read-only evidence)
  |     `-- submission identity (candidate + hold + command outbox only)
  |-- deterministic virtual-venue worker
  |     `-- sole fenced venue-source identity (next input + immediate batch only)
  |-- ordered event projector
  |     `-- projector identity (every lifecycle/finality/hold/tail atomically;
  |                            fills also position/account)
  |-- dedicated ELVIS PostgreSQL 15 project
  |     |-- TLS, restrictive HBA, distinct roles, named persistent volume
  |     `-- encrypted backup copied to an independently verified off-host store
  |-- metrics/log collector and one V2 dashboard
  `-- existing unrelated stacks (separate projects/networks; untouched)

operationally read-only V1 archive / disposable rehearsal clones
  |-- read-only decision comparison only
  `-- never attached to the active V2 runtime network and never granted write
```

The first paper-production release deliberately accepts one Kali host as the
runtime/database availability domain because that is the selected installation
target. This does not make the host its own recovery domain: activation requires
a successful restore from an encrypted off-host backup onto disposable
infrastructure. ELVIS uses unique Compose projects, networks, volumes, ports and
resource limits and never joins the networks of existing Kali stacks.

## Failure and rescue policy

- Before activation: discard and rebuild only the disposable target when the
  opening approval, immutable opening, catalog, or rehearsal evidence drifts.
- During preparation, pause, or activation: an unknown commit is reconciled by
  the exact audit/transition ID, intent digest, control mode, transition
  sequence, generation, release-candidate digest, activation-intent-manifest
  digest and durable activation-receipt readback; never retry a non-idempotent
  transition blindly.
- While active: new-risk admission stops on database or authority uncertainty.
  The process remains observable but not ready; already accepted work continues
  only through the fenced ingest/settle/cancel/reconcile paths.
- Code rollback: set the kill switch, drain to proven venue finality and zero
  queues/gaps/holds/non-terminal orders, then enter `PAUSED`; prove one owner,
  preserve the current database, redeploy a known-good V2 image, and activate a
  new epoch. Never restore the V1 writer or run two writers to “see which
  works.”
- Disaster recovery: restore only into isolation, declare a new incarnation,
  rotate credentials, validate historical maxima and obtain a fresh activation
  approval. Never describe a data-rewinding restore as ordinary rollback.
- Corrupt or wrongly opened durable state: preserve evidence, rebuild from the
  approved fresh-opening intent or restore a verified V2 backup, and do not
  patch immutable opening facts in place.
- Host regression: stop only the uniquely named V2 project, preserve database
  evidence and logs, and verify all unrelated stack identities and restart
  counts. Never use global prune, broad `down -v`, or Docker daemon restart as
  an ELVIS recovery step.

## Documentation and media deliverables

- production architecture and security model;
- install, upgrade, backup, restore, cut-over, pause, rollback, and incident
  runbooks;
- versioned bug ledger with reproduction, severity, fix, regression test, and
  release linkage;
- evidence manifest tying commit, images, SBOMs, attestations, configs,
  database identifiers, test runs, screenshots, and video;
- screenshots of release verification, clean host preflight, install, image
  digest, configuration, healthy runtime, fail-closed behavior, metrics,
  dashboard, backup/restore, and cleanup/rollback;
- course storyboard, narration script, transcript/captions, media provenance,
  approved Studio preview, final MP4, and checksums; and
- GitHub PRs and a production paper release with verified public assets.

Screenshots and video must use dedicated non-secret fixtures. Private host
addresses, usernames, tokens, passwords, `.pgpass`, certificates, source data,
and unrelated service details are cropped or redacted before publication.

## Approval boundaries

The user has authorised code changes, tests, paper-production deployment,
documentation, screenshots, course preparation, GitHub PRs, merges, and release
publication within this programme. The following still require a separate
explicit confirmation at the moment of action:

- any real-money/live exchange capability or credential;
- the exact fresh-opening collateral asset, Decimal amount, policy and logical
  target, unless already present in a separately authenticated business-
  opening approval;
- the irreversible `RETIRE_V1_WRITER` intent, signed separately from opening
  and activation and binding the exact V1 process, database/login, drained
  sessions, archive-evidence digest, target V2 identity and preflight receipt;
- any production cut-over whose physical target account/database and compatible
  `rollback_candidate_manifest_sha256` are not bound by a separate activation/
  cut-over approval;
- the final runtime authority transition to `ACTIVE`; and
- the final course-video render after reviewing the Studio preview.

## Immediate next step

Merge this planning/evidence slice only after its focused guards and final
independent review are green. Then implement PR 2, the pure signed fresh-opening
intent/approval contract, before any schema or runtime mutation. Continue in the
ordered PR sequence below; every later gate uses independent review and
frozen-byte verification.

## GSTACK REVIEW REPORT

### Settled product decisions

- The user selected trajectory B and then 1B: a fresh, paper-only V2 account
  with a true asynchronous deterministic virtual venue.
- V1 remains the current writer only until one separately signed,
  evidence-bound retirement. It is never a rollback writer afterward.
- The first production venue is bounded to one collateral asset, MARKET/linear
  products, collateral-asset fees and one non-terminal order per position key.
- Final course rendering remains a separate approval gate after the exact
  runtime release has passed G16.

### Review synthesis

- CEO review accepted B/1B but rejected the initial plan's overloaded
  generation, opening, target and rollback identities. The plan now separates
  authority transition sequence, runtime generation, release candidate,
  deployment target, activation manifest and mutable gate snapshots.
- Engineering review replaced the terminal full-fill batch assumption with
  gapless durable venue inputs (`submit`, `cancel`, market-step and virtual
  timer), immediate-only source-event batches, an independent causal-event
  sequence, account-global holds, terminal finality and one atomic all-event
  projector.
- Operator/DevEx review fixed staged admission, stable reason/action codes,
  `ECONOMIC_ACCOUNT_STATE`, quiescent DR, target-specific backup, one-way V1
  retirement, per-epoch activation approvals, public-twin reinstallation and
  production readback-only verification.
- Design/media review isolated the historical alpha.2 operator-preview course,
  removed unlicensed imagery, made the capture manifest the sole command
  source, retained raw/public provenance boundaries, and froze measurable G17
  reproducibility, accessibility, loudness, redaction and Studio-approval gates.
- Independent async-domain, storage and operations audits supplied the
  cancel/fill races, finality watermark, lease/fence, recovery, backpressure,
  alert, resource and soak requirements incorporated in G7-G15.

### Ordered implementation and release sequence

1. PR 1: merge only this B/1B plan, failure register, G0-G17 matrix, historical
   media quarantine and executable documentation guards.
2. PR 2: pure canonical opening intent/approval/signature/nonce/replay contract.
3. PR 3: forward-only atomic opening/provenance storage and physical-target
   admission, leaving authority `LEGACY/0/S0`.
4. PR 4: dormant gapless authority ledger, writer fence and disposable-only V1
   retirement/activation capability.
5. PR 5: pure asynchronous venue domain, deterministic inputs/events, holds and
   `venue-scenario-v1` compiler.
6. PR 6: PostgreSQL venue-input/outbox/event/inbox/projector/finality/lease/
   quarantine storage with least-authority roles and crash tests.
7. PR 7: dedicated V2 runtime composition, staged admission, kill switch,
   truthful health and no legacy production reachability.
8. PR 8: deployment, observability, encrypted backup/restore, incident and
   rollback automation.
9. PR 9: golden paper specifications and optional stopped-V1 informational
   comparison.
10. R1-R4: build once; qualify through G0-G15; promote/verify through G16; then
    capture, approve and publish media through G17.

### Remaining risk and verdict

All 32 register entries remain `OPEN`; the plan records work, not closure. No
runtime, database, authority, Kali or GitHub mutation is authorised by this
document. Review verdict: **GO only for the PR 1 planning/evidence contract;
NO-GO for runtime production, V1 retirement, `ACTIVE`, release promotion or
course publication until their named gates and approvals pass on frozen bytes.**
