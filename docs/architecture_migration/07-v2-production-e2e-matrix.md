# ELVIS V2 paper-production end-to-end gate matrix

> **Evidence contract, not a deployment receipt.** Every gate below is
> initially `NOT_RUN`. A green unit-test suite, a successful command, an
> installed image, or a generated screenshot is not evidence that a gate has
> passed. Production in this document means **paper trading only**. Live
> exchange credentials, real-money order paths, and the final transition of a
> production account to `ACTIVE` remain outside unattended execution and
> require separate explicit approval.

## Purpose and scope

This matrix defines the ordered, reproducible acceptance programme for one
ELVIS V2 paper-production candidate. It covers supply-chain verification,
installation, PostgreSQL authority, data provenance, deterministic execution,
failure behaviour, recovery, operations, observability, packaging,
documentation, and course media.

The matrix implements the approved **trajectory B only**: one fresh V2 paper
account with an explicit signed opening, with V1 retained as operationally
read-only evidence. It may be called immutable only after separate integrity
and WORM/retention proof. It makes no claim of historical balance, P&L, position,
event, journal or accounting continuity. V1 may be an offline decision
comparator, but it cannot seed the opening, regain writer authority or serve as
a rollback target.

An item marked `WAIVED_NA` still requires a signed waiver in the gate receipt
that explains why it cannot affect trajectory B. It must never conceal missing
evidence, and it never unlocks a dependent gate unless that gate explicitly
declares the item inapplicable.

## Frozen candidate and evidence rules

### Five records that must not be conflated

Before G0 begins, assign a unique `run_id` and freeze only the immutable release
candidate. Target and observed state do not belong in that record:

| Record | Frozen contents | Created or refreshed |
|---|---|---|
| Release candidate | repository/full commit, clean-tree assertion, candidate reference, bundle/image/platform digests, SBOM/provenance/attestation/scan digests, Python/application/dependency lock, migrations, configuration schema/templates/defaults, model/features, strategy/risk/venue policy, fixture and threshold-manifest digests | before G0; never mutated |
| Deployment target | logical target, PostgreSQL version/system identifier, database incarnation, migration/catalog/role/ACL fingerprints, approved network and secret-reference identities, exact rendered secret-free configuration digest | G2 for rehearsal; repeated on the new production target in G14 |
| Activation manifest (pre-CAS intent) | activation/idempotency IDs, scope/account, release-candidate, deployment-target, rendered-config, opening-receipt, provisioning-receipt, `rollback_candidate_manifest_sha256`, `paused_backup_restore_receipt_sha256` and prior accepted retirement-or-pause receipt digests; activation approval, coupled kill-clear intent and trust/key/expiry/nonce; expected current and next mode/generation/transition/fence; deployment/database/venue incarnations; expected zero-work high-water marks | before each activation CAS; immutable thereafter |
| Activation receipt (post-CAS observation) | activation-manifest digest, committed activation/transition IDs, result/side-effect state, previous and committed mode/generation/transition/fence, deployment/database/venue incarnations, commit time and exact durable-readback digest | canonical bytes persisted in the accepted CAS transaction; emitted after commit or returned by exact commit-unknown readback; immutable thereafter |
| Gate state snapshot | economic state, runtime mode, readiness dimensions, trading authorisation, generation/transition/fence, venue-input/event source/applied tails, holds, leases, quarantine and non-terminal orders before and after one gate | every gate receipt; observed and expected to change |

Changing executable bytes, dependencies, image layers, migrations, admission
rules, configuration templates/defaults, model/policy or fixture inputs creates
a new release candidate and invalidates G0 and every dependent gate. Changing
the physical target creates a new deployment-target record and repeats its
dependent admission gates; it does not rewrite the approved business opening
intent. Documentation embedded in the bundle creates a new candidate; an
external evidence-only redaction change requires G16 and, when media changes,
G17 to be repeated, with a recorded proof that no runtime gate was invalidated.

### Gate receipt

Each run retains a private evidence directory named
`<run_id>/<gate_id>/`. Its machine-readable receipt records:

- receipt schema/version and ID; gate ID; trajectory (`B`); status exactly
  `NOT_RUN`, `RUNNING`, `BLOCKED`, `PASS`, `FAIL`, or `WAIVED_NA`; start/end
  UTC; reviewer/signer identity; trust-policy digest; signature; and previous
  receipt digest;
- the immutable release-candidate digest, applicable deployment-target,
  activation-manifest and activation-receipt digests, before/after gate-state
  snapshots, and
  previous-gate receipt digests;
- test fixture identifiers, exact commands or API intents, exit status, and
  redacted stdout/stderr digests;
- database system identifier, migration head, account state, generation, and
  fence before and after the scenario where applicable;
- all evidence filenames and SHA-256 digests;
- every injected fault, expected observation, actual observation, and recovery
  action;
- public-redaction status and the digest mapping from each private original to
  its public derivative; and
- an explicit decision. Partial execution is `FAIL`, never `PASS`; only `PASS`
  unlocks an ordinary dependent gate.

Secrets, DSNs, private keys, certificates, `.env` contents, `.pgpass`, tokens,
private host addresses, raw production data, and unrelated stack details must
not enter receipts, logs, screenshots, videos, or public evidence. The private
evidence store may reference a secret by stable identifier and digest, but may
not copy its value.

### Universal hard failures

Any one of the following fails the current gate and blocks all later gates:

- a real-money or live-exchange submission, or presence of a credential able
  to make one;
- more than one authoritative paper writer;
- a submission while readiness, generation, fence, provenance, kill switch,
  database state, or market-data validity is unknown or false;
- runtime DDL, role management, migration, `GRANT`, activation, or secret
  fallback;
- a secret or private infrastructure detail in retained/public evidence;
- an unexplained state mutation, duplicate economic event, missing committed
  event, post-fence stale-writer commit, negative effective collateral,
  skipped causal gap, conflicting economic message outside quarantine, or
  fill without exactly one account settlement; or
- a material change to an unrelated Kali stack caused by the ELVIS exercise.

## Gate dependency graph

```text
G0 -> G1 -> G2 -> G3 -> G4 -> G5 -> G6 -> G7 -> G8 -> G9 -> G10
                                                        |
                                                        v
                         G13 <- G12 <- G11 <-------------+
                           |       |
                           v       |
                          G14 <----+
                           |
                           v
                          G15 -> G16 -> G17
```

Before G11, the same candidate must produce a narrowly scoped
`G12_INSTRUMENTATION_SMOKE` receipt proving metrics/log/alert transport exists;
it grants no truth claim. G12 accepts instrumentation truth only after G11
faults. G13 may be developed earlier, but its
restore proof must use the exact candidate that passed G12. G14 must not make
the production account `ACTIVE` without the separate explicit approval; all
state transitions can first be rehearsed on disposable infrastructure.

G1–G13 run on a dedicated rehearsal cluster and rehearsal account. Their
immutable journal history is never cleaned, copied into, or relabelled as the
fresh production account. G14 provisions and admits a separate production
target from its own G1/G2/G3-equivalent receipts before any cut-over. Only the
immutable release-candidate record is shared between those lineages.

| Gate | Acceptance boundary |
|---|---|
| G0–G4 | Exact candidate, isolated target, trust boundary and fresh opening |
| G5–G6 | Truthful staged admission and market input |
| G7–G10 | Async venue economics, concurrency, kill/drain and crash recovery |
| G11–G13 | Faults, operator truth and independent restore |
| G14–G15 | One-way cut-over, V2-only recovery and 24h/72h soak |
| G16 | Exact-byte public runtime release and clean-install destination proof |
| G17 | Separate screenshot/course production and destination proof |

## G0 — Supply chain and immutable candidate

**Preconditions**

- A pre-G0 candidate-build receipt already records the independent build and
  reproducibility comparison, its declared reproducibility boundary, and the
  immutable digests selected for the release-candidate record. The checkout is
  clean and the unique pre-promotion candidate reference resolves to the
  recorded full commit.
- CI is associated with that exact commit and uses only approved hosted
  runners and pinned workflow actions.
- No paper or live credential is present in the build or verification
  environment.

**Exact scenario**

1. Fetch the already frozen source archive, staged bundle, checksum file, SBOM,
   provenance, attestations, image manifest, and per-platform images from the
   restricted CI candidate store into a new empty directory. Independently
   confirm every digest in the pre-G0 candidate-build receipt; never build,
   select or replace candidate bytes inside G0.
2. Verify the candidate-ref-to-commit relationship, every checksum, signer,
   workflow identity, source digest, subject digest, and immutable image
   digest without trusting filenames or mutable tags.
3. Inspect the archive for traversal, absolute paths, unsafe links, devices,
   unexpected ownership/modes, and files not declared in the manifest.
4. Run source, dependency, container, license, and secret scans against the
   frozen bytes. Resolve the image manifest to each claimed platform and run
   `version` and Python-version probes by digest.
5. Verify the pre-G0 reproducibility comparison and its declared boundary,
   expected non-deterministic metadata and two independently produced subject
   digests. G0 performs no rebuild and never replaces the selected candidate.
6. Attempt verification after changing one byte of each signed/checksummed
   artefact. Every tampered subject must be rejected.

**Evidence to retain**

- Candidate-build receipt, candidate-store references, metadata, resolved
  ref/commit, manifest/platform digests, checksum output, attestation
  verification output, SBOM and scan reports, archive inventory, tamper-test
  output, and version/Python output.
- CI run URL and immutable job/workflow references.

**Pass / fail**

- **PASS:** every subject is bound to the frozen commit and intended workflow;
  all archive entries are safe; all claimed platforms resolve; Python is 3.14;
  there are zero detected secrets, zero critical vulnerabilities, and zero
  unwaived high vulnerabilities; tampering is rejected; no mutable image tag
  is used by deployment assets.
- **FAIL:** any subject is missing, unverifiable, mutable, unexpectedly signed,
  unsafe, built from different source, or exceeds the frozen scan policy.

**Rollback / containment**

- Quarantine the candidate and prevent promotion. Preserve the failed bytes
  and report privately; do not overwrite or silently replace release assets.
  Produce a new candidate identity after correction.

**Dependencies:** none.

**Applicability:** trajectory B only.

## G1 — Clean install and host isolation

**Preconditions**

- G0 passed.
- A supported, newly provisioned disposable host exists; Kali rehearsal also
  has a signed baseline inventory of unrelated containers, networks, volumes,
  health states, restart counts, and resource headroom.
- The installer uses an unprivileged operator account and a unique Compose
  project name. G1 uses non-secret placeholder references only; no functional
  database or venue credential exists until the controlled G2 identity setup.

**Exact scenario**

1. Start from an empty install directory and verify that no ELVIS project,
   container, network, volume, service account, or credential already exists.
2. Retrieve the G0 bundle from the restricted candidate store with a
   read-only, authenticated identity and verify it by digest using the staged
   installation guide. Anonymous public download is tested only after exact
   byte promotion in G16.
3. Check file ownership and modes, render the deployment configuration with
   secrets suppressed, and prove that every executable image is pinned by
   digest.
4. Pull the exact platform image, run `help`, `version`, and Python probes, then
   start the stack in its non-authoritative/pre-admission state.
5. Verify isolation: unique names, dedicated networks/volumes, no unexpected
   host mounts, no Docker socket, read-only root filesystem where declared,
   dropped capabilities, `no-new-privileges`, resource limits, and only the
   documented loopback or internal ports.
6. Stop and recreate the project once. Then remove only the uniquely named
   disposable project and compare the host inventory with the baseline.
7. Repeat the install on Kali without restarting Docker or joining, stopping,
   or modifying an unrelated stack.

**Evidence to retain**

- Host/platform facts, pre/post inventory hashes, directory inventory and
  modes, sanitized rendered configuration, image inspection, help/version
  output, service/network/volume inventory, security settings, resource
  measurements, and cleanup comparison.

**Pass / fail**

- **PASS:** a clean reader can reproduce the install; only frozen artefacts are
  used; the runtime remains non-authoritative before admission; cleanup is
  scoped; all unrelated Kali identities, health states, and restart counts are
  unchanged.
- **FAIL:** undocumented manual repair is needed, a mutable tag/default secret
  is used, a port or privilege is broader than documented, the runtime becomes
  authoritative, or unrelated host state changes.

**Rollback / containment**

- Stop only the unique ELVIS project. Preserve logs and volumes needed for
  diagnosis; never run a global prune, broad `down -v`, or Docker-daemon
  restart. Restore only host files proven to have been created by this run.

**Dependencies:** G0.

**Applicability:** trajectory B only.

## G2 — PostgreSQL 15 TLS, HBA, roles, ownership, and ACLs

**Preconditions**

- G1 passed on the disposable environment.
- A fresh PostgreSQL 15 cluster has a recorded system identifier and externally
  supplied CA/server/client material. The role/ACL matrix and allowed network
  paths are frozen before bootstrap.
- Separate migration, schema-owner, bootstrap, activation, runtime-owner,
  submission, venue-source, projector, drain/reconciliation, readiness,
  backup, and audit/read-only identities are available.

**Exact scenario**

1. Bootstrap over verified TLS using only the offline privileged identities;
   apply forward migrations, ownership, grants, default privileges, HBA, and
   network policy.
2. Re-run bootstrap to prove idempotency and compare catalog fingerprints.
3. For every identity, attempt every allowed operation and a representative
   denied operation: connect, schema/table/sequence read/write, function
   execution, DDL, role creation/change, grant, migration, activation, backup,
   and access to another account. Submission, venue-source, projector and drain
   identities receive only narrow capability functions; direct table DML is
   denied.
4. Attempt plaintext, wrong-CA, wrong-hostname, expired/revoked-certificate,
   password-only, wrong-database, wrong-source-network, and undeclared-role
   connections. Inspect HBA logs without exposing credentials.
5. Run runtime startup as the runtime owner and prove that it cannot create or
   repair database objects, alter grants, activate itself, read secrets, or
   fall back to a shared/default credential.
6. Rotate the runtime credential, terminate old sessions, revoke the old
   identity, and prove that the stale credential cannot reconnect or commit.

**Evidence to retain**

- Redacted PostgreSQL version/system identifier, TLS negotiation facts,
  certificate identifiers and expiries, HBA fingerprint, role membership,
  ownership/ACL/default-ACL fingerprints, migration head, connection test
  matrix, catalog idempotency diff, and credential-rotation/stale-session
  results.

**Pass / fail**

- **PASS:** only declared encrypted paths connect; every identity has exactly
  its frozen privileges; runtime/readiness identities cannot perform DDL,
  grants, migration, role management, activation, or cross-account writes;
  bootstrap is idempotent; old credentials and sessions are ineffective.
- **FAIL:** any connection or operation is broader than the matrix, runtime can
  self-repair authority, TLS can be bypassed, or old credentials remain usable.

**Rollback / containment**

- Revoke runtime access, stop ELVIS, preserve catalog evidence, and destroy
  only the disposable cluster. For an adopted target, do not reverse grants
  blindly: restore the pre-change catalog/backup in isolation and reconcile
  before any further attempt.

**Dependencies:** G0 and G1.

**Applicability:** trajectory B only, on a fresh dedicated cluster. The V1
archive remains operationally read-only and outside runtime ownership and
networking; it is called immutable only after separate integrity and
WORM/retention proof.

## G3 — Fresh opening intent, approval, and durable preparation

**Preconditions**

- G2 passed.
- The target is the dedicated rehearsal cluster/account; it is not the future
  production account and its immutable test history will never be promoted.
- The logical target, execution scope, account key, positive owner generation,
  opening codec/version, exact positive Decimal collateral, margin quantum,
  opening policy, review window, nonce, trust domain/key, and authorised
  approvers are frozen.
- All candidate evidence is read-only; no opening or activation write occurs
  during evidence collection.

**Exact scenario**

1. Build canonical business-intent bytes with stable ordering. Bind logical
   target, account identity, owner generation, codec/version, exactly one
   positive unreserved collateral balance, quantum, policy, expiry, trust
   domain/key and nonce. Deliberately exclude the runtime candidate and the
   physical PostgreSQL `system_identifier` from this business approval.
2. Verify the detached signature against the pinned trust-policy digest. Prove
   absent, unknown, revoked, expired and altered approvals block any absent or
   new mutation. An exact already-committed replay remains read-only resolvable
   after expiry/revocation through durable readback.
3. Recompute the prospective opening payload with the real V2 opening codec.
   Reject zero/negative/unquantized amounts, additional assets, reserved
   balances, duplicates, non-canonical Decimals, examples, defaults and any
   balance inferred from V1.
4. Optionally bind the stopped V1 archive digest as operationally read-only
   evidence. Exercise the offline comparator and prove its result is
   informational, non-blocking and unable to modify the opening, target or any
   authoritative receipt.
5. Under the migration/owner lock, re-read the exact target, terminal catalog,
   roles and empty V2 state. Atomically register **target-local** uniqueness over
   `(trust_domain, signer_key_id, nonce)`, with the logical target held inside
   the signed intent digest, provision the account once and record the
   physical provisioning receipt while leaving writer authority exactly
   `LEGACY/0/S0`. Repeat the exact digest concurrently and after approval
   expiry to prove read-only `REPLAYED`; within that physical target, submit the
   same nonce with different content or logical target under the same trust/key
   namespace to prove `CONFLICT` without mutation. Prove that another trust
   domain or signer key is a distinct namespace in that registry rather than a
   false conflict. On a second isolated physical database, prove explicitly
   that the target-local registry makes no cross-database uniqueness claim:
   reuse may obtain a separate provisioning receipt, both targets remain
   `LEGACY/0/S0`, and sole-writer admission/activation can select only one
   database incarnation.
6. Drop the commit acknowledgement at every durable boundary and resolve only
   by re-reading the exact nonce, intent, opening hash and physical target
   identity; never generate a new nonce or repeat blindly.

**Evidence to retain**

- Canonical intent/policy/trust digests, approval identity, freshness,
  recomputation output, V1 evidence digest, nonce-registry result, target
  system identifier, migration/catalog fingerprint, incarnation,
  provisioning/opening receipt and every negative/tamper result. Store only
  sanitized summaries publicly.

**Pass / fail**

- **PASS:** exactly one independently approved fresh opening exists per
  physical target, exact concurrent, lost-ack and post-expiry replays return
  the same target receipt, a target-local nonce conflict is rejected, the
  cross-database scope limitation is explicit, every tested target remains
  `LEGACY/0/S0`, and no V1-derived fact entered an account.
- **FAIL:** ambiguity, missing/stale evidence, digest mismatch, a conflicting
  opening, unauthorised review, post-hoc policy change, or any invented causal
  fact.

**Rollback / containment**

- Before the opening transaction, reject and collect better evidence. After a
  wrong or ambiguous opening, freeze the target, preserve the receipt, and
  rebuild/restore a new target from authenticated inputs; never patch account
  history with `UPDATE` or `DELETE`.

**Dependencies:** G0–G2.

**Applicability:** trajectory B only.

## G4 — Fresh-state proof, post-opening replay, and offline comparison

**Preconditions**

- G3 passed.
- Replay schemas, ordering keys, versioned economic rules, resource bounds,
  checkpoint format and expected coverage are frozen.
- The fresh target is disposable or backed up; the optional V1 comparator input
  is immutable and read-only.

**Exact scenario**

1. Prove the account contains only the approved opening and no legacy order,
   fill, fee, P&L, position, journal, balance or reset event.
2. Replay only deliberately created post-opening V2 fixtures in bounded,
   deterministic order. Run the optional frozen V1 decision comparator in a
   separate read-only process and prove zero V2 state side effects.
3. Interrupt before a checkpoint, after a batch commit, and after commit but
   before acknowledgement. Resume from each state with the same intent.
4. Run the post-opening fixture replay twice against clean targets and once
   against the already completed target. Compare canonical state, sequence,
   balance, position, journal, coverage, and checkpoint digests.
5. Inject duplicate, missing-key, malformed, out-of-order, oversized,
   unsupported-version, and digest-changed fixtures. Verify deterministic
   rejection and bounded memory/time behaviour.

**Evidence to retain**

- Input/output manifests, batch/checkpoint receipts, resource/time profiles,
  interruption/restart traces, rejection reasons, coverage report,
  sequence/account/position/journal digests, offline-comparator side-effect
  proof, and clean-target versus idempotent-rerun comparisons.

**Pass / fail**

- **PASS:** repeated runs yield byte-equivalent canonical results; interrupted
  work resumes without duplicates or omissions; resources remain within the
  frozen bounds; rejection is complete and deterministic.
- **PASS additional:** zero legacy economic events appear in the fresh V2
  account and the offline comparator has zero side effects.
- **FAIL:** inferred history, non-determinism, unbounded work, a missing or
  duplicate economic event, or mutable raw evidence. A V1 decision difference
  is retained as information and never fails this gate by itself.

**Rollback / containment**

- Stop replay, preserve manifests/checkpoints, and rebuild the disposable
  target or restore the pre-replay backup. Never edit opening or journal facts
  in place.

**Dependencies:** G0–G3.

**Applicability:** trajectory B only; V1 is an optional offline decision
comparator, never an import source.

## G5 — Disposable authority rehearsal, startup admission, and fail-closed health

**Preconditions**

- G4 passed.
- A signed rehearsal-only `RETIRE_V1_WRITER` intent binds the rehearsal target
  and expected `LEGACY/0/S0 -> PAUSED/0/S1` transition. The activation approver,
  trust policy and unsigned manifest template are available, but the disposable
  activation approval cannot be signed until the retirement receipt exists.
  Nothing from this rehearsal is valid for the production target.
- Startup ordering; `liveness`, `admission_valid`, `market_ready`,
  `trading_authorized`; admission invariants; reason codes; and the rule that
  no side-effectful component starts before staged admission are frozen.
- Network observation can detect database, market, order, and background-worker
  activity.

**Exact scenario**

1. On the rehearsal target at `LEGACY/0/S0`, prove direct activation rejects.
   Block new work from the disposable compatibility-writer stand-in, drain its
   in-flight work and sessions to zero, prevent reconnect, stop its process,
   revoke its writer identity, prove it cannot reconnect or commit, and use
   only the rehearsal retirement approval to CAS
   `LEGACY/0/S0 -> PAUSED/0/S1`.
2. Stage the exact candidate in `PAUSED` and record configuration load, process
   liveness, database connection, admission, health publication, market/venue
   worker creation, market readiness, activation observation, and submission
   enablement. `trading_authorized` remains false even when every non-authority
   observation is healthy.
3. With zero commands, events, gaps, holds, leases, quarantine and non-terminal
   orders, create the rehearsal activation-intent core bound to the accepted
   retirement receipt, a canonical null PAUSED-backup field allowed only for
   this disposable no-DR activation, and the qualified same-candidate rollback
   manifest. Sign that exact core with the disposable activation approval, add
   its detached digest to the completed manifest, and commit
   `PAUSED/0/S1 -> ACTIVE/1/S2`. The CAS checks durable authority, candidate,
   opening, target, fence, kill and zero-work predicates only and emits the
   separate activation receipt. Re-evaluate transient `market_ready` after
   commit and at every command admission before `trading_authorized` may become
   true.
4. Repeat startup while changing one invariant at a time: database unavailable,
   invalid TLS, wrong database/system identifier, wrong role, migration drift,
   ownership/ACL drift, missing or altered provenance/opening/provisioning or
   activation-manifest digest,
   wrong account/mode, stale generation, invalid fence, unsupported economic
   version, missing secret reference, and unreadable durable state.
5. After a valid start, induce each durable-state failure and query all health
   dimensions plus the submission port before and after recovery.
6. Attempt submission while each individual authorisation predicate is false,
   through every public/internal application path. Verify one common
   fail-closed decision and stable reason code.
7. Restore valid state without restarting once, then with a restart, following
   the frozen recovery policy. Readiness must return only after a complete
   fresh admission; trading authorisation additionally requires current
   `ACTIVE` authority and market readiness.
8. Verify human and `--json` status conform byte-for-field to
   `operator-status-v1`: result, trading authorisation, runtime mode, liveness,
   admission, market and venue readiness, economic state, runtime generation,
   transition sequence/writer fence, kill switch, primary reason, ordered
   blockers, side-effect state, safe-action code and receipt digest. No colour
   alone conveys status.

**Evidence to retain**

- Signed rehearsal retirement intent/receipt and subsequently signed activation
  intent/receipt, writer-revocation
  proof, transition/generation/fence timeline, timestamped startup trace,
  network/process/thread observation, health payloads, reason-coded logs/
  metrics, admission fingerprints, direct submission test results, and before/
  fault/recovery timelines for every invariant.

**Pass / fail**

- **PASS:** the disposable transition path is legal and target-bound; no
  side-effectful market/venue/submission component starts before staged
  admission; every authority uncertainty makes `admission_valid` or
  `trading_authorized` false and blocks submission immediately; health names
  the exact reason; recovery requires full revalidation.
- **FAIL:** a false-green health result, background side effect before
  admission, fallback/default state, inconsistent block paths, or readiness
  recovery from cached evidence.

**Rollback / containment**

- If already active, set kill, keep recovery-only processing alive, drain to
  finality, then pause. Otherwise keep the candidate stopped or `PAUSED`.
  Revoke the runtime connection if necessary and restore only the rehearsal
  target from its verified fixture. Never override admission to continue
  testing.

**Dependencies:** G0–G4.

**Applicability:** trajectory B only.

## G6 — Market-data validity and feed faults

**Preconditions**

- G5 passed.
- A deterministic recorded paper-market corpus and frozen rules for freshness,
  ordering, deduplication, gaps, clock skew, reconnect, backfill, recovery
  window, and cooldown exist.
- Order egress is a paper-only test adapter that records every call.

**Exact scenario**

1. Replay the normal corpus at controlled time and prove deterministic signal,
   risk, and no-order/order intent results.
2. Inject, separately and at boundary values: stale ticks/candles, duplicates,
   out-of-order data, sequence gaps, malformed fields, `NaN`/infinity, wrong
   symbol, wrong interval, future timestamps, local clock jumps, partial
   snapshots, disconnects, reconnects, DNS failure, rate limiting, timeouts,
   and inconsistent backfill.
3. Inject disconnect and stale data immediately before and during decision
   evaluation. Attempt direct and scheduled submission while the feed is
   invalid.
4. Restore a contiguous valid stream. Verify that recovery occurs only after
   the configured validation window/backfill/cooldown and a fresh risk check,
   never merely because the socket reconnects.
5. Repeat the corpus and fault schedule to compare decisions, state, reason
   codes, and recovery timing.

**Evidence to retain**

- Corpus and virtual-clock digests, injected-event schedule, ordered input and
  decision digests, readiness/health timeline, submission-recorder output,
  reconnect/backfill evidence, state before/after each fault, and deterministic
  rerun comparison.

**Pass / fail**

- **PASS:** every invalid/unknown feed condition makes the runtime non-ready
  and prevents new submission; no cooldown/account/position state mutates from
  rejected data; recovery meets the frozen rule; identical inputs/faults yield
  identical outcomes.
- **FAIL:** stale or malformed data can submit, reconnect alone restores
  readiness, feed faults mutate durable economic state, or outcomes vary.

**Rollback / containment**

- Block new risk, disconnect market input, drain/reconcile accepted work, then
  pause the rehearsal runtime. Reset only the disposable fixture target before
  another run.

**Dependencies:** G0–G5.

**Applicability:** trajectory B only.

## G7 — Deterministic paper order, submission, fills, and accounting

**Preconditions**

- G6 passed.
- A versioned `venue-scenario-v1` contract freezes the symbol specification,
  opening, policy, immutable market snapshots, execution terms, price/quantity
  quantums, fee asset/rate, maximum adverse price/slippage, TIF, virtual clock
  and expected projection/account digests.
- The immutable economic venue-event ledger is separate from its delivery/fault
  schedule. Any seed additionally binds its algorithm and version; the first
  release uses no global RNG.
- Command admission can atomically commit the execution candidate, one
  account-global worst-case capacity hold, order identity and submit outbox.
  Submit, cancel, immutable market-step and virtual-timer actions enter one
  gapless `venue_input_sequence`. One fenced source owner consumes only
  `next_input` and atomically appends that input's immediate deterministic
  source-event batch before advancing the input-applied tail.
- The runtime is isolated from live endpoints and has no live credential.

**Exact scenario**

1. From the exact opening/state, evaluate one accepted signal and one rejected
   signal. Record the typed policy/risk decision and deterministic client order
   ID before submission.
2. Run independent clean scenarios for: venue rejection; ACK then zero fill and
   final cancellation; one full fill; multiple partial fills to full; partial
   fill then cancellation of the remainder; cancel rejection followed by fill;
   and the causal fill/cancel race in both valid orders.
3. For each admitted scenario, prove the hold reserves worst-case exposure plus
   capped collateral-asset fees before venue evaluation. Reduce it fill by fill
   and release only the terminal remainder after the declared final sequence is
   present and every earlier sequence has applied.
4. For submit and cancel separately, prove candidate/order/hold or cancellation
   state and outbox commit exactly once. Consume submit as `next_input` and
   prove it appends only its immediate ACK or rejection. Require a later durable
   market-step or virtual-timer input before every partial fill; no future fill
   may exist in the source log before its input is committed.
5. Race a cancel input against the market-step or virtual-timer input that makes
   a fill due, once with cancel first and once with due fill first. Cancel-first
   terminal finality cancels every remaining scheduled action and permits no
   later fill; fill-first applies that fill before cancellation releases only
   the remaining hold.
6. Retry each input before acknowledgement and after a lost acknowledgement;
   send exact duplicates, reverse delivery, delay/drop one prefix once, then
   converge. Reuse the same ID/sequence with different bytes and inject wrong
   correlation, unsupported fee/precision and overfill events.
7. Crash/restart between acceptance and fill, between partial fills, while
   cancellation is pending, and after each durable commit before its caller
   receives acknowledgement. Recover every non-terminal state.
8. Recompute lifecycle, venue-input/event global and per-order sequence, holds,
   finality, journal, position,
   cash, margin, fees and realised/unrealised P&L independently. Repeat each
   scenario from a clean target and compare all canonical digests.

**Evidence to retain**

- Fixture/economic-ledger/delivery-schedule/policy/specification digests; typed
  decisions and candidates; idempotency keys; input/outbox/inbox receipts;
  venue-input and event source/applied global tails; per-order ordinals/finality; hold/risk and
  account versions; quarantines; canonical journal rows; independent economic
  calculation; and clean-rerun digest comparison.

**Pass / fail**

- **PASS:** every scenario has exactly its declared inputs/orders/events/effects;
  duplicates are no-ops; delivery permutations converge only through the
  contiguous venue-input and event prefixes; no future fill is pre-appended;
  the two cancel/due-fill input orders produce their declared causal results;
  conflicts quarantine and block new
  risk without canonical economic mutation; effective collateral never
  becomes negative; every canonical event has exactly one lifecycle/finality/
  hold/application effect, and every fill has exactly one position/account
  effect; terminal holds and all clean-run queues are zero.
- **FAIL:** an order bypasses policy/readiness, IDs vary, a retry duplicates an
  order/fill, arrival order changes economics, a gap is skipped, finality is
  guessed, an accepted fill is rejected by local accounting, state is lost
  after restart, accounting differs, or any call reaches a live endpoint.

**Rollback / containment**

- Set the kill switch, retain the raw causal stream, and drain/reconcile on the
  disposable target. If a conflict cannot be resolved, keep the applied tail
  blocked and rebuild the fixture target from its pre-scenario snapshot. Never
  delete, rewrite, reorder or skip an economic event to make the test pass.

**Dependencies:** G0–G6.

**Applicability:** trajectory B only.

## G8 — Concurrency, fencing, and stale generation

**Preconditions**

- G7 passed.
- Two separately identified runtime instances, submission/venue/projector
  workers, database sessions, lease epochs, a frozen candidate/generation/
  fence, controllable transaction barriers, and submission recording exist.
- The operator can pause both instances without using a runtime credential.

**Exact scenario**

1. Start instance R1 at current generation and admit it. Start R2 with the same
   generation and race ownership/admission; only the state-machine policy may
   select an owner.
2. Race two identical submissions and two different orders whose combined
   worst-case holds exceed available collateral. Place barriers before/after
   authority read, risk-stream lock, candidate/hold/outbox insert and commit.
3. Race two claimants for the sole ordered venue-source lease and two
   projectors on the same and adjacent global sequences. Only one source owner
   may consume `next_input` and append its immediate event batch. Race claimants
   on submit, cancel, market-step and virtual-timer inputs. Expire/reclaim a
   lease, then let the former worker attempt its commit. Deliver fills for
   different positions in reverse arrival order.
4. While R1 has a non-terminal order, hold, causal gap or leased message,
   attempt pause and generation advance. Both must reject without mutation.
   Set kill to block every new-risk admission, keep fenced recovery workers
   alive, drain to finality and zero work, then pause: only transition
   sequence/fence advances; activation generation remains unchanged.
5. After a separately approved next activation with coupled kill-clear, attempt
   every mutation from
   the stale R1 session, worker lease, deployment/venue incarnation and cached
   candidate. Reconnect alone must not restore admission.
6. Restart R1 with cached state and disconnect it from authority updates, then
   attempt submission. Restore connectivity and require a fresh admission.
7. Run concurrent operator intents with identical, stale, and conflicting
   expected generations. Repeat enough times to cover each deterministic race
   schedule, not probabilistic timing alone.

**Evidence to retain**

- Instance/session/worker/lease/incarnation IDs, barrier schedule, transaction
  and lock traces, holds, global/per-order tails, before/after generations and
  fences, operator intent/receipt digests, journal/account results, stale-
  session errors, and canonical final-state comparison.

**Pass / fail**

- **PASS:** at most one runtime is authoritative; exactly one input/hold/outbox
  commit, one venue-input-source append and one next-event application win;
  scarce collateral admits at most one order; stale sessions, leases,
  candidates and incarnations cannot mutate; economic ordering follows global
  sequence; premature pause/
  activation is rejected; pause does not advance activation generation.
- **FAIL:** split brain, a post-fence commit, duplicate submission, last-write-
  wins authority, stale-cache recovery, or an unexplained lock/deadlock escape.

**Rollback / containment**

- Set the kill switch with the independent operator identity. Drain and
  reconcile before entering `PAUSED`; if finality is unknown, stay kill-set and
  unauthorised rather than forging a pause. Terminate stale sessions/leases and
  restore the clean fixture. Do not choose a winner by manually editing rows.

**Dependencies:** G0–G7.

**Applicability:** trajectory B only.

## G9 — Durable kill switch and loss of authority state

**Preconditions**

- G8 passed.
- The kill-switch source of truth, state transitions, operator identity,
  in-flight-order policy, health/readiness reason codes, persistence, and clear
  procedure are frozen.
- A submission recorder and transaction barriers exist.

**Exact scenario**

1. Start a disposable activation with both a new candidate and already accepted
   partial/open orders. Set the durable kill switch, restart each worker, and
   prove it persists. Attempt every new-risk path.
2. While set, deliver accepted fills, a causal gap that later closes, and
   cancellation outcomes. Prove command admission remains blocked while inbox
   ingestion, contiguous settlement, reconciliation and idempotent cancel/drain
   continue to venue finality.
3. Set the switch immediately before candidate admission, after candidate
   evaluation/before the hold/outbox transaction, during commit, and after
   venue acceptance/before local acknowledgement. Resolve every ambiguous
   identity by durable readback.
4. While the last observed state is clear, remove database connectivity,
   revoke read privilege, corrupt the response, delay it beyond the deadline,
   and return an unknown/unsupported value. Attempt submission throughout.
5. Restore the database with the switch set. Verify that cached clear state,
   process restart, and network reconnect cannot reopen submission.
6. Drain to zero gaps/quarantine/unresolved work, enter `PAUSED` without
   changing activation generation, and obtain a fresh signed next-epoch
   activation approval. Clear kill only as one atomic part of that approved
   `PAUSED -> ACTIVE` transition after fresh admission; a standalone clear or
   any clear while `ACTIVE` must reject. Verify that runtime, public API and
   monitoring credentials cannot clear it.

**Evidence to retain**

- Operator intent/receipt digests, durable switch rows/events, restart and
  fault timeline, readiness/health/log/metric reason parity, transaction-
  barrier results, submission-recorder output, and reconciliation of every
  in-flight intent; plus pause and next-epoch activation manifest/approval/
  receipt digests proving the coupled clear.

**Pass / fail**

- **PASS:** set state persists across restart; every new-risk path stops before
  candidate/hold/outbox commit; loss/ambiguity fails closed; already accepted
  work settles/cancels exactly once to finality; clearing is offline,
  authenticated, audited, possible only from `PAUSED`, coupled to one approved
  next activation, and followed by full admission.
- **FAIL:** runtime ignores the switch, cached clear state authorises work,
  database loss fails open, public/runtime code clears it, the switch strands
  accepted work, a standalone or `ACTIVE` clear succeeds, health stays falsely
  authorised, or an in-flight order is unexplained.

**Rollback / containment**

- Leave the kill switch set and runtime unauthorised. Keep the bounded drain/
  settlement path alive, reconcile outstanding orders and durable state, and
  require a fresh next-epoch activation approval while paused; never clear
  independently or auto-clear on restart/recovery.

**Dependencies:** G0–G8.

**Applicability:** trajectory B only.

## G10 — Graceful stop, crash/restart, and commit-unknown recovery

**Preconditions**

- G9 passed.
- Deterministic crash points exist around candidate/hold/outbox commit, command
  delivery, venue source-event append, event delivery, each projector DML,
  transaction commit/acknowledgement, lease handoff and shutdown drain.
- The journal/idempotency read-back algorithm and shutdown deadline are frozen.

**Exact scenario**

1. Send `SIGTERM` while idle, admitting a command, processing a partial fill,
   holding a causal gap, cancelling and committing. Verify immediate new-risk
   closure, bounded drain, final status and exit code. Unresolved work at the
   deadline must produce non-zero exit and an alert, never false clean.
2. Send `SIGKILL` at each crash point. Restart with the same immutable image,
   account, candidate, generation, deployment/venue incarnation and stable
   identities; prohibit blind retry until durable state is reconciled.
3. Cause PostgreSQL to commit but drop the acknowledgement, then cause rollback
   with the same client-visible timeout at each owner boundary. Resolve both by
   exact ID/hash/binding readback; never mint a replacement identity.
4. Drop the venue acknowledgement after acceptance and after rejection. Replay
   the exact same command ID and canonical hash: the committed acceptance or
   rejection must read back identically with no new command, event, hold or
   economic effect. Replay the same ID with different bytes and require
   `CONFLICT` with no mutation.
5. Recover orders in awaiting-acceptance, open, partial and cancel-pending
   states; recover a dropped prefix before applying higher global sequences.
6. Repeat with a stale generation/incarnation/lease introduced during downtime
   and with the kill switch set before restart.

**Evidence to retain**

- Signal/crash schedule, process exits, drain timeline, transaction/venue and
  lease traces, stable IDs/hashes, source/applied and risk/account/position
  tails, durable readbacks, generation/candidate/incarnations/fence, restart
  admission, journal/account digests and duplicate/missing-event audit.

**Pass / fail**

- **PASS:** graceful stop meets the frozen deadline or exits non-zero with exact
  unresolved evidence; no new risk opens after shutdown begins; every restart
  reconciles all non-terminal work before authorisation; committed effects
  occur exactly once, rolled-back effects not at all; every unknown resolves by
  readback; stale generation/incarnation/lease and set kill remain blocking.
- **FAIL:** shutdown reports success while work is unresolved, restart submits
  blindly, a committed event is lost/duplicated, unknown state becomes success,
  or admission is bypassed.

**Rollback / containment**

- Keep the kill switch set and `trading_authorized=false`, preserve transaction
  and venue evidence, and reconcile every unknown identity. Enter `PAUSED` only
  after finality and exact account/journal agreement. Never repair ambiguity by
  resubmitting with a new ID.

**Dependencies:** G0–G9.

**Applicability:** trajectory B only.

## G11 — Controlled fault injection and recovery

**Preconditions**

- G10 passed.
- `G12_INSTRUMENTATION_SMOKE=PASS` exists for the same candidate. Its signed
  receipt binds candidate/config digests, synthetic event IDs, expected bounded
  log/metric/alert signals, sink readbacks and zero economic side effects. Its
  semantic truth remains unaccepted until G12.
- Faults are executed first on disposable infrastructure with resource and
  network isolation. The Kali host has headroom limits and abort criteria that
  protect unrelated services.
- Fault duration, recovery timeout, retry/backoff policy, restart budget, and
  expected reason codes are frozen.

**Exact scenario**

1. Before any economic fault, emit a non-economic canary through every required
   log, metric and alert route. Read each sink back independently, verify exact
   reason/labels/timestamps, scan the receipt/output for secrets and unbounded
   IDs, and prove account/journal/venue/authority digests are unchanged. A
   missing, late, mismatched or economically mutating canary fails the gate.
2. Establish a normal paper workload and capture a no-fault baseline.
3. Inject one fault at a time at boundary durations: PostgreSQL latency,
   timeout, connection exhaustion, restart, read-only transaction, storage
   read-only/full, WAL/checkpoint pressure, DNS failure, packet loss, reset,
   TLS expiry/revocation, secret rotation, market disconnect/rate limit, clock
   skew, CPU throttling, memory pressure, disk pressure, log-sink loss,
   metrics-sink loss, and dependency restart.
4. Inject the reviewed compound cases: database loss plus stale market data;
   commit-unknown plus process crash; metrics loss plus runtime degradation;
   restart plus stale generation/kill switch; delayed projector plus queue
   backpressure; causal gap plus worker lease expiry; and cancellation plus a
   delayed causally earlier fill.
5. For each fault, attempt direct and scheduled submission, observe liveness,
   readiness, source/applied tails, holds, finality, retries, backoff, resource
   use, logs/metrics/alerts, and recovery.
6. Repeat the bounded host-safe subset on Kali. Compare unrelated-stack
   identities, health, restart counts, and resources before and after.

**Evidence to retain**

- Instrumentation-smoke intent/receipt, expected-signal manifest, independent
  sink readbacks, zero-side-effect digest comparison, secret/cardinality scan;
  baseline and fault schedules, injection-tool versions/configuration, complete
  timelines, reason-coded health/log/metric/alert evidence, retry/restart counts,
  resource profiles, submission recorder, recovery measurements, and Kali
  pre/post inventory hashes.

**Pass / fail**

- **PASS:** the instrumentation smoke proves every required route before fault
  injection with zero economic mutation; every authority/data fault blocks new-risk admission; already
  accepted work follows its bounded drain policy; retries, leases and queues
  stay within frozen limits; no economic event is skipped; no restart storm or
  resource escape occurs; truth remains accurate; recovery requires full
  admission; the Kali baseline is unchanged outside ELVIS.
- **FAIL:** false-green state, unbounded retry/resource growth, lost or duplicate
  event, auto-recovery from ambiguous state, missing alert, or collateral host
  impact.

**Rollback / containment**

- Abort injection at the frozen limit and set kill. If ELVIS is `ACTIVE`, keep
  it active but unauthorised while fenced ingest/settlement/cancel/reconcile
  drains to finality; commit `PAUSED` only after zero asynchronous work. Remove
  only the named fault controls, reconcile state, and restore from the pre-fault
  snapshot if necessary. Stop the Kali exercise before an unrelated service
  breaches its baseline; never restart Docker globally.

**Dependencies:** G0–G10. G12 instrumentation must already be deployed, though
G12 is accepted only after these truth tests.

**Applicability:** trajectory B only.

## G12 — Observability truth, dashboards, and alerts

**Preconditions**

- G11 passed.
- Metric names/units/labels, structured-log schema, health reason codes, SLOs,
  alert expressions/delays, cardinality budgets, retention, and redaction rules
  are versioned and frozen.
- One shared precedence table includes at least `READY`, runtime/admission/
  database/candidate/fence/kill/market failures, venue-worker not ready,
  outbox backlog, causal gap/conflict/finality unknown, settlement
  reconciliation, insufficient reservation, resource exhaustion and
  untrustworthy observability.
- One V2 dashboard is provisioned from source; synthetic fixed balances and
  legacy dashboard sprawl are excluded.

**Exact scenario**

1. For normal, `PAUSED`, `ACTIVE` rehearsal, kill-switch, stale-feed, database-
   loss, stale-generation, fence, provenance-drift, crash-recovery, backup, and
   restore states, compare durable truth with liveness, readiness, health,
   logs, metrics, dashboard, and alert output.
2. Re-run representative G5–G11 failures and assert exact reason-code parity
   across health, logs, metrics, and alerts. Measure detection and notification
   delay against the frozen thresholds.
3. Verify account equity, cash, positions, P&L, order/fill counts, generation,
   transition sequence, fence, writer identity, source/applied venue tails,
   outbox/inbox backlog, gap age, open orders by bounded state, oldest
   unresolved order, holds, settlement lag, quarantine, last-valid-market age,
   backup age and duplicate-economic-effect count against independent queries.
4. Remove the metrics collector, log sink, dashboard, and alert route
   separately. The runtime must follow the frozen observability-degradation
   policy without blocking its own shutdown or leaking data.
5. Run secret/private-data scans on labels, logs, traces, screenshots, alert
   bodies, dashboard JSON, and exported evidence; drive maximum-cardinality
   fixtures and verify limits.

**Evidence to retain**

- Versioned dashboard/alert/config digests, truth-comparison table, timestamped
  health/log/metric/alert samples, alert delivery receipts, independent account
  calculations, reason-code precedence table, loss-of-observability results,
  cardinality/resource reports, and redaction/secret-scan output.

**Pass / fail**

- **PASS:** every displayed value and status matches durable truth within the
  frozen timing/precision rules; no false green; required alerts arrive within
  threshold and recover correctly; duplicate economic effects stay zero; no
  secret/private data, IDs as labels or unbounded cardinality; observability
  failure follows policy.
- **FAIL:** synthetic or stale values appear authoritative, reason codes
  disagree, an alert is late/missing/noisy beyond policy, cardinality is
  unbounded, or sensitive data is exposed.

**Rollback / containment**

- Set kill when required observability is untrustworthy. If trading was
  `ACTIVE`, remain active but unauthorised while the still-trusted fenced
  recovery path drains accepted work; commit `PAUSED` only after finality and
  zero asynchronous work. Revert to the last verified versioned dashboard/alert
  configuration, preserve the failed samples, and repeat G11/G12; do not hide a
  runtime failure by changing visualization thresholds.

**Dependencies:** G0–G11.

**Applicability:** trajectory B only.

## G13 — Independent backup, corruption detection, and restore

**Preconditions**

- G12 passed.
- Backup scope, encryption, retention, off-host destination, immutability,
  credentials, restore procedure, and `RPO_TARGET`/`RTO_TARGET` are frozen
  before execution. Backup identity is separate and least-authority.
- The first 1B release admits only a quiescent DR backup: new risk is blocked,
  accepted work is drained to finality, and authority is `PAUSED` before the
  backup starts. Continuous recovery with in-flight async work is not claimed.
- A new isolated restore cluster/network is available; restore never overwrites
  the source cluster.

**Exact scenario**

1. While the rehearsal runtime is `ACTIVE`, set kill and prove
   `trading_authorized=false`. Continue fenced ingest, settlement,
   reconciliation and cancellation until venue-input/event source and applied
   tails match and orders, holds, gaps, leases, quarantine and queues are zero.
   Then CAS to `PAUSED` without advancing runtime generation.
2. Create the quiescent base backup and required manifest. Record exact
   authority, journal/account/risk, venue-input/event and per-order high-water
   marks. Freeze the measured pause-to-backup RPO contract.
3. Verify encryption, destination retention/immutability, checksums, manifest,
   ownership, and the ability of the backup identity to read only the declared
   scope. Prove the runtime identity cannot create/delete backups.
4. Restore into a new isolated PostgreSQL 15 cluster. Apply no ad-hoc repair;
   run catalog, migration, provenance, opening, replay, account, journal,
   authority/venue/risk/account sequences, generation, fence, holds, quarantine
   and manifest admission checks.
5. Reconcile the restored recovery point with the recorded high-water marks,
   measure actual RPO/RTO, record a separate new-database-incarnation admission
   receipt, rotate restored credentials, and stage the runtime in `PAUSED`
   only. The receipt does not change the restored runtime mode,
   `runtime_generation`, `authority_transition_sequence`, or writer fence; no
   `PAUSED -> PAUSED` authority transition exists. Require zero leases,
   equal source/applied tails and no unresolved order/hold/quarantine before
   admission. Perform read-only journal/venue replay and accounting verification
   while paused. A G7
   transaction requires a separate disposable activation and receipt.
6. Repeat with a deliberately truncated/corrupted backup, missing WAL,
   unavailable destination, expired credential, and wrong encryption key.
   Every invalid restore must be rejected before runtime admission.

**Evidence to retain**

- Backup/restore intent and receipts, tool/PostgreSQL versions, encrypted object
  identifiers/digests, manifest/checksum output, high-water marks, independent
  restore system identifier, catalog/state comparisons, RPO/RTO measurements,
  credential-rotation proof, paused replay receipt, separately approved
  disposable-activation/G7 receipt, and corrupt-restore rejection results.

**Pass / fail**

- **PASS:** an independently verified backup restores without source mutation;
  all durable invariants and economic equations match the declared recovery
  point; measured RPO/RTO meet frozen targets; invalid backups fail closed;
  restored runtime remains `PAUSED` until deliberately admitted.
- **FAIL:** backup cannot be independently restored, corruption is accepted,
  state/sequence/provenance differs, RPO/RTO is changed post hoc or missed, or
  restore starts authoritative.

**Rollback / containment**

- Destroy only the failed isolated restore cluster after preserving evidence.
  Keep the source account paused if backup confidence is lost; produce a new
  backup and repeat the entire gate. Never restore over the sole known-good
  cluster.

**Dependencies:** G0–G12.

**Applicability:** trajectory B only. Backups contain the approved opening and
fresh V2 history; V1 evidence remains in its separate read-only archive.

## G14 — Cut-over, pause, rollback, and authority state machine

**Preconditions**

- G13 passed.
- The transition graph, activation-generation/transition-sequence rules,
  activation-manifest and database-incarnation schemas, staged-admission proof,
  drain/reconcile procedure, writer credential revocation, intent expiry,
  approver identities, and recovery action for every state are frozen.
- G5 produced `G5_DISPOSABLE_ACTIVE_PASS` on the separate rehearsal lineage.
  The production target/account is new and has no rehearsal events.
- The signed production fresh-opening approval exists and binds the logical
  target. The retirement and activation approver identities/trust policies are
  available, but those two approvals do not yet exist: they must be signed in
  scenario order only after their target-specific receipts exist. The archive,
  backup and isolated-restore facilities are ready. G13 rehearsal evidence
  cannot substitute for any production receipt.
- Because this is the first V2 production activation, a same-candidate
  safe-redeploy manifest has already proved the exact candidate compatible with
  the migration head, codecs, economic versions and rendered configuration. Its
  digest is the real initial `rollback_candidate_manifest_sha256`; no older V2
  production candidate is invented.

**Exact scenario**

1. Rebuild a disposable qualification twin from the frozen `LEGACY` fixture.
   Repeat the bounded block-new, drain-to-zero, stop, checksummed archive,
   WORM/retention, isolated-restore and read-only-comparator proof; bind those
   receipts to a twin-only retirement approval before
   `LEGACY/0/S0 -> PAUSED/0/S1`. Take a quiescent twin-specific `PAUSED/0/S1`
   backup, restore it in isolation and bind that exact receipt to a separate
   twin-only activation approval with coupled kill-clear before reaching
   `ACTIVE/1/S2`. Repeat
   every forbidden transition and concurrent/stale/expired/wrong-target case.
   Rehearse normal rollback from `ACTIVE/N/S`: set kill, keep fenced
   ingest/settle/cancel/reconcile alive,
   drain to finality and zero async work, then CAS
   `ACTIVE/N/S -> PAUSED/N/(S+1)` without changing generation. Redeploy the
   approved rollback-candidate manifest, already proved compatible with the
   exact migration head, codecs and economic versions. Take and independently
   restore a fresh quiescent twin `PAUSED/N/(S+1)` backup. Obtain a fresh
   twin-specific next-epoch activation approval that binds this new backup/
   restore receipt and its coupled kill-clear, then activate
   `PAUSED/N/(S+1) -> ACTIVE/(N+1)/(S+2)`. Rehearse isolated DR with a new
   database incarnation that remains `PAUSED`. Inject lost acknowledgements
   around every twin transition and prove exact readback. Retain
   `G14_DISPOSABLE_CONTROL_PASS`; never copy this history to production.
2. On the new production target, repeat target-specific G1 isolation, G2 TLS/
   catalog/roles/ACL and G3 signed-opening procedures. Bind new deployment-
   target and provisioning records to the unchanged release candidate. Prove
   opening-only state: `LEGACY/0/S0`, zero venue-input/event tails, orders, holds,
   leases and quarantine. Never copy or reset the rehearsal account.
3. Before touching V1, take an encrypted opening-only backup of this exact
   production target and restore it into a new isolated cluster. Verify catalog,
   opening/provenance, zero async tails/work, authority `LEGACY/0/S0`, checksums
   and measured RPO/RTO; keep the restored cluster non-authoritative. This is a
   pre-retirement recovery proof only. It may bind the retirement decision but
   expires as a rollback point when retirement commits and can never restore a
   post-retirement writer. The G13 rehearsal backup cannot substitute for it.
4. Starting from that fresh opening, re-run the non-mutating forbidden-
   transition matrix: direct `LEGACY -> ACTIVE`, every transition to `LEGACY`,
   stale generation/sequence, expired intent, wrong target/incarnation/
   candidate, unauthorised caller and conflicting concurrent intents all reject
   without partial mutation.
5. Block and quiesce all new V1 work without revoking its identity, drain its
   in-flight work and sessions to zero, prevent reconnect, then stop the V1
   process. Take the final V1 evidence archive, verify its manifest, integrity,
   retention/WORM controls and independent isolated restore, then run only the bounded
   read-only comparator. Now sign the production `RETIRE_V1_WRITER` approval,
   binding the exact V1 process/database/login, stopped-and-drained receipt,
   archive/restore/comparator receipt, opening-backup receipt, target V2
   identity and preflight receipt. Permanently revoke the writer credential,
   prove the old process/credential/session cannot reconnect, commit or submit,
   and only then CAS `LEGACY/0/S0 -> PAUSED/0/S1`. Exact commit-unknown replay is
   read-only; rejected intents do not consume `S`.
6. Immediately take an encrypted quiescent production backup at `PAUSED/0/S1`
   and restore it into another isolated cluster. Verify opening/provenance,
   retirement receipt, catalog, zero asynchronous tails/work, checksums and
   measured RPO/RTO. Restore preserves `PAUSED/0/S1` and emits only a new
   database-incarnation admission receipt; it performs no authority transition.
   This is the first post-retirement DR point. If the source is lost before this
   receipt exists, remain unavailable and reconstruct only through the signed
   fresh-B opening and retirement evidence on a new target; never revive V1 or
   treat the earlier `LEGACY` backup as authoritative.
7. Start the exact V2 candidate in staged mode. Prove `liveness=true`, then
   independently prove `admission_valid` and `market_ready` while
   `trading_authorized=false`. Only now sign the production activation approval,
   binding the accepted retirement receipt, exact
   `paused_backup_restore_receipt_sha256`, and exact proposed epoch. Create the
   pre-CAS activation manifest and execute
   `PAUSED/0/S1 -> ACTIVE/1/S2` with its digest, opening, release candidate,
   rollback candidate, database/deployment/venue incarnations, zero async work,
   writer fence and coupled clear-kill intent under one lock. Emit the separate
   activation receipt. Transient market readiness is not part of that CAS;
   authorisation becomes true only after the runtime observes the committed
   epoch and freshly revalidates market readiness at command admission.
8. Inject no deliberate fault into the production cut-over. If acknowledgement
   is naturally ambiguous, stop and resolve only by transition/audit ID,
   activation-manifest digest, activation receipt, target state, sequence,
   generation, candidate and incarnation readback. Exact replay is idempotent;
   conflicts reject; rejections do not consume `S`.
9. Execute one separately approved bounded production paper canary: commit one
   deterministic order/hold/outbox, observe immediate ACK, advance it only with
   declared later venue inputs, apply exact partial/multiple fills to finality,
   reconcile the account and prove all holds, queues, gaps and non-terminal work
   return to zero. No live venue endpoint or credential may exist. Bind the
   complete economic and causal receipt to the production activation receipt.
10. Record `G14_PRODUCTION_CUTOVER_PASS` only after that canary and after the
   observed Kali endpoint is the approved `ACTIVE_PAPER_PRODUCTION` target and
   every unrelated stack is unchanged. Do not exercise pause, rollback or DR
   against this fresh production history before G15. A stopped/paused deployment
   or activation without the canary is useful evidence but is not a G14 pass and
   cannot unlock G15.

**Evidence to retain**

- Economic-state/runtime-mode/admission/market/authorisation timeline;
  `G5_DISPOSABLE_ACTIVE_PASS` and `G14_PRODUCTION_CUTOVER_PASS` receipts;
  separate production opening, post-archive retirement and post-retirement-
  backup activation approvals; production opening-only backup, V1 WORM/archive/
  isolated-comparator receipt, and production `PAUSED` backup/restore receipt;
  sequence/generation/fence/incarnation timeline; signed intents/receipts and
  expiries; CAS outcomes; release-candidate, deployment-target, activation-
  manifest and activation-receipt digests; session/credential drain and
  revocation proof; old/new writer matrix; venue/account reconciliation;
  commit-unknown readbacks; deterministic production-canary receipt; separate
  twin initial and reactivation PAUSED backup/restore, code-rollback and DR
  receipts; backup references; and Kali pre/post unrelated-stack inventory.

**Pass / fail**

- **PASS:** the separate production target reaches the approved
  `ACTIVE_PAPER_PRODUCTION` endpoint and completes the approved deterministic
  paper canary; only legal intermediate states occur;
  every transition is atomic,
  idempotent, least-authority, and bound to the frozen target; exactly one
  writer exists after each accepted transition; lost acknowledgements resolve
  by readback; every twin activation binds its applicable quiescent PAUSED
  backup/restore receipt; no pause/activation can cross unresolved venue work;
  normal rollback preserves the database and DR uses a new incarnation; V1
  never regains authority.
- **FAIL:** a forbidden direct transition, self-activation, partial state,
  stale/concurrent success, dual/no unexplained writer, post-fence commit,
  blind retry, V1 reactivation under B, missing post-retirement `PAUSED` backup,
  or production activation/canary without their separate explicit approvals.

**Rollback / containment**

- Set the kill switch immediately. Remain `ACTIVE` but unauthorised while the
  bounded recovery path drains already accepted work, and enter `PAUSED` only
  after finality; then terminate/redeploy and follow the verified code-rollback
  or isolated-DR path. Never run
  both writers, edit activation rows manually, restore over the only known-good
  database, or infer transition success from a timed-out command.

**Dependencies:** G0–G13.

**Applicability:** trajectory B only.

## G15 — Staged 24-hour and 72-hour paper soak

**Preconditions**

- `G14_PRODUCTION_CUTOVER_PASS` exists for the exact Kali target and candidate,
  and the observed endpoint is `ACTIVE_PAPER_PRODUCTION`. A disposable
  rehearsal or `DEPLOYED_PAUSED/PRODUCTION_READY` cannot start G15.
- The identities, trust policies and canonical templates for every planned soak
  reactivation are frozen. No reactivation approval is signed in advance: each
  one must be created only after its target-specific pause and quiescent
  backup/restore receipts exist. The G14 activation approval is single-epoch
  and cannot be reused.
- Workload/corpus, symbols, virtual/real time usage, market regimes, backup
  schedule, controlled-fault schedule, resource budgets, alert thresholds,
  latency/error SLOs, and discrepancy tolerances are frozen before the clock
  starts.
- Hard-zero invariants are declared: live orders, duplicate economic effects,
  unexplained account differences, stale-writer commits, false-green readiness,
  leaked secrets, negative effective collateral, causal gaps/quarantine/
  dead-lettered economics at gate exit, unresolved orders/holds/leases/queues at
  gate exit, and collateral host changes must all remain zero.
- A machine-readable threshold manifest is reviewed after the fresh Kali
  no-fault baseline and before the clock starts. Initial safety ceilings are:
  candidate/hold/outbox commit p99 <= 250 ms; command-to-ACK, ingest-to-apply
  and fill-to-settlement p99 <= 1 s; kill commit to new-risk authorisation false
  <= 1 s; gap detection <= 2 s; critical alert delivery <= 60 s; staged restart
  <= 30 s; graceful drain <= 30 s or non-zero exit; and recovered backlog zero
  <= 60 s. These values are unproved until measured and may not be relaxed
  post hoc to turn a failure into a pass.
- Compose hard limits, no OOM/restart storm, p95 CPU below 70% of its assigned
  allocation, final 12-hour RSS drift at most 10% after warm-up, database
  connections below 80% of the declared pool/server limit, and at least 20%
  free disk plus seven days of projected growth are initial resource gates;
  logs and metrics have explicit caps.

**Exact scenario**

1. Run a continuously monitored 24-hour paper stage covering quiet/active periods,
   rejection, zero/full/partial/multiple fills, cancellations/races, duplicate/
   reordered/delayed delivery, disconnect/reconnect, worker/process/dependency
   restart, secret rotation, alert firing/recovery, and one planned
   kill/drain/pause. Take the quiescent backup while paused, verify it on the
   isolated restore target, then sign a distinct next-epoch approval whose
   activation-intent core binds that pause and backup/restore receipt. Use it to
   clear kill and reactivate with a new activation manifest.
   Monitoring and the stage clock continue through the planned pause.
2. Review the complete 24-hour evidence. Any hard-zero violation or severity-1
   defect fails the gate. Corrected runtime/configuration bytes create a new
   candidate and reset the soak clock.
3. Only after signed 24-hour acceptance, run a new continuously monitored
   72-hour soak with
   the same candidate and declared workload. Include at least one backup and
   independent restore check using the same quiescent pause contract, one
   commit-unknown exercise, one kill-switch exercise, and bounded G11 fault
   cases. After each pause and backup/restore receipt, sign and consume a new
   target/epoch-specific approval; missing or expired approval leaves the target
   safely `PAUSED` and fails the stage.
4. Continuously compare market input, decisions, submissions, venue events,
   global source/applied tails, per-order finality, holds, journal/account/
   position truth, health, alerts, CPU, memory, disk, network, database growth/
   locks/connections, restarts, and unrelated Kali baseline.
5. At completion, independently reconcile every order/fill/account equation,
   inspect all alerts/logs, verify the latest backup, and repeat readiness,
   stale-writer, kill-switch, and deterministic G7 probes.

**Evidence to retain**

- Signed threshold/workload manifest, start/end UTC and uptime proof, immutable
  time-series/log/alert exports, order/fill/reconciliation manifests, resource
  and database-growth reports, restart/fault/pause/backup/restore receipts,
  per-epoch activation manifests/approvals/receipts, daily
  reviewer notes, issue ledger, 24-hour acceptance, 72-hour acceptance, and
  Kali baseline comparison.

**Pass / fail**

- **PASS:** both continuously observed stages complete on the same candidate;
  planned pause intervals remain inside the declared stage clocks; every
  hard-zero invariant remains zero; all frozen SLOs/resource/discrepancy limits
  pass; source/applied tails converge and all async work/holds are zero at each
  stage exit; final independent replay/reconciliation and backup verification
  pass; no unresolved P0/P1 defect remains.
- **FAIL:** missing monitoring interval/evidence, threshold changed after start,
  clock reset concealed, unplanned restart, hard-zero violation, unresolved
  accounting/authority ambiguity, SLO breach, or unrelated host regression.

**Rollback / containment**

- Set the kill switch while still `ACTIVE`, make
  `trading_authorized=false`, keep fenced ingest/settlement/cancel/reconcile
  alive, and drain to finality. Enter `PAUSED` only after all async work is
  zero, retain the failed interval, and follow G14 rollback. Fixes produce a
  new candidate; repeat affected gates and restart both soak stages from zero.

**Dependencies:** G0–G14.

**Applicability:** trajectory B only.

## G16 — Bit-for-bit runtime packaging, documentation, and public release

**Preconditions**

- G15 passed on the exact release candidate.
- All P0/P1 defects are closed with reproduction, fix, regression evidence and
  release linkage. Installation, operation, backup, restore, pause, rollback,
  upgrade, incident and uninstall/cleanup runbooks are frozen.

**Exact scenario**

1. Re-read the candidate that passed G15 and prove every bundle/image digest is
   unchanged. Promote those exact bytes to immutable GitHub/GHCR version and
   commit tags without rebuilding or moving an existing tag.
2. Download every promoted asset and image anonymously, verify checksums,
   attestations, platform SBOM/provenance subjects and equality to the G15
   candidate. Only this destination readback makes the release `RELEASED`.
3. Have an independent operator follow the public documentation on clean,
   disposable production-twin environments for supported amd64 and, where
   claimed, arm64 platforms. Execute every command exactly as published;
   verify configuration, health/status envelope, deterministic venue smoke,
   upgrade, backup/restore, pause/rollback and cleanup without a live
   credential. No destructive runbook is exercised against the accepted Kali
   production target in this step.
4. Scan all V2 documentation/release assets for stale V1 authority claims,
   unsupported Python versions, mutable tags, unsafe defaults, live-trading
   instructions, broken links, private details and CLI drift. Published V2
   installation requires Python 3.14 and never offers an earlier interpreter.
5. On Kali, perform read-only public-destination digest, attestation, status,
   activation-manifest/receipt and durable-tail readback only. Prove the
   endpoint still
   matches `ACTIVE_PAPER_PRODUCTION` and all unrelated-stack identities,
   health and restart counts match the G15 baseline. Any repoint or redeploy
   requires the G14 kill/drain/pause procedure and a new approval.

**Evidence to retain**

- Final G0 receipt; clean-install transcripts per platform; docs/link/config/
  secret/version scans; closed bug ledger; GitHub PR/check/release URLs;
  anonymously downloaded public assets and verification output; Kali endpoint
  evidence; and the final runtime release-manifest digest.

**Pass / fail**

- **PASS:** a new operator reproduces the paper install from public assets;
  every claim matches observed endpoint state; public bytes equal the G15
  candidate; Python 3.14-only and paper-only boundaries are exact; assets are
  downloadable and cryptographically bound; Kali remains healthy and unrelated
  stacks are unchanged.
- **FAIL:** any command/link is unverified, public bytes differ, docs overclaim
  runtime/authority, an unsupported interpreter or live trading remains an install path,
  destination state is assumed rather than read back, or Kali differs from the
  accepted candidate.

**Rollback / containment**

- Keep or return new-risk admission to kill-set and then `PAUSED` under G14 if
  published bytes or runtime evidence are wrong. Mark the release withdrawn or
  superseded without replacing immutable assets; preserve the evidence chain
  and publish a new version only after corrected gates.

**Dependencies:** G0–G15 and the separate explicit approval already required
for the observed production `ACTIVE` canary.

**Applicability:** trajectory B only.

## G17 — Separate screenshots, course video, and media publication

**Preconditions**

- G16 passed and the exact public runtime release is immutable and verified.
- A separate storyboard, narration, transcript, captions, media-rights review,
  provenance schema, redaction checklist, target formats and public/private
  evidence boundary are approved. The historical alpha.2 preview capture stays
  labelled as such and is never rewritten as production evidence.
- A content-addressed offline media manifest pins every source asset, licence,
  font, renderer, encoder, dependency and configuration byte. A clean-room
  rebuild procedure and expected output hashes are approved; no CDN, floating
  package invocation or host font may enter the build.
- Final rendering still requires the user’s explicit approval of a Studio
  preview.

**Exact scenario**

1. Capture the production-course journey from the exact G16 release with
   dedicated non-secret paper fixtures and no real endpoint/credential. Cover
   release identity, isolation/checksums, TLS/roles, disposable signed opening,
   paused readiness, reject/zero/partial/multi-fill/cancel, duplicate/reorder/
   gap/restart, kill-with-drain, truthful dashboard, restore-stays-paused,
   rollback/reactivation rehearsal, cleanup and public reinstall verification.
2. Retain each private raw original outside Git with capture UTC, exact command
   or source URL, commit, release/image digest, raw SHA-256, crop/redaction
   operations, public SHA-256, reviewer and rights status. Private IPs, users,
   DSNs, keys, certificates, secrets and unrelated stacks never appear.
3. Produce public-safe screenshots and a Studio preview at 1920x1080/30 fps.
   Measure final-frame terminal/body text at least 24 px, normal-text contrast
   at least 4.5:1 (3:1 for large text), and keep captions to at most two lines
   of 42 characters inside the declared title/caption safe areas. Captions
   cover 100% of spoken content; state is never conveyed by colour alone;
   flashes never exceed three per second. The reduced-motion checklist requires
   no parallax, continuous pan/zoom, cursor blink or non-essential animation;
   transitions are cuts or dissolves no longer than 200 ms, and every required
   state change remains understandable as a static frame. Measure audio with an
   ITU-R BS.1770-4/EBU R128 meter: integrated loudness must be between -17 and
   -15 LUFS and true peak no higher than -1 dBTP.
   Verify command/narration alignment, paper/rehearsal watermarking and exact
   authority labels on a full contact sheet plus first/middle/last frames.
4. Rebuild the pre-approval package twice from two clean offline work
   directories using the pinned manifest. Require byte-identical screenshots,
   captions, transcript, contact sheet and frame/audio render plan, including
   normalized deterministic metadata. Do not encode a final MP4; any hash
   difference fails reproducibility and blocks approval.
5. Stop for explicit user approval. Only after approval, render the final
   master/delivery MP4 twice from clean offline work directories, require
   byte-identical outputs, perform visual/audio/frame/duration/checksum QA and
   bind every output to the media evidence manifest.
6. Publish media as a separate course asset/update; verify screenshots,
   transcript/captions, video and checksums at the destination without replacing
   or mutating runtime-release artefacts.

**Evidence to retain**

- Raw-to-public media digest map, source/licence and redaction reviews,
  storyboard/transcript/captions, Studio-preview approval, final render QA,
  content-addressed toolchain/source/font manifest, two clean-build logs and
  hash comparisons, measured text/contrast/caption/flash/motion/loudness
  results with meter version, reduced-motion checklist, contact sheet, delivery
  checksums, destination URLs/readback and the final media-manifest digest.

**Pass / fail**

- **PASS:** every frame and statement corresponds to the exact G16 release;
  both clean offline rebuilds are byte-identical; every numeric
  media/accessibility threshold passes; media is rights-cleared,
  secret-safe, explicitly paper only, approved and independently verified at
  its destination.
- **FAIL:** provenance/rights/redaction is missing, a private or live detail is
  visible, an authority claim is false, final render precedes preview approval,
  destination state is assumed, or media is attached by replacing runtime
  artefacts.

**Rollback / containment**

- Restrict or remove only the unsafe media through an explicit logged
  publication action, preserve private originals/evidence and issue corrected
  media under a new manifest. A media failure does not invalidate safe G16
  runtime bytes but blocks course completion.

**Dependencies:** G16 and explicit Studio-preview approval for final render.

**Applicability:** trajectory B only.

## Final acceptance record

The programme is accepted only when one final manifest lists `PASS` receipts
for G0 through G17, in order, on the same valid candidate lineage. It must also
state:

- confirmation that trajectory B reached production with no V1 continuity;
- the exact paper-only account, economic state, authority mode, transition
  sequence, activation generation, fence and database incarnation; the exact
  release-candidate, deployment-target, pre-CAS activation-manifest, post-CAS
  activation-receipt and final gate-state-snapshot digests; and the commit,
  image, bundle, rendered configuration, opening and gate-receipt digests;
- the final authoritative writer and proof that every stale writer is unable to
  commit;
- the verified backup/restore point, normal V2 code-rollback path and separate
  disaster-recovery path;
- the separate activation and production-canary approvals and observed canary
  that placed and qualified the exact Kali target in
  `ACTIVE_PAPER_PRODUCTION`, plus the final course-render approval;
- the destination verification of GitHub and Kali; and
- zero unresolved P0/P1 defects and the disposition of every lower-severity
  issue.

If any item is absent, indirect, stale, associated with different bytes, if the
endpoint is only `DEPLOYED_PAUSED/PRODUCTION_READY`, or if recovery depends on
restoring V1 writer authority, the outcome remains **not accepted**.
