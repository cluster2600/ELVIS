# ELVIS V2 paper-production failure register

> **Status authority:** every entry in this register is `OPEN`. A passing unit
> test, an installable operator preview, or a documentation update does not
> close a runtime-production failure. Closure requires the named regression
> test and gate evidence on the exact release commit and immutable image digest.
> `ACTIVE` remains a **NO-GO** while any P0 entry is open.

## Scope and evidence rules

This register covers confirmed gaps between the current `v2.0.0-alpha.2`
operator preview/compatibility runtime and the approved trajectory B objective
in the [production plan](05-v2-production-plan.md): one fresh V2 paper opening,
no V1 balance, P&L, position, event, or accounting continuity, and V1 retained
only as operationally read-only evidence. Immutability additionally requires
independent integrity and WORM/retention proof. The register records defects and
operational risks; it does not authorise an account opening, activation,
database cut-over, live trading, or destructive recovery.

- **P0:** can permit an unsafe paper order, report false authority, corrupt or
  lose durable state, or makes the claimed V2 production runtime impossible.
- **P1:** materially weakens reliability, security, operability, supply-chain
  reproducibility, or production evidence, but is not by itself an immediate
  unsafe-submission path.
- **P2:** misleading or maintainability debt that can confuse operators or
  obscure evidence.
- A status may change from `OPEN` only when its closure gate is evidenced. Code
  presence alone is not evidence that a runtime path consumes it.
- A rollback must preserve evidence and fail closed. Under B it may pause,
  restore, or redeploy only V2; it must never return writer authority to V1,
  silently patch history, create two writers, or bypass an approval gate.
- `runtime_generation` is an activation epoch. Entering `PAUSED` does not change
  it. A separate append-only `authority_transition_sequence` advances on every
  accepted authority transition.
- Every containment path that may begin in `ACTIVE` follows one order: set or
  confirm kill and `trading_authorized=false`; keep only fenced ingest,
  settlement, idempotent cancellation and reconciliation alive; recover every
  unknown identity; drain to venue finality and zero queues, gaps, quarantine,
  leases, non-terminal orders and holds; then commit `PAUSED` at the unchanged
  runtime generation. Only after that pause may an operator stop, revoke,
  restore or redeploy the runtime. If the target is already `LEGACY` or
  `PAUSED`, it remains there and no synthetic transition is created.

## Closure-gate glossary

| Gate | Required evidence |
|---|---|
| G0 | Supply-chain lock, build reproducibility, SBOM, attestation, vulnerability and immutable-digest checks |
| G1 | Clean-directory install and isolated host preflight |
| G2 | PostgreSQL 15 bootstrap, TLS, HBA, role, ownership and ACL proof |
| G3 | Canonical fresh-opening intent, independent approval, exact provisioning and no-V1-continuity proof |
| G4 | Empty legacy-economic-state proof, deterministic post-opening V2 replay and side-effect-free offline V1 decision comparison |
| G5 | Pre-side-effect admission plus distinct liveness, service-readiness and `trading_authorized` truth |
| G6 | Missing, stale, malformed and disconnected market-feed fault tests |
| G7 | Deterministic asynchronous paper venue, capacity-hold, fill-by-fill accounting and side-effect-free comparison proof |
| G8 | Concurrency, sole-writer, stale activation-epoch, authority-transition-sequence and fence tests |
| G9 | Durable kill-switch activation, restart, dependency-loss and clear tests |
| G10 | SIGTERM, SIGKILL, crash/restart and commit-unknown recovery tests |
| G11 | Dependency and infrastructure fault injection |
| G12 | Metrics, logs, dashboard, alerts and operator-truth contract |
| G13 | Independently verified backup plus full restore rehearsal |
| G14 | B-only `LEGACY/0 -> PAUSED/0 -> ACTIVE/1`, V1 retirement plus archive integrity/WORM/isolated-restore/read-only-comparator proof, pause/resume and V2-only recovery rehearsal |
| G15 | Reviewed 24-hour then 72-hour paper soak against explicit SLO/resource thresholds |
| G16 | Bit-for-bit runtime packaging, public release and clean-install destination QA |
| G17 | Separate screenshot/course capture, provenance, approval, render and publication QA |

## Operator triage index

Every row is `OPEN`; the next action is containment or the next implementation
boundary, never permission to activate.

| ID | Operator symptom | State | Gate | Next safe action |
|---|---|---|---|---|
| P0-001 | Emergency stop says success but new orders can submit | OPEN | G9 | Keep V2 unauthorised; implement command-boundary kill/drain |
| P0-002 | Health/account `ACTIVE` can look authorised | OPEN | G5/G12 | Treat status as unavailable; add canonical authority envelope |
| P0-003 | Direct `LEGACY -> ACTIVE` and no transition ledger | OPEN | G8/G14 | Keep `LEGACY`; add forward-only transition capability |
| P0-004 | No supported pause or V2-only recovery | OPEN | G10/G14 | Use no runtime authority; build offline pause/readback path |
| P0-005 | Executable process is still the compatibility runtime | OPEN | G5/G7/G8 | Do not deploy as V2; build separate async composition |
| P0-006 | Runtime DDL can fail open | OPEN | G2/G5/G11 | Remove runtime schema authority; fail admission closed |
| P0-007 | SIGTERM can crash during close reporting | OPEN | G10/G11/G14 | Treat shutdown as unresolved; implement typed drain receipt |
| P0-008 | Fresh opening lacks signed approval/nonce binding | OPEN | G3/G4/G14 | Keep target unopened; add signed intent and atomic opening |
| P0-009 | No verified backup/restore | OPEN | G13/G14 | Keep production inactive; implement isolated restore drill |
| P0-010 | Public V2 image has operator tools, no runtime | OPEN | G0/G1/G16 | Keep alpha.2 labelled preview; build separate runtime image |
| P0-011 | V1 writer retirement/immutability unproved | OPEN | G7/G8/G14 | Do not activate; add retirement plus G14 archive/WORM/restore evidence |
| P0-012 | Owner/schema require terminal full fill | OPEN | G2/G7/G10 | Retire batch path from production; add progressive async path |
| P0-013 | Open orders reserve no collateral | OPEN | G7/G8/G15 | Block async submission; add account-global holds |
| P0-014 | No causal venue-input/event transport or atomic event projector | OPEN | G2/G7/G8/G10/G11 | Preserve raw facts; add ordered-input inbox/outbox/projector |
| P0-015 | Restart/pause can strand non-terminal work | OPEN | G9/G10/G14/G15 | Stay kill-set; recover/drain before any pause/reactivation |
| P1-001 | Activation role has broad direct DML | OPEN | G2/G14 | Revoke direct mutation; expose narrow capabilities only |
| P1-002 | Replay/readiness load unbounded history | OPEN | G4/G15 | Keep paused on limits; add bounded replay/checkpoints |
| P1-003 | Full side-effect-free V2 comparison absent | OPEN | G7/G15 | Use only frozen offline fixtures; never grant V1 oracle status |
| P1-004 | Dependencies/images are mutable | OPEN | G0/G16 | Pin locks/digests before candidate creation |
| P1-005 | Root Compose is unsafe compatibility topology | OPEN | G1/G2/G16 | Quarantine it; create isolated V2 composition |
| P1-006 | Metrics fabricate equity/suppress DB failure | OPEN | G5/G12 | Publish unavailable state; remove synthetic production values |
| P1-007 | Prometheus target is topology-dependent | OPEN | G1/G12 | Keep paused without trusted monitoring; use internal service DNS |
| P1-008 | Performance/SLO checks are absent from CI | OPEN | G4/G7/G15 | Freeze bounded release budgets before candidate |
| P1-009 | Start/stop API mutates process-local fiction | OPEN | G12/G14 | Remove mutating claims; make status read-only |
| P1-010 | No runtime crash/commit-unknown E2E | OPEN | G8/G10/G11 | Keep candidate unreleased; add deterministic failpoints |
| P1-011 | Model approval lacks full provenance | OPEN | G0/G5/G7 | Bind reviewed model/feature manifest into candidate |
| P1-012 | TLS/HBA/secrets/network policy not composed | OPEN | G1/G2/G11 | Use no production credential; build isolated trust boundary |
| P1-013 | 24h/72h SLO/resource proof absent | OPEN | G15 | Keep non-production label; run same-byte staged soak |
| P1-014 | Async retention/backpressure/SLOs undefined | OPEN | G11/G12/G15 | Stop producers on bounds; retain causal evidence |
| P2-001 | Docs claim a kill-switch hot-loop wiring that is absent | OPEN | G16 | Correct claim to unavailable until P0-001 closes |
| P2-002 | Console begins with fabricated portfolio values | OPEN | G12/G16 | Display `UNAVAILABLE`; keep fixtures separate |
| P2-003 | REST API returns mock balances/positions | OPEN | G12/G16 | Return typed unavailable/503; never fallback to mocks |

## P0 — production blockers

### ELVIS-V2-P0-001 — Emergency halt is not enforced at submission

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** [`trading/utils/trade_history_api.py:706-729`](../../trading/utils/trade_history_api.py#L706-L729) exposes an in-memory/Redis-backed `is_trading_halted()` helper, while the hot order path reaches `order_service.submit(intent)` without consulting it at [`main.py:2420-2435`](../../main.py#L2420-L2435).
- **Impact:** `POST /emergency_stop` can report success while the running loop continues to submit paper orders. Redis loss also leaves the authoritative halt state ambiguous.
- **Reproduction and expected regression:** activate the emergency stop immediately before a valid BUY/SELL decision and observe that the current submit spy is still called. The regression must prove zero new candidate, hold or submit-outbox commit while halted, after restart and when durable halt state is unreadable. It must simultaneously prove that inbox ingestion, settlement of already accepted fills, reconciliation and idempotent cancel/drain continue. Clearing requires an authenticated offline operation and fresh admission.
- **Remediation:** move the halt decision into the command-admission transaction, back it with one durable authority, and fail closed on unreadable activation epoch, transition sequence or kill-switch state. Treat the REST API as an operator client, not the authority; never use the switch to strand accepted venue work.
- **Closure gate:** G9, with G7 and G11 evidence on the exact runtime image.
- **Rollback:** if the new halt authority or its dependency fails while `ACTIVE`, block new risk independently, keep fenced recovery processing alive, reconcile and drain accepted work to finality and zero asynchronous work, then commit `PAUSED` at the unchanged activation epoch. If already `PAUSED`, remain there. Do not fall back to the current unchecked hot loop or V1.

### ELVIS-V2-P0-002 — Health and account state can report false authority

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** the compatibility health endpoint unconditionally returns `"status": "healthy"` at [`trading/utils/trade_history_api.py:950-960`](../../trading/utils/trade_history_api.py#L950-L960); the second API health endpoint is likewise static at [`trading/api/app.py:127-138`](../../trading/api/app.py#L127-L138). The durable account stream also defaults its economic solvency state to `ACTIVE` at [`trading/persistence/sql_migrations/0003_paper_account_ledger.sql:38-68`](../../trading/persistence/sql_migrations/0003_paper_account_ledger.sql#L38-L68). None of these facts proves an approved opening, runtime `ACTIVE`, current activation epoch, transition sequence, fence, feed, or readable kill switch.
- **Impact:** Docker/orchestrators and operators can confuse process liveness, service readiness, or an economically solvent account with permission to submit. They can keep routing or trusting a runtime that is deliberately `PAUSED`, has lost its database, or holds stale authority.
- **Reproduction and expected regression:** sever PostgreSQL, corrupt the expected activation epoch or transition sequence, make kill-switch state unreadable, or open a solvent account while runtime mode remains `PAUSED`; current surfaces can still appear healthy/`ACTIVE`. Regression must expose distinct liveness, service-readiness and `trading_authorized` values with stable reason codes. `trading_authorized` may be true only for runtime `ACTIVE` with the exact approved opening, current activation epoch/transition tail, fence, candidate and readable clear kill switch; economic account state `ACTIVE` alone must never authorise submission.
- **Remediation:** add pre-side-effect admission, keep liveness narrow, derive service readiness and `trading_authorized` from the same exact durable evidence consumed at the submission boundary, and label account `ACTIVE` explicitly as economic solvency rather than runtime authority.
- **Closure gate:** G5 and G12, including G11 dependency-loss evidence.
- **Rollback:** on probe uncertainty, make `trading_authorized=false`. If `ACTIVE`, set or confirm kill, keep the trusted fenced recovery path alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`; if already non-active, remain there. Keep the process observable and never restore static green as a compatibility fallback.

### ELVIS-V2-P0-003 — Activation can skip PAUSED and no append-only authority transition history exists

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** the singleton control starts at `LEGACY/0` but stores no transition sequence or immutable transition receipt at [`trading/persistence/sql_migrations/0004_paper_runtime_control.sql:1-23`](../../trading/persistence/sql_migrations/0004_paper_runtime_control.sql#L1-L23). `PaperRuntimeActivationSource` accepts `LEGACY` generation `0` at [`trading/application/paper_runtime_activation.py:53-96`](../../trading/application/paper_runtime_activation.py#L53-L96), and the SQL capability accepts `LEGACY/0` then writes `ACTIVE/1` at [`trading/persistence/sql_migrations/0006_paper_runtime_activation_capabilities.sql:72-105`](../../trading/persistence/sql_migrations/0006_paper_runtime_activation_capabilities.sql#L72-L105) and [`:125-134`](../../trading/persistence/sql_migrations/0006_paper_runtime_activation_capabilities.sql#L125-L134). The existing append-only generation table records only activation epochs at [`trading/persistence/sql_migrations/0005_paper_runtime_generation.sql:1-12`](../../trading/persistence/sql_migrations/0005_paper_runtime_generation.sql#L1-L12); readiness currently assumes those epochs are contiguous up to `runtime_generation` at [`trading/persistence/paper_account_readiness.py:1422-1467`](../../trading/persistence/paper_account_readiness.py#L1422-L1467).
- **Impact:** activation can skip the mandatory fresh-opening pause, V1 writer retirement and reconciliation boundary. Reusing `runtime_generation` for pauses would also corrupt its activation-epoch meaning and current readiness assumptions, while updating only the singleton would leave no auditable transition history.
- **Reproduction and expected regression:** invoke activation from `LEGACY/0`; current contracts accept it. Regression must enforce `LEGACY/0@sequence=0 -> PAUSED/0@sequence=1 -> ACTIVE/1@sequence=2` initially and `ACTIVE/N@sequence=S -> PAUSED/N@sequence=S+1 -> ACTIVE/N+1@sequence=S+2` thereafter. `runtime_generation` changes only on accepted `PAUSED -> ACTIVE`; the separate append-only `authority_transition_sequence` increases by exactly one on every accepted transition. Pure, PostgreSQL and operator tests must reject direct `LEGACY -> ACTIVE`, every transition to `LEGACY`, `SHADOW` as an authority state, stale epoch/sequence, gaps and conflicting concurrent intents. The CAS must consume the exact pre-CAS activation-intent manifest and persist one post-CAS activation receipt in the same transaction as the accepted transition. Exact replay returns its byte-identical durable bytes without advancing either counter; conflict and unresolved commit-unknown attempts emit no activation receipt.
- **Remediation:** add a forward migration with an append-only authority-transition ledger and a tail sequence bound to the singleton. Each least-authority SQL capability must lock and compare the expected mode, activation epoch and transition sequence, append one intent/receipt-bound row, update the singleton and persist the canonical receipt core in the same transaction. Keep the existing generation table exclusively for accepted ACTIVE epochs and update readiness to verify it against ACTIVE transitions rather than treating pauses as activation rows. Activation consumes an immutable pre-CAS intent manifest bound to the prior retirement/pause receipt, applicable PAUSED backup/restore receipt, approval, candidate, target, fully qualified rollback candidate, incarnations and expected next counters; only a committed or exact replay returns the separate immutable post-CAS activation receipt.
- **Closure gate:** G14 with G8 concurrency and stale-writer evidence.
- **Rollback:** a failed forward migration leaves the target `LEGACY/0`. For an ambiguous later transition, block new risk externally and resolve the committed mode/epoch/sequence by intent/receipt readback. If the committed state is `ACTIVE`, keep fenced ingest/settlement/cancel/reconcile alive until finality and zero asynchronous work, then commit `PAUSED` at that activation epoch. Never edit an applied migration, decrement counters, return to V1, or force `ACTIVE` manually.

### ELVIS-V2-P0-004 — No supported B-only pause or V2 recovery control surface exists

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** the application port exposes only `activate()` at [`trading/application/paper_runtime_activation.py:148-162`](../../trading/application/paper_runtime_activation.py#L148-L162), and the PostgreSQL adapter implements only activation/replay at [`trading/persistence/paper_runtime_activation.py:75-179`](../../trading/persistence/paper_runtime_activation.py#L75-L179). No implemented pause or one-way retirement capability enforces the trajectory-B graph, while the operator dispatcher exposes only four migration/review commands at [`scripts/v2_operator.py:15-33`](../../scripts/v2_operator.py#L15-L33).
- **Impact:** an active or ambiguous runtime cannot be safely drained, reconciled and returned to a single known writer through a supported, auditable path.
- **Reproduction and expected regression:** enumerate the application/operator command surfaces; no authenticated pause or V2 recovery operation is callable. Regression must first set the kill switch while `ACTIVE`, continue ingest/settle/cancel, prove global source/applied tails equal and queues/gaps/quarantine/leases/non-terminal orders/holds are zero, then exercise idempotent `ACTIVE/N -> PAUSED/N`. The activation epoch remains unchanged while `authority_transition_sequence` advances. Commit-unknown resolves by exact readback; a later `PAUSED/N -> ACTIVE/N+1` requires a new approval. Concurrent, stale, premature-pause and every V1-reactivation request must fail.
- **Remediation:** implement least-authority offline preflight, pause, V2 restore/redeploy and activation commands with forward SQL capabilities, stable intent IDs and append-only receipts. Runtime/public APIs must not own authority transitions, and no command may grant V1 writer authority.
- **Closure gate:** G14 and G10, with G8 sole-writer proof.
- **Rollback:** for any ambiguous control transition, block submission independently and preserve/read back the append-only transition receipt. If the committed state is `ACTIVE`, keep fenced recovery processing alive, reconcile/drain to finality and zero asynchronous work, then execute the authenticated pause. The resolved terminal state is `PAUSED` at the committed activation epoch. Never jump directly to `LEGACY` or `ACTIVE`, decrement an epoch, or regrant V1 writer access.

### ELVIS-V2-P0-005 — The running process does not compose a V2 asynchronous owner

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** startup composes `LegacyPaperExecutionAdapter` and `OrderService` at [`main.py:617-624`](../../main.py#L617-L624). `PaperSubmissionPlanner` is only a protocol at [`trading/application/durable_submission.py:237-246`](../../trading/application/durable_submission.py#L237-L246), while the architecture records atomic owners as dormant at [`docs/V2_ARCHITECTURE.md:108-121`](../V2_ARCHITECTURE.md#L108-L121).
- **Impact:** the executable bot remains the compatibility runtime; V2 journals, activation evidence and readiness cannot govern an end-to-end asynchronous order. Calling it V2 production would be false.
- **Reproduction and expected regression:** boot `main.py --mode paper` and inspect the submitted adapter; it resolves to the legacy adapter. Regression must boot the packaged V2 entrypoint and prove a generation/candidate-bound command-admission owner, deterministic virtual venue and ordered fill projector are composed with distinct credentials. Rejection, zero fill, partial/multi-fill, cancel and restart fixtures must produce exact journal/account effects with no legacy writes.
- **Remediation:** create an explicit V2 composition root around asynchronous command admission, durable venue transport and one fill-by-fill projector; remove the legacy adapter and the terminal `PaperSubmissionPlan` from the production closure.
- **Closure gate:** G7 and G8, preceded by G5 admission.
- **Rollback:** if the defective composition is `ACTIVE`, set kill, keep its fenced recovery closure alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`; only afterward stop the uniquely named V2 project. If already non-active, leave it there. V1 data remains operationally read-only evidence and can never become a rollback writer.

### ELVIS-V2-P0-006 — Runtime DDL and database initialisation fail open

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** `init_db()` grants schema creation and executes `CREATE TABLE/INDEX IF NOT EXISTS` during application use at [`utils/paper_trade_db.py:47-140`](../../utils/paper_trade_db.py#L47-L140). It returns normally when no connection exists and absorbs DDL errors. The executor then logs success and sets `db_available = True` after any normal return at [`trading/execution/binance_executor.py:630-641`](../../trading/execution/binance_executor.py#L630-L641).
- **Impact:** a runtime identity has schema-mutation authority and can declare its database available after connection or migration failure, defeating least privilege and fail-closed startup.
- **Reproduction and expected regression:** make `get_conn()` return `None` or force a DDL failure; `_init_paper_trading_db()` currently completes without an exception and marks the database available. Regression must prove the V2 image contains/executes no runtime DDL and refuses startup before APIs/threads when exact migrated catalog evidence is absent.
- **Remediation:** move all DDL/GRANT/migrations to the offline migrator, give the runtime no CREATE privileges, replace `init_db()` with read-only admission, and propagate typed storage/admission failure.
- **Closure gate:** G2 and G5, including G11 database-failure injection.
- **Rollback:** if admission fails before activation, keep V2 stopped or `PAUSED`. If catalog/admission truth is lost while `ACTIVE`, set kill independently, keep only the still-trusted fenced recovery path alive, reconcile/drain to finality and zero asynchronous work, commit `PAUSED`, and only then stop and repair/rebuild through the offline operator. Do not reintroduce runtime DDL or shared credentials.

### ELVIS-V2-P0-007 — Graceful shutdown reads the wrong liquidation result contract

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** `signal_handler()` indexes `results["closed"]` at [`main.py:68-108`](../../main.py#L68-L108), while `close_all_positions()` initialises and returns `{"liquidated", "errors", "total_pnl"}` at [`trading/execution/binance_executor.py:926-940`](../../trading/execution/binance_executor.py#L926-L940) and appends to `liquidated` at [`:972-984`](../../trading/execution/binance_executor.py#L972-L984).
- **Impact:** SIGINT/SIGTERM can raise `KeyError` inside shutdown, skip accurate reconciliation/reporting, and proceed to reset session state without proof that positions closed.
- **Reproduction and expected regression:** invoke `signal_handler()` with an executor returning the actual `close_all_positions()` shape; current code accesses a missing key. Regression must cover zero, successful, partial and failed closes plus SIGTERM, require durable reconciliation before exit, and never reset/claim clean state after an error.
- **Remediation:** replace ad-hoc dictionaries with one typed shutdown receipt, separate pause/drain/reconcile from liquidation policy, and make signal handling idempotent and bounded.
- **Closure gate:** G10 and G11, with G14 pause semantics.
- **Rollback:** if graceful shutdown starts while `ACTIVE`, atomically block new risk with kill set and keep fenced ingest/settlement/cancel/reconciliation alive until finality. Commit `PAUSED` only after zero asynchronous work; if that cannot be proved, exit non-zero while still `ACTIVE` but unauthorised, preserve evidence, and require operator reconciliation. Do not force a false clean result.

### ELVIS-V2-P0-008 — No approved, atomic fresh V2 opening exists

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** `PostgresPaperAccountJournal.provision_account()` accepts only execution scope, owner generation and an account at [`trading/persistence/paper_account_journal.py:996-1013`](../../trading/persistence/paper_account_journal.py#L996-L1013). Its canonical opening envelope contains account/opening fields but no trajectory-B intent, approval identity, signature or trust-policy digest at [`trading/persistence/paper_account_journal_codec.py:342-353`](../../trading/persistence/paper_account_journal_codec.py#L342-L353). Exact retry is named `EXISTING`, not the required `REPLAYED`, at [`trading/persistence/paper_account_journal.py:1085-1090`](../../trading/persistence/paper_account_journal.py#L1085-L1090). The domain permits multiple non-negative opening assets at [`trading/domain/paper_accounting.py:296-310`](../../trading/domain/paper_accounting.py#L296-L310), and the current regression fixture provisions both BNB and USDT at [`tests/test_paper_account_journal.py:225-254`](../../tests/test_paper_account_journal.py#L225-L254).
- **Impact:** a caller can create a solvent-looking economic account without a separately authenticated approval of the exact fresh collateral asset and amount. Nothing atomically proves trajectory B or absence of V1 continuity; account-state `ACTIVE` can then be mistaken for trading authority.
- **Reproduction and expected regression:** provision an account today without an approval envelope, with multiple assets, or with a zero available collateral balance; the current primitive does not enforce the B contract. Regression must verify a canonical secret-free intent and detached signature that include positive `owner_generation` and the exact opening codec/version, require exactly one positive finite quantized collateral amount and no default/example/env/compatibility fallback, consume under lock a target-local nonce unique over `(trust_domain, signer_key_id, nonce)`, and atomically create one opening while authority remains `LEGACY/0/S0`. Within that physical target, the logical target is inside the signed intent digest: concurrent and lost-ack exact retries of the identical committed digest return byte-equivalent `REPLAYED`, including read-only resolution after approval expiry/revocation, while reuse under the same trust/key namespace with different bytes, account or logical target returns `CONFLICT`. The same raw nonce under another trust domain or signer key belongs to a different namespace in that registry. A separate physical database has a separate registry; the system makes no cross-database nonce-uniqueness claim, and neither opening grants authority until sole-writer admission selects one database incarnation. Expired/revoked approval before an absent/new mutation, an extra asset, or any V1-derived row blocks without partial state.
- **Remediation:** add forward-only append-only intent, approval, target-local nonce and provisioning-authority records without changing existing opening-codec golden hashes. Bind the resulting account opening to their exact digests in one least-authority target transaction and emit a separate physical target-admission receipt. Prove with a two-target negative fixture that target-local nonce registration never masquerades as global uniqueness and that both targets remain `LEGACY/0/S0`; only a separately approved activation may select one physical incarnation. Keep authority transition, candidate/runtime/model provenance and business opening distinct; insert no V1 balance, P&L, order, fill, fee, position, settlement or journal event.
- **Closure gate:** G3, followed by G4 no-continuity/post-opening replay evidence, G5 status truth and the separate G14 retirement/transition proof.
- **Rollback:** before activation, freeze evidence and rebuild only a fresh disposable target from the approved intent; never `UPDATE`/`DELETE` an opening. After activation, set kill, remain `ACTIVE` but unauthorised while accepted work drains to finality, and commit `PAUSED` at the unchanged activation epoch only after zero asynchronous work. Then restore a verified V2 backup or redeploy known-good V2 through G13/G14; V1 remains read-only.

### ELVIS-V2-P0-009 — PostgreSQL backup and full restore are neither implemented nor rehearsed

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** the bootstrap runbook declares a backup and rehearsed recovery path as a precondition at [`docs/V2_POSTGRES_BOOTSTRAP.md:159-167`](../V2_POSTGRES_BOOTSTRAP.md#L159-L167), while the importer explicitly states that neither its receipt nor importer owns a backup store or target lifecycle at [`docs/V2_LEGACY_SNAPSHOT_IMPORT.md:397-430`](../V2_LEGACY_SNAPSHOT_IMPORT.md#L397-L430). No V2 PostgreSQL backup/restore command or production restore runbook is present.
- **Impact:** operator error, corruption, host loss or a bad opening has no verified recovery point; activation would be irreversible in practice.
- **Reproduction and expected regression:** inventory the V2 operator commands and runbooks, then attempt a clean-cluster restore; there is no supported procedure. Regression must create an independently checksummed backup, restore it to a distinct PostgreSQL 15 system identifier, preserve the backed-up runtime mode, activation epoch, authority-transition tail and writer fence byte-for-field, emit only a new database-incarnation admission receipt, re-admit catalog, opening intent/approval and account state, and prove deterministic post-opening V2 replay/readiness without any V1 dependency or synthetic authority transition.
- **Remediation:** add encrypted, access-controlled backup policy and bounded backup/verify/restore runbooks/scripts, retention and monitoring, plus automated disposable restore drills and a recorded Kali rehearsal. Restore never rewrites mode/generation/transition/fence and never creates `PAUSED -> PAUSED` or `LEGACY -> PAUSED`; it remains isolated and non-authoritative until a separately approved activation.
- **Closure gate:** G13, then G5 readiness and G14 recovery rehearsal.
- **Rollback:** all restore development and drills run on a distinct disposable target. A failed verification destroys only that isolated target after preserving evidence. If backup confidence is lost while the source is `ACTIVE`, set kill, keep fenced recovery alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`; never overwrite the only database.

### ELVIS-V2-P0-010 — The installable V2 release contains no trading runtime

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** the operator image copies only bounded migration/review modules and dispatches `scripts.v2_operator` at [`deploy/v2/operator.Dockerfile:21-71`](../../deploy/v2/operator.Dockerfile#L21-L71). The dispatcher explicitly cannot start a runtime or authorise `ACTIVE` at [`scripts/v2_operator.py:1-5`](../../scripts/v2_operator.py#L1-L5) and exposes only four offline commands at [`:15-33`](../../scripts/v2_operator.py#L15-L33).
- **Impact:** users can install the preview but cannot install or run the intended V2 paper-production service; release/install success is not production evidence.
- **Reproduction and expected regression:** install the published bundle and enumerate commands/services; only the operator is available. Regression must install from a clean directory by immutable digest, start a separate paper-only V2 runtime image and dependencies, pass admission/health, execute a deterministic paper fixture, and cleanly stop without affecting unrelated stacks.
- **Remediation:** publish distinct minimal runtime and operator images plus a production paper bundle, each with least privilege, immutable references, SBOM/provenance/attestations and documented separation of authority.
- **Closure gate:** G0, G1 and G16 after all runtime P0 gates are green.
- **Rollback:** retain the operator preview as inactive tooling. If a failed runtime release is already `ACTIVE`, set kill, keep fenced recovery alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`; only afterward stop its uniquely named project and deploy a previously approved immutable V2 digest. The redeployed candidate remains `PAUSED` until a fresh pre-CAS activation-intent manifest is approved and its activation produces the post-CAS receipt.

### ELVIS-V2-P0-011 — V1 writer retirement and archive immutability are unproven

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** the current database fence installs write-blocking triggers only on seven compatibility relations in the target schema at [`trading/persistence/sql_migrations/0004_paper_runtime_control.sql:70-131`](../../trading/persistence/sql_migrations/0004_paper_runtime_control.sql#L70-L131). The archive contract test merely proves that selected compatibility documents still exist and mention compatibility at [`tests/test_v1_archive_contract.py:282-286`](../../tests/test_v1_archive_contract.py#L282-L286). Neither proves that the V1 process is stopped, existing sessions are drained, its writer login/secret is revoked, or retained database evidence is immutable.
- **Impact:** an old process, credential or already-open session may continue or resume writes while V2 becomes active. A mutable “archive” could also change the offline comparison baseline and invalidate incident or audit evidence.
- **Reproduction and expected regression:** retain an authenticated V1 session across the proposed fence, retry with the old process and credential, reconnect after restart, and attempt INSERT/UPDATE/DELETE/DDL/commit. Current evidence does not show these paths fail. Regression must stop new V1 work, drain or terminate its in-flight work/sessions, revoke writer login/CONNECT/DML/DDL, remove or rotate the writer secret, and prove every old session/process/credential write and reconnect fails. In G14 it must create a checksummed V1 archive manifest, independently verify every object, enforce the frozen WORM/retention policy, restore the archive into an isolated target, prove restored integrity, and run only the separately identified bounded read-only comparator with zero V2 effects. The qualification twin and production retirement receipts must bind these exact archive-proof digests.
- **Remediation:** implement an offline retirement capability and append-only receipt bound to the exact V1 process/database/archive identities, credential-revocation evidence, immutable archive manifest, integrity report, WORM/retention receipt, isolated-restore receipt and read-only-comparator receipt. G14 performs and retains that proof before production activation. Require the accepted retirement receipt plus fresh readback in the pre-CAS `PAUSED/0 -> ACTIVE/1` activation-intent manifest. It must never seed the V2 opening or claim accounting continuity.
- **Closure gate:** G14, with G8 stale-writer/session proof and G7 zero-side-effect comparison. G14 cannot pass until both the disposable and production-bound V1 retirement paths prove archive integrity, WORM/retention enforcement, isolated restore and comparator read-only behaviour.
- **Rollback:** if retirement or archive verification is incomplete or ambiguous before activation, remain in the last committed `LEGACY/0` or `PAUSED/0` state with V2 submissions disabled and repair the evidence boundary. If an unexpected `ACTIVE` state is observed, set kill, keep fenced recovery alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`. Never activate V2 speculatively, restore V1 writer access after a committed retirement, or modify retained evidence to make checks pass.

### ELVIS-V2-P0-012 — Durable submission is structurally terminal full-fill only

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** `PaperSubmissionPlan` requires at least one fill and exact completion of the instruction at [`trading/application/durable_submission.py:207-234`](../../trading/application/durable_submission.py#L207-L234). The atomic submission owner rejects a non-terminal stream at [`trading/persistence/atomic_paper_submission_owner.py:67-100`](../../trading/persistence/atomic_paper_submission_owner.py#L67-L100), while `paper_account_batch_manifests` requires a positive predeclared `fill_count` and exact terminal ranges at [`trading/persistence/sql_migrations/0003_paper_account_ledger.sql:164-252`](../../trading/persistence/sql_migrations/0003_paper_account_ledger.sql#L164-L252).
- **Impact:** accepted zero-fill orders, incremental partial/multiple fills, cancellation and restart mid-order are impossible to persist honestly. Emulating them as one complete batch would invent future facts and make 1B false.
- **Reproduction and expected regression:** construct an acknowledged order with no fill, a 0.25 then 0.75 fill, and a partial-fill cancellation; the current plan/owner/schema cannot commit those prefixes. Regression must persist every valid prefix independently and replay to the same terminal result without predeclaring fill count or final versions.
- **Remediation:** preserve old batch tables and owners as read-only historical/test primitives, remove them from the production dependency closure, and add forward async order/candidate/venue-input/event/settlement relations plus a progressive account-journal boundary.
- **Closure gate:** G7 and G10, with G2 exact catalog/ACL evidence.
- **Rollback:** leave the new runtime dormant and the target `PAUSED`; never weaken or rewrite migrations 0001–0006 or manufacture terminal fills to reuse the legacy batch schema.

### ELVIS-V2-P0-013 — Open orders reserve no account capacity

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** current account capacity and reserved margin are derived only from confirmed positions at [`trading/domain/paper_accounting.py:418-482`](../../trading/domain/paper_accounting.py#L418-L482). The existing `reserve_instruction()` protects order identity, not collateral exposure.
- **Impact:** two accepted but unfilled orders can overcommit the same collateral. A venue fill is then an external paper fact that local accounting may reject after the fact, producing an irreconcilable order/account split.
- **Reproduction and expected regression:** concurrently admit two individually affordable orders whose combined worst-case notional, slippage and fee cap exceed effective available collateral. Regression must serialize on an account-global risk stream so at most one is admitted; candidate, hold, order and outbox commit atomically. Each fill converts part of the hold into exact margin/fees, and every terminal order releases the remainder.
- **Remediation:** add immutable hold events and a replay-checked hold projection; compute spendable collateral as economic available minus all active holds. Bind maximum adverse price/slippage, fee cap/asset, quantity, TIF, policy and activation candidate into the admitted execution terms.
- **Closure gate:** G7 and G8, with G15 conservation/resource evidence.
- **Rollback:** set or confirm kill, retain every hold, keep fenced recovery alive and reconcile venue finality before releasing capacity. If the runtime was `ACTIVE`, drain all asynchronous work to zero and only then commit `PAUSED`. Never release a hold merely because delivery or a worker timed out.

### ELVIS-V2-P0-014 — No durable causal venue-input/event transport or atomic event projector exists

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** no migration defines a submission outbox, venue-input inbox, immutable source event log, delivery receipt, execution inbox, source/applied tail, finality watermark, lease fence or quarantine. `PostgresOrderPositionJournal.append_event()` can append a `ConfirmedFill` and advance position without account settlement at [`trading/persistence/order_position_journal.py:876-1086`](../../trading/persistence/order_position_journal.py#L876-L1086).
- **Impact:** delivery order can become economic order, cross-order FIFO/account versions can diverge, and a crash may split lifecycle/position from balances/postings/holds. Duplicate, conflicting or missing events have no authoritative resolution.
- **Reproduction and expected regression:** admit a submit input and prove it emits only its immediate acknowledgement, then require a later durable market-step or virtual-timer input for each partial fill. Race cancel against a due-fill input in both causal orders, deliver two orders' fills in reverse transport order, drop a causal prefix, duplicate each message, reuse an event ID/sequence with different bytes, and crash after every DML boundary. Regression must use separate gapless global venue-input and event sequences per scope/generation/incarnation plus a per-order ordinal. One fenced source owner consumes only `next_input` and atomically appends that input's immediate deterministic event batch; it never pre-appends a future fill. The projector applies only the contiguous event prefix and atomically commits lifecycle, finality, hold, receipt and tail for every event; a fill additionally commits position and account postings. When cancellation reaches terminal finality, every remaining scheduled action is cancelled. A conflicting economic event quarantines the stream and disables new risk; it is never skipped.
- **Remediation:** add forward-only venue-input/outbox, canonical venue source log, event/inbox, delivery/lease, venue-input/event global and per-order sequence, finality, scheduled-action cancellation, quarantine and progressive account-journal schemas with distinct least-authority roles. Make standalone fill append unreachable from production.
- **Closure gate:** G7, G8, G10 and G11, preceded by G2.
- **Rollback:** set the kill switch and preserve the raw causal stream. Keep fenced recovery alive and reconcile the exact missing/conflicting identity; if the runtime was `ACTIVE`, commit `PAUSED` only after finality and zero asynchronous work. Never reorder by arrival time, advance the tail over a gap, or delete a source event.

### ELVIS-V2-P0-015 — Recovery omits non-terminal orders and pause can strand accepted work

- **Severity:** P0
- **Status:** OPEN
- **Evidence:** `list_unresolved_submissions()` retrieves only `PENDING` and `RECONCILING` at [`trading/persistence/order_position_journal.py:1165-1207`](../../trading/persistence/order_position_journal.py#L1165-L1207), omitting `OPEN`, `PARTIAL` and `CANCEL_PENDING`. No durable pause cutoff/finality contract exists.
- **Impact:** restart can falsely report clean while venue-accepted exposure remains. A kill or pause that stops every worker can strand fills/cancellations and release capacity prematurely; a stale worker may also source effects after pause.
- **Reproduction and expected regression:** crash/restart in awaiting-acceptance, open, partial and cancel-pending states, including a fill/cancel race and a missing causal sequence. Regression must rediscover all non-terminal orders, keep new risk blocked, continue fenced ingest/settle/cancel/reconcile, and refuse the pause CAS or next activation until source/applied tails match and every queue, gap, quarantine, lease, hold and non-terminal order is zero.
- **Remediation:** replace the bounded unresolved query with a complete non-terminal recovery owner, implement two-phase kill/drain/pause, and bind venue/deployment incarnations plus worker lease epochs to every mutation.
- **Closure gate:** G9, G10, G14 and G15.
- **Rollback:** remain kill-set and not authorised while fenced recovery drains/reconciles. Commit `PAUSED` only after finality and zero asynchronous work. If finality cannot be proved, remain `ACTIVE` but unauthorised or in the already committed non-active state, exit non-zero with an alert and operator receipt; never claim a clean pause or reactivate.

## P1 — reliability, operations, packaging and observability risks

### ELVIS-V2-P1-001 — Activation role has broad direct table mutation privileges

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** bootstrap grants the activation role `SELECT` on every authority table and `UPDATE` on every activation-lock table at [`trading/persistence/postgres_bootstrap.py:4195-4214`](../../trading/persistence/postgres_bootstrap.py#L4195-L4214), in addition to owning/executing the security-definer activation functions.
- **Impact:** compromised activation credentials can attempt direct business-table or control-row updates outside the intended opening/transition receipt contracts, increasing blast radius and allowing the singleton to diverge from the append-only authority-transition tail.
- **Reproduction and expected regression:** connect as the activation role and issue direct UPDATE/INSERT/DELETE/TRUNCATE against each opening, activation, control and transition relation; current ACLs permit part of that surface. Regression must deny direct DML while the exact bounded fresh-opening, pause and activation functions can acquire required locks and complete. It must also prove the runtime cannot invoke authority functions and that the singleton always equals the latest `authority_transition_sequence` row.
- **Remediation:** give login roles EXECUTE only on the minimum `SECURITY DEFINER` capabilities, keep function/table ownership non-login, make opening and transition histories reject UPDATE/DELETE/TRUNCATE, and add exact role/ACL/catalog admission tests.
- **Closure gate:** G2 and G14 security/transition evidence.
- **Rollback:** if the privilege defect is found before activation, revoke the activation credential and remain `PAUSED`. If found while `ACTIVE`, use an independent operator identity to set kill, keep the fenced recovery path alive, reconcile/drain to finality and zero asynchronous work, commit `PAUSED`, and only then revoke the defective activation credential. Restore grants only through offline bootstrap after exact catalog review.

### ELVIS-V2-P1-002 — Readiness and post-opening V2 replay load unbounded histories into memory

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** position replay calls `fetchall()` for every stored order/event at [`trading/persistence/order_position_journal.py:475-524`](../../trading/persistence/order_position_journal.py#L475-L524); account replay does the same for all batches/settlements at [`trading/persistence/paper_account_journal.py:823-872`](../../trading/persistence/paper_account_journal.py#L823-L872). Readiness inventories and replays every account and position at [`trading/persistence/paper_account_readiness.py:1643-1725`](../../trading/persistence/paper_account_readiness.py#L1643-L1725) and [`:1780-1803`](../../trading/persistence/paper_account_readiness.py#L1780-L1803).
- **Impact:** history growth can exhaust memory, hold locks too long, exceed health timeouts and turn startup/activation into an availability incident.
- **Reproduction and expected regression:** seed a large but valid post-opening V2 history and measure readiness/replay memory, duration and lock hold time. Regression must enforce documented row/byte/time bounds, stream or checkpoint work, emit a typed blocked/quarantine result at limits, and remain restart-idempotent. No test may seed or replay V1 accounting into the V2 account.
- **Remediation:** add bounded pagination/streaming, checkpoints and explicit operational budgets without weakening full-prefix integrity checks.
- **Closure gate:** G4 post-opening replay and G15 B-only resource/SLO evidence.
- **Rollback:** on limit or timeout, abort/roll back the read transaction and make new-risk admission false. If `ACTIVE`, set kill, keep bounded fenced recovery alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`; if already paused, remain there. Never truncate history or skip unseen records.

### ELVIS-V2-P1-003 — No full side-effect-free V2 shadow path exists

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** the architecture lists V2 shadow comparison as pending at [`docs/V2_ARCHITECTURE.md:128-135`](../V2_ARCHITECTURE.md#L128-L135). Current `main.py` shadow hooks cover isolated RSI and take-profit observations, while the actual order path still submits through the legacy adapter at [`main.py:2420-2435`](../../main.py#L2420-L2435).
- **Impact:** there is no evidence that complete V2 decisions and risk outcomes can be compared with a frozen V1 baseline without creating effects. Under B, demanding equality of fees, balances, positions, P&L or account projections would falsely reintroduce an accounting-continuity requirement.
- **Reproduction and expected regression:** run current shadow modes and show that no full V2 decision/risk projection is produced for the same frozen market input. Regression must compare only explicitly versioned decision and risk fields under reviewed tolerances and prove zero submission, cooldown, feedback, journal, account or position mutation. It must assert that no V1 balance, fee, position, P&L or journal value enters V2 or becomes an activation prerequisite.
- **Remediation:** build a pure/frozen V2 decision comparator and bounded discrepancy store separate from the authoritative owner. Quarantine unexplained decision/risk divergence, but make no V1 accounting-parity or historical-continuity claim.
- **Closure gate:** G7 and G15.
- **Rollback:** disable the comparator only; authoritative V2 paper submission remains governed by `trading_authorized`, and unexplained decision/risk discrepancies remain quarantined without restoring V1 authority.

### ELVIS-V2-P1-004 — Runtime dependencies and several container images are mutable

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** most application dependencies are unconstrained in [`requirements.txt:1-49`](../../requirements.txt#L1-L49) and [`pyproject.toml:15-51`](../../pyproject.toml#L15-L51). Compatibility Compose uses mutable tags including `postgres:15-alpine`, `redis:7-alpine`, Prometheus/Grafana `latest` at [`docker-compose.yml:3-4`](../../docker-compose.yml#L3-L4), [`:83-97`](../../docker-compose.yml#L83-L97) and [`:111-114`](../../docker-compose.yml#L111-L114).
- **Impact:** two builds of the same commit can contain different code or images, invalidating tests, SBOMs, CVE decisions and rollback reproducibility.
- **Reproduction and expected regression:** resolve/build twice after upstream tag/index drift and compare locks/digests. Regression must install all production Python dependencies from a hash-locked closure and every production image by immutable digest, then reproduce the same manifest/SBOM from the release commit.
- **Remediation:** generate reviewed Python 3.14 lock files per image/architecture, pin base/service images by digest, and automate dependency refresh as an explicit audited change.
- **Closure gate:** G0 and G16.
- **Rollback:** if the mutable candidate is `ACTIVE`, set kill, keep fenced recovery alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`; only afterward redeploy the last approved immutable V2 runtime/bundle digests. Never roll back to an unrecorded mutable tag or reactivate without a fresh manifest/approval.

### ELVIS-V2-P1-005 — Root Compose is exposed, shared-credential compatibility topology

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** the root composition exposes PostgreSQL, Redis, bot, Prometheus, Grafana and Loki host ports at [`docker-compose.yml:3-21`](../../docker-compose.yml#L3-L21), [`:70-103`](../../docker-compose.yml#L70-L103) and [`:111-139`](../../docker-compose.yml#L111-L139). Bot and trainer share `elvis_user` credentials at [`:41-46`](../../docker-compose.yml#L41-L46) and [`:169-176`](../../docker-compose.yml#L169-L176).
- **Impact:** accidental use as “V2 production” expands attack surface, prevents role separation and can collide with existing Kali services.
- **Reproduction and expected regression:** render root Compose and inspect published ports, users and networks. Regression must render a distinct V2 bundle with internal-only dependencies, explicit bind addresses, dedicated identities/secrets, unique project resources and unchanged unrelated-host inventory.
- **Remediation:** keep root Compose clearly labelled legacy evidence and never a B rollback path; create a separate restrictive V2 production composition with external secret files, resource limits and no unnecessary published ports.
- **Closure gate:** G1, G2 and G16.
- **Rollback:** if the V2 project is `ACTIVE`, set kill, keep fenced recovery processing alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`; only afterward stop the uniquely named V2 project. Do not run global prune, restart Docker or alter unrelated projects/networks.

### ELVIS-V2-P1-006 — Metrics fabricate equity and suppress storage failures

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** `/metrics` hard-codes base equity `2000.0`, catches database failures and still returns HTTP 200 at [`trading/utils/trade_history_api.py:922-947`](../../trading/utils/trade_history_api.py#L922-L947). The executor's paper opening defaults to `100.0` at [`trading/execution/binance_executor.py:643-660`](../../trading/execution/binance_executor.py#L643-L660).
- **Impact:** dashboards and alerts can show plausible but false portfolio value while the database is unavailable, masking both economic-state drift and loss of submission authority. An economic account state `ACTIVE` can also be misread as runtime `trading_authorized`.
- **Reproduction and expected regression:** start from an approved non-2000 fresh opening, keep runtime `PAUSED`, or break database reads; the current metric remains 2000-based and scrape-successful. Regression must source equity only from the admitted fresh V2 account, expose account solvency, runtime mode, activation epoch, authority-transition sequence, `trading_authorized`, freshness and error as distinct fields, and align alert/health reason codes without leaking secrets.
- **Remediation:** remove all synthetic/default production values, export the exact approved-opening projection and explicit unavailable gauges/counters, and make dashboards visually distinguish economic solvency from runtime submission authority.
- **Closure gate:** G12 with G5 authority-state parity.
- **Rollback:** if authoritative metrics fail, publish unavailable/stale state, make `trading_authorized=false` and alert. When required observability is untrusted while `ACTIVE`, set kill, keep the still-trusted fenced recovery path alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`. Do not reinstate synthetic values.

### ELVIS-V2-P1-007 — Prometheus target does not match the Compose service topology

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** Prometheus is attached to `elvis-network` but scrapes `host.docker.internal:5050` at [`observability/prometheus.yml:4-13`](../../observability/prometheus.yml#L4-L13), while the bot is a named service on the same Compose network at [`docker-compose.yml:23-81`](../../docker-compose.yml#L23-L81). Linux host-gateway configuration is not defined for the Prometheus service.
- **Impact:** monitoring can silently fail or depend on host-specific routing even when the application service is reachable internally.
- **Reproduction and expected regression:** run the observability profile on a clean Linux host and query Prometheus target health. Regression must prove the pinned internal target is `UP`, then force application/DB loss and verify expected `DOWN`/not-ready alerts.
- **Remediation:** use the runtime service DNS name/internal port in the production scrape config, validate it with `promtool`, and add an E2E target/alert smoke.
- **Closure gate:** G12 and G1.
- **Rollback:** when authority monitoring is unavailable, make `trading_authorized=false`. If `ACTIVE`, set kill through the trusted independent path, keep fenced recovery alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`; if already paused, remain there. Revert only to a previously proven internal scrape configuration.

### ELVIS-V2-P1-008 — Performance tests are explicitly excluded from CI

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** `perf` is defined as an explicit marker at [`pyproject.toml:74-80`](../../pyproject.toml#L74-L80), and the main CI command excludes both `perf` and PostgreSQL tests at [`.github/workflows/ci.yml:86-90`](../../.github/workflows/ci.yml#L86-L90). PostgreSQL has separate jobs; no corresponding performance/SLO job is present.
- **Impact:** post-opening V2 replay, readiness, hot-path or API latency/memory regressions can merge without production-budget evidence.
- **Reproduction and expected regression:** introduce a controlled latency or large-history regression and observe the standard matrix remain green. Regression must run stable, bounded performance contracts on protected release paths and store thresholds/results with the soak evidence.
- **Remediation:** separate deterministic micro-budgets from noisy soak/load tests, add release gating for the former and scheduled/reviewed evidence for the latter.
- **Closure gate:** G15, with G4/G7 budgets where applicable.
- **Rollback:** if a new budget is flaky before activation, keep release blocked and fix the harness/threshold from measured evidence. If a frozen budget fails while `ACTIVE`, set kill, keep fenced recovery alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`. Do not silently remove or relax the gate.

### ELVIS-V2-P1-009 — Bot start/stop API changes only process-local fiction

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** `bot_state` is a process-local dictionary at [`trading/api/app.py:63-71`](../../trading/api/app.py#L63-L71). `/api/bot/start` and `/api/bot/stop` explicitly only mutate it rather than the bot at [`:198-245`](../../trading/api/app.py#L198-L245).
- **Impact:** authenticated operators can receive “started” or “stopped successfully” while the actual runtime authority and submission loop are unchanged.
- **Reproduction and expected regression:** call start/stop and inspect the durable runtime mode, activation epoch, transition sequence and owner; only the API dictionary changes. Regression must remove these control claims or make them read-only status views. The authenticated offline operator must prove `ACTIVE/N -> PAUSED/N` advances only `authority_transition_sequence`, while any later `PAUSED/N -> ACTIVE/N+1` requires a separate approval.
- **Remediation:** make runtime APIs read-only for authority, expose exact durable mode/epoch/transition tail plus `trading_authorized`, and reserve pause, V2 restore/redeploy and activation for the least-authority offline operator/orchestrator path. Expose no V1 resume/rollback operation.
- **Closure gate:** G12 and G14.
- **Rollback:** disable the misleading mutating endpoints. If V2 is `ACTIVE`, set kill through the offline authority, keep fenced recovery alive, reconcile/drain to finality and zero asynchronous work, commit the authenticated durable pause at the unchanged activation epoch, and only then stop the process. V1 remains operationally read-only and never regains authority.

### ELVIS-V2-P1-010 — No runtime-level crash/restart and commit-unknown E2E suite

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** component-level commit-unknown tests exist, but [G10](07-v2-production-e2e-matrix.md#g10--graceful-stop-crashrestart-and-commit-unknown-recovery) still requires PostgreSQL/runtime crash, restart and ambiguous-commit evidence. The executable process remains the legacy composition at [`main.py:617-624`](../../main.py#L617-L624).
- **Impact:** individually correct repositories may still double-submit, lose readiness, reuse a stale activation epoch/transition sequence or misreport state when the process/database dies between effects and acknowledgement.
- **Reproduction and expected regression:** interrupt the full runtime before commit, after commit/before acknowledgement, during health loss, and with SIGKILL; no packaged V2 harness currently exists. Regression must prove exact replay or quarantine, zero duplicate submission, unchanged activation epoch across pause, exactly one transition-sequence increment per accepted authority transition, correct readiness/`trading_authorized`, and deterministic recovery for each fault point. Around activation, the immutable pre-CAS manifest must resolve to no receipt before rollback, one canonically persisted digest-bound post-CAS receipt after commit, or `COMMIT_UNKNOWN` until exact readback; the receipt core and committed timestamp are written in the accepted CAS transaction, so a lost acknowledgement returns identical bytes and never manufactures a receipt or reuses an approval for another epoch.
- **Remediation:** add a disposable PG15/runtime fault harness with named failpoints, immutable intents and external observation of journal/account/order effects.
- **Closure gate:** G10 and G11, with G8 concurrency evidence.
- **Rollback:** every uncertain case blocks submission independently and resolves the exact intent/transition-tail state. If the resolved state is `ACTIVE`, keep fenced recovery alive, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED` at the unchanged activation epoch; if already `PAUSED`, remain there. Never pause before reconciliation, blind-retry a non-idempotent transition or return to V1.

### ELVIS-V2-P1-011 — Model approval lacks required training provenance

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** the registry records only version, artefact SHA-256, path, optional metrics, status and timestamp at [`core/models/model_registry.py:61-80`](../../core/models/model_registry.py#L61-L80). The training guide requires source-data provenance, configuration hash, code commit, Python/package environment, metrics and validation split at [`docs/UNIFIED_TRAINING_GUIDE.md:53-58`](../UNIFIED_TRAINING_GUIDE.md#L53-L58).
- **Impact:** an artefact can be labelled production without evidence that its data, features, environment and validation are causal and reproducible.
- **Reproduction and expected regression:** register and approve an arbitrary file with empty metrics; current registry accepts it. Regression must reject promotion unless a canonical signed/hashed provenance manifest, feature schema, validation evidence and runtime compatibility checks are present.
- **Remediation:** unify model registry and feature-manifest contracts around one immutable provenance envelope and explicit approval identity, then bind the selected digest into the release-candidate manifest, runtime admission and activation approval. Keep model provenance separate from the fresh-account opening intent/approval.
- **Closure gate:** G0, G5 and G7.
- **Rollback:** if model provenance fails before activation, demote the artefact and keep V2 stopped or `PAUSED`. If found while `ACTIVE`, set kill, keep fenced recovery alive, reconcile/drain to finality and zero asynchronous work, commit `PAUSED`, and only then deploy the last fully evidenced model candidate. Never auto-select a file by path/name or reactivate without a fresh manifest/approval.

### ELVIS-V2-P1-012 — Production TLS, restrictive HBA, secrets and network policy are not composed

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** the authoritative status explicitly lists dedicated credentials, restrictive database/network policy and fail-closed health as undeployed at [`docs/V2_ARCHITECTURE.md:128-135`](../V2_ARCHITECTURE.md#L128-L135), and the reconciliation runbook confirms production identities, SCRAM secrets, restrictive HBA and network policy are missing at [`docs/V2_LEGACY_SNAPSHOT_RECONCILIATION.md:438-443`](../V2_LEGACY_SNAPSHOT_RECONCILIATION.md#L438-L443).
- **Impact:** a runtime deployment would rely on rehearsal/compatibility trust boundaries rather than an admitted production identity and encrypted, restricted database path.
- **Reproduction and expected regression:** render available V2 Compose assets; none starts a production runtime with TLS/HBA/role-separated services. Regression must prove certificate verification, exact SCRAM roles, denied cross-role access, internal network isolation, secret-file permissions and rotation/restart behaviour on Kali.
- **Remediation:** add external secret/certificate provisioning contracts, server/client TLS verification, exact HBA and network policy, and bootstrap admission/readback without exposing secret values.
- **Closure gate:** G2 and G1, with G11 credential/network fault tests.
- **Rollback:** if credentials/network policy fail while `ACTIVE` but the fenced recovery channel remains trustworthy, set kill, reconcile/drain to finality and zero asynchronous work, then commit `PAUSED`; only afterward revoke/rotate identities and stop the isolated project. If the recovery channel is already untrustworthy, keep external admission blocked and preserve the unresolved `ACTIVE`-but-unauthorised evidence; re-establish a separately approved fenced recovery identity, reconcile/drain to finality and zero asynchronous work, commit `PAUSED`, and only then revoke/rotate the obsolete identities. Never fall back to plaintext/shared credentials.

### ELVIS-V2-P1-013 — Soak SLOs and resource budgets are not yet defined or evidenced

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** the historical architecture lists soak as pending. [G15](07-v2-production-e2e-matrix.md#g15--staged-24-hour-and-72-hour-paper-soak) requires frozen reliability, correctness, observability and resource thresholds but records no accepted result yet.
- **Impact:** “ran for a while” can be mistaken for production readiness despite leaks, lag, error accumulation, stale data or unrelated-host impact.
- **Reproduction and expected regression:** no exact B release commit currently has a reviewed 24-hour/72-hour report against accepted thresholds. Closure evidence must define and pass liveness, service-readiness, `trading_authorized`, zero duplicate/unsafe submissions, source/applied-tail convergence, zero gaps/quarantine/holds/non-terminal orders at exit, fresh-V2 reconciliation, latency, memory/CPU/disk, log growth, alert and unrelated-stack invariants, with V1 remaining operationally read-only throughout.
- **Remediation:** publish a machine-readable soak contract and evidence manifest, run 24-hour qualification then 72-hour acceptance on the pinned Kali deployment, and independently review exceptions.
- **Closure gate:** B-only G15 after G0-G14 are green.
- **Rollback:** any threshold breach sets kill and blocks new submissions. If the target was `ACTIVE`, remain active but unauthorised while fenced recovery drains accepted work to finality, then commit `PAUSED` without changing the activation epoch. Preserve logs/metrics/database and stop only the isolated V2 project if host safety is affected.

### ELVIS-V2-P1-014 — Async queue retention, backpressure and operator SLOs are undefined

- **Severity:** P1
- **Status:** OPEN
- **Evidence:** the repository has no production async venue-input/event tables or metrics, so it also has no frozen byte/row/age limits, retry/lease budgets, retention policy, queue-growth alerts or measured venue-worker resource envelope.
- **Impact:** a slow projector, poison message or retry storm can grow PostgreSQL without bound, exhaust disk/connections, hide settlement lag and destabilise unrelated services on Kali even when economic correctness remains intact.
- **Reproduction and expected regression:** delay the projector, inject duplicates/gaps and restart workers under a bounded workload. Regression must enforce reviewed queue byte/row/age ceilings, stop new risk before storage exhaustion, retain raw evidence, expose bounded-cardinality backlog/gap/finality/settlement metrics and recover to zero without skipping economic messages.
- **Remediation:** freeze machine-readable backpressure, retry/lease, retention, disk/headroom and latency contracts from a no-fault Kali baseline; add release micro-budgets plus 24h/72h evidence rather than changing thresholds after a failure.
- **Closure gate:** G11, G12 and G15.
- **Rollback:** keep the switch set, stop new-risk producers while retaining fenced consumers, preserve and reconcile the blocked prefix, and drain to finality and zero asynchronous work before committing `PAUSED`. Expand resources only through reviewed configuration/candidate changes. Never stop recovery consumers early or purge an unapplied economic event to regain readiness.

## P2 — operator-truth and maintainability debt

### ELVIS-V2-P2-001 — Kill-switch documentation claims nonexistent hot-loop wiring

- **Severity:** P2
- **Status:** OPEN
- **Evidence:** documentation says `is_trading_halted()` caches state “for the hot loop” at [`docs/utilities_monitoring.md:263-267`](../utilities_monitoring.md#L263-L267), but the actual submission path at [`main.py:2420-2435`](../../main.py#L2420-L2435) does not call it.
- **Impact:** operators may believe an unsafe control is effective and stop investigating after an HTTP success.
- **Reproduction and expected regression:** run a source-reference check for `is_trading_halted` and compare it with the order path. Documentation regression must be executable: either point to the exact enforced boundary/test after P0-001 closes or state clearly that the control is unavailable.
- **Remediation:** correct the current claim immediately and later link the authoritative halt design, reason codes and G9 evidence without duplicating implementation details.
- **Closure gate:** G16, but final closure also depends on ELVIS-V2-P0-001/G9.
- **Rollback:** if wiring changes, default documentation back to `ACTIVE: NO-GO`/control unavailable until source and tests agree.

### ELVIS-V2-P2-002 — Console dashboard starts with fabricated portfolio values

- **Severity:** P2
- **Status:** OPEN
- **Evidence:** `main.py` initialises the dashboard with hard-coded portfolio value, realised/unrealised P&L and leverage at [`main.py:637-651`](../../main.py#L637-L651).
- **Impact:** a fresh or disconnected operator view can display plausible financial state unrelated to the admitted account, confusing screenshots, course material and incident triage.
- **Reproduction and expected regression:** boot without an account/database and inspect the initial dashboard. Regression must show explicit unavailable/not-admitted state until an exact projection is loaded and must label fixture/demo mode unmistakably.
- **Remediation:** remove production mock seeds, consume one authoritative read model with economic-account state, runtime mode, activation epoch, transition sequence, `trading_authorized` and freshness as distinct fields, and keep demos in dedicated fixtures excluded from production composition.
- **Closure gate:** G12 and G16.
- **Rollback:** display `UNAVAILABLE` rather than restoring synthetic values if the projection cannot load.

### ELVIS-V2-P2-003 — REST monitoring API returns mock balances and positions

- **Severity:** P2
- **Status:** OPEN
- **Evidence:** `/api/account/balance` returns a hard-coded `10000/8500/1500` fixture at [`trading/api/app.py:248-272`](../../trading/api/app.py#L248-L272), and `/api/positions` returns a fabricated BTC position at [`:275-303`](../../trading/api/app.py#L275-L303) when cache entries are absent.
- **Impact:** operator, dashboard or course evidence can show fictitious holdings as if observed runtime state, undermining incident response and publication accuracy.
- **Reproduction and expected regression:** clear Redis/cache and query both authenticated endpoints; mock financial data is returned. Regression must return an explicit unavailable/empty authoritative response with activation-epoch, transition-sequence, `trading_authorized` and freshness metadata and never substitute examples in production mode.
- **Remediation:** bind monitoring endpoints to the admitted fresh-V2 read model, expose economic account state separately from runtime authority, move examples into tests/docs, and make fixture mode a distinct, impossible-to-confuse process.
- **Closure gate:** G12 and G16.
- **Rollback:** on read-model failure, return typed unavailable/503 and alert; never fall back to mock data.

## Register exit criteria

The register can be declared closed only when:

1. every entry is updated from `OPEN` to `CLOSED` with links to the exact fixing
   commit, regression test, CI run and gate-evidence artefact;
2. no P0/P1 closure relies solely on mocks, unit tests, generated receipts or
   documentation claims;
3. the exact trajectory-B release digest passes G0-G16, including the
   independently reviewed Kali install, failure injection, backup/restore,
   V2-only recovery and soak evidence;
4. the separate course/media lineage passes G17; public screenshots/video show
   only the verified release and redact private infrastructure;
5. the fresh-opening intent and approval remain distinct from candidate and
   activation approval, and all are bound to their exact receipts;
6. every activation uses one immutable pre-CAS activation-intent manifest and
   one immutable post-CAS activation receipt; initial activation binds the
   accepted V1-retirement receipt, later activation binds the accepted pause
   receipt, and kill-clear is never a standalone mutation;
7. the final `PAUSED/N -> ACTIVE/N+1` paper authority transition and final
   course render receive their separately required approvals; and
8. V1 writer retirement/read-only archive evidence remains valid and no release,
   runbook or recovery path can restore V1 authority or claim V1 accounting
   continuity.
