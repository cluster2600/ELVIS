# ELVIS target architecture

## Decision

ELVIS will evolve into a **modular monolith with a synchronous, deterministic
trading path**. Domain and application code will depend on small ports;
Binance, PostgreSQL, model files, clocks, notifications, and dashboards will be
adapters. Events will describe completed state transitions for observability,
not control whether an order is safe to submit.

This is intentionally smaller than NautilusTrader's actor system, Hummingbot's
connector estate, or OpenMarket's service topology. ELVIS currently trades a
small symbol set through one process. Network service boundaries would add
serialization, deployment, and failure modes without removing the immediate
coupling inside the process.

## Design principles

1. **Fail closed.** Missing, stale, non-finite, invalid, or exceptional inputs
   produce `HOLD` or a rejected order, never synthetic live data or fallback
   leverage.
2. **One owner per state transition.** A single application service owns each
   decision, order submission, fill reconciliation, and position exit.
3. **Typed boundaries.** Domain objects replace ad hoc dictionaries and
   positional tuples at module boundaries.
4. **Direct hot path.** The decision and risk path uses ordinary synchronous
   calls with no global bus, reflection, or unnecessary queues.
5. **Ports at I/O boundaries.** Protocols exist for behaviour with multiple
   implementations or external effects, not for every function.
6. **Research/live parity.** Paper, replay, and live modes invoke the same
   application services; only adapters and clocks change.
7. **Progressive replacement.** Legacy adapters keep current functionality
   running while one vertical slice moves at a time.
8. **Measured performance.** Optimise only a measured hot path. Prefer fewer
   network calls, cached immutable snapshots, and bounded allocations before a
   language rewrite.

## Logical component map

```mermaid
flowchart TD
    subgraph Interfaces
      CLI["CLI / runtime control"]
      API["trade API"]
      TUI["console dashboard"]
    end

    subgraph Runtime
      LIFE["lifecycle state machine"]
      LOOP["clocked trading runner"]
      COMPOSE["explicit composition root"]
    end

    subgraph Application
      CYCLE["TradingCycle"]
      SIGNALS["SignalPolicyPipeline"]
      PLAN["PreTradeRiskService"]
      ORDERS["OrderService"]
      POSITIONS["PositionService"]
    end

    subgraph Domain
      TYPES["MarketSnapshot / Signal / OrderIntent"]
      STATE["Order / Fill / Position state"]
      RULES["invariants and pure calculations"]
    end

    subgraph Ports
      MARKET["MarketDataPort"]
      MODEL["StrategyPort"]
      PORTFOLIO["PortfolioSnapshotPort"]
      EXEC["ExecutionPort"]
      REPO["TradeRepository"]
      CLOCK["Clock"]
      SINK["TelemetrySink"]
    end

    subgraph Adapters
      BINANCE["Binance market + execution"]
      PAPER["paper/replay broker"]
      PG["PostgreSQL repositories"]
      MODELS["versioned model artefacts"]
      OBS["logs / metrics / notifications"]
    end

    CLI --> LIFE --> LOOP --> CYCLE
    COMPOSE --> LOOP
    CYCLE --> SIGNALS --> PLAN --> ORDERS
    ORDERS --> POSITIONS
    CYCLE --> TYPES
    SIGNALS --> RULES
    PLAN --> RULES
    POSITIONS --> STATE

    CYCLE --> MARKET
    CYCLE --> MODEL
    PLAN --> PORTFOLIO
    ORDERS --> EXEC
    POSITIONS --> REPO
    LOOP --> CLOCK
    CYCLE -. "post-transition only" .-> SINK

    BINANCE --> MARKET
    BINANCE --> EXEC
    PAPER --> EXEC
    PG --> REPO
    MODELS --> MODEL
    OBS --> SINK
    API --> REPO
    TUI --> REPO
```

Dependency direction is inward: domain imports no ELVIS infrastructure;
application imports domain and port definitions; adapters implement ports;
runtime composes them. UI and observability never call the exchange executor.

## Canonical trading cycle

```mermaid
sequenceDiagram
    participant R as TradingRunner
    participant D as MarketDataPort
    participant S as StrategyPort
    participant G as SignalPolicyPipeline
    participant K as PreTradeRiskService
    participant J as JournaledOrderService
    participant P as DurableOrderJournal
    participant E as OrderService
    participant X as ExecutionPort
    participant L as PositionService
    participant O as TelemetrySink

    R->>D: snapshot(symbol, deadline)
    D-->>R: fresh MarketSnapshot or typed failure
    R->>S: decide(snapshot)
    S-->>R: Signal
    R->>G: evaluate(signal, snapshot, account)
    G-->>R: approved Signal or HOLD with reasons
    R->>K: plan(approved signal, portfolio snapshot)
    K-->>R: rejected or approved RiskDecision
    R->>J: submit(PositionInstruction)
    J->>P: register PENDING instruction
    P-->>J: committed reservation
    J->>E: submit(instruction.order_intent)
    E->>X: submit once with client order ID
    X-->>E: SubmissionReport
    E-->>J: SubmissionReport
    J->>P: append submission observation
    P-->>J: durable journal disposition
    J-->>R: report plus journal disposition
    R-->>O: publish completed submission outcome and timings
    X-->>L: confirmed order/fill or reconciliation result
    L->>P: load ordered position stream
    P-->>L: persisted facts
    L->>L: apply lifecycle and position reducers
    L->>P: append validated event at next position version
    P-->>L: committed event
    L-->>O: publish completed position transition
```

No exception crosses the cycle boundary without becoming a typed failure and a
non-actionable outcome. If execution times out after a possible submission, the
`SubmissionReport` is `AMBIGUOUS`, not a definite rejection; reconciliation
must resolve it before any retry. A `SUBMITTED` report records an acknowledgment,
not a fill. Only `PositionService` applies confirmed fill and position
transitions.

## Domain contracts

Successive migration slices introduce small immutable values:

- `SignalAction`: `BUY`, `SELL`, or `HOLD`;
- `OrderSide`: `BUY` or `SELL`, so `HOLD` is unrepresentable in an order;
- `Signal`: symbol, action, confidence, reference price, timestamp, strategy ID,
  and reasons;
- `OrderIntent`: client order ID, decision ID, symbol, order side, quantity,
  order type, reference
  price, leverage, and decision timestamp;
- `RiskDecision`: approved/rejected plus reasons and optional order intent;
- `SubmissionReport`: `NOT_SENT`, `VENUE_REJECTED`, `SUBMITTED`, or `AMBIGUOUS`,
  with client/venue IDs and a retry-safety decision kept separate from outcome;
- `OrderLifecycleState`: pending/reconciling/open/partial/filled,
  cancel-pending/cancelled/failed, projected only through validated immutable
  events;
- `PositionEffect`: explicit `OPEN` or `REDUCE_ONLY`, coupled to one approved
  order intent before submission;
- `PositionSide`: `LONG` or `SHORT`, deliberately distinct from order direction;
- `TakeProfitProfile` and `PositionExitContext`: the resolved entry-time exit
  policy retained for the lifetime of a position key;
- `Position`: an immutable projection of confirmed fills with exact opened,
  reduced, and remaining quantities; and
- `CycleOutcome`: terminal result and per-stage timings.

M2 implements `SignalAction`, `OrderSide`, `Signal`, the market-only
`OrderIntent`, and `SubmissionReport`. M7a adds the correlated `RiskDecision`
contract. M8a adds the pure `OrderLifecycle` reducer without wiring it into the
runtime. M8b adds pure `PositionInstruction`, `PositionFill`, and `Position`
transitions without a runtime consumer. M9b.1 prepares the durable journal
schema, M9b.2 adds its pure, lossless persistence codec, and M9b.3 adds the
unwired transactional repository and reducer-based replay boundary. M9b.4 adds
the application-level `JournaledOrderService`, still with no runtime consumer.
`PositionService`, the pre-trade service, and `CycleOutcome` remain later
slices; they are not placeholder classes in the current package.

Constructors validate symbol presence, finite positive prices and quantities,
confidence bounds, non-negative fees, legal state transitions, and timezone-
aware timestamps. Invalid values raise before reaching an adapter.

Model features and confidence may remain ordinary `float`. At the order,
balance, fee, and realised-PnL boundaries, values use `Decimal` constructed from
strings and quantised against venue tick/lot rules. This is a narrow conversion
at the safety boundary, not a broad rewrite of pandas and model calculations.

## Application services

### `TradingCycle`

Owns one symbol decision from a market snapshot to a terminal outcome. It has no
loop, sleeps, environment reads, database globals, or UI calls. It is cheap to
construct and deterministic under injected ports and clock.

### `SignalPolicyPipeline`

Applies named policies in a fixed order and records every reason. A policy may
approve, adjust confidence, or veto to `HOLD`. An unexpected exception vetoes
the trade. Expensive optional policies have explicit deadlines and cannot block
the core loop indefinitely.

### `PreTradeRiskService`

Reads one coherent portfolio snapshot and produces either a rejection or one
fully specified `OrderIntent`. It owns leverage limits, exposure, cooldown,
stake sizing, fee viability, stale-account checks, and portfolio kill switches.
It never submits an order. `PortfolioSnapshotPort` is only its read boundary to
account and position state; risk policy remains inside this application service,
not in a second interchangeable `RiskPort`.

### `OrderService`

Makes one execution-adapter call per invocation for an already-approved intent,
using a stable client order ID, and never retries internally. It distinguishes
proof that nothing was sent, a definite venue rejection, a submission
acknowledgment, and an ambiguous outcome. Ambiguous submissions are reconciled
and never blindly retried. Durable idempotency and restart reconciliation are
added with the order repository; an in-memory deduplication cache would provide
false safety. The service does not invent quantity, leverage, price, or balance
fallbacks.

M9b.5 closes only the transport-correlation gap in the still-authoritative
paper path. `LegacyPaperExecutionAdapter` passes the stable client order ID as
a keyword-only argument to `BinanceExecutor`, and accepts a nominal paper
`FILLED` response as a submission acknowledgement only when the response echoes
that exact ID. The mock venue order ID is opaque and collision-resistant rather
than derived from wall-clock seconds. This echo proves which invocation
produced one response; it is not a durable receipt, a fill fact, deduplication,
or a query-by-client-ID implementation. Legacy executor callers that do not
provide a client ID remain supported but cannot satisfy the typed adapter's
correlation check.

### `JournaledOrderService`

M9b.4 supplies the unwired durable wrapper. It accepts one
`PositionInstruction`, asks a storage-neutral `OrderJournalPort` to commit its
reservation, and treats an exact boolean `is_created=True` receipt as the only
permission to call `OrderService`. Any existing reservation returns a
conservative reconciliation-required result without reading the clock,
submitting, or appending. A registration failure also makes no adapter call.

After a new reservation is committed, the service reads and validates one
injected aware timestamp before the external effect, delegates the embedded
`OrderIntent` exactly once, translates the `SubmissionReport` into the matching
M8a submission event, and appends it under the stable per-client identity
`submission-attempt-1`. `SUBMITTED` remains an acknowledgement even if a legacy
`venue_status` string says `FILLED`; only an independent `ConfirmedFill` may
advance position quantity. The service returns `RECORDED` only after the event
receipt confirms the expected identity and a positive position-stream version.
If append fails or its commit acknowledgement is unknown, the typed
`SubmissionObservationNotRecorded` preserves the exact report, event, event ID,
and original cause for reconciliation. It never retries the venue call.

The application module depends only on domain values and structural ports; it
does not import the PostgreSQL repository. Its package facade re-exports the
contract, so another `trading.application` import may load the pure module, but
no runtime, adapter, startup, or composition-root module references or invokes
it. M9b.1 supplies the tables, M9b.2 the typed codec, and M9b.3 the committed
reservation and append implementation. Activation still requires a truthful
pre-submit `PositionInstruction`, stable client-order propagation through the
side-effect owner, fill observation, unresolved-order inventory,
reconciliation, durable quarantine, and startup readiness.

### `PositionService`

Is the sole owner of open-position transitions, stop loss, take profit,
trailing stop, partial fill, and close reconciliation. The background risk
thread and inline exit loop are retired only after this service covers their
behaviour. M8b supplies its pure confirmed-fill transition contract: an `OPEN`
fill can create or scale a stable position key, while `REDUCE_ONLY` can only
reduce the opposite side and can never create or flip a position. That reducer
does not yet constitute this service, select an exit, calculate cost basis or
PnL, or perform runtime I/O. The persistence codec and repository replay these
instruction and event values, but no runtime module applies the reducer yet.

## Runtime and configuration

The runner has explicit `STARTING`, `RUNNING`, `PAUSED`, `DEGRADED`, `STOPPING`,
and `STOPPED` states. It uses an injected monotonic clock for deadlines and a
wall clock only for exchange timestamps.

Environment and YAML inputs are parsed once into a frozen configuration object.
Validation includes mode, symbols, thresholds, leverage ceiling, database
settings, model schema, and required adapter credentials. Code below the
composition root does not call `os.getenv`.

Paper is the default. Live startup requires an explicit mode, validated
credentials, a healthy account snapshot, a compatible model artefact, and no
active migration shadow mismatch. It also requires an execution adapter that
explicitly declares and passes a live-submission capability check; the current
paper-only Binance executor cannot satisfy this gate.

## Model and feature contract

Every deployable model artefact must carry:

- `schema_id` and schema version;
- ordered feature names and dtypes;
- preprocessing/scaler version;
- model kind and library version;
- training source identifier and source commit;
- training window and random seed;
- evaluation metrics and acceptance thresholds; and
- creation timestamp and content hash.

Inference resolves features by the artefact's declared schema, validates all
values, and rejects incompatible artefacts at load time. Padding/truncating an
unknown schema is not permitted. A deliberate optional feature set, such as the
9-versus-11 social-feature variants, receives distinct schema IDs.

## Storage

Repositories return named records, not positional tuples. Schema migrations
create the namespace and tables before health is declared ready. Order, fill,
and position transitions share a transaction where needed. Client order IDs
are globally stable, while venue order IDs are unique only within an explicit
execution scope and symbol; a bare venue ID is not a cross-account identity.

M9a introduces the packaged, checksummed, forward-only migration runner and an
additive, schema-only baseline for the existing legacy tables. It neither seeds
paper capital nor accepts an incompatible pre-existing layout as migrated. It
also rejects migration-owned transaction control and verifies the durable ledger
write before commit. It is deliberately not wired to startup yet; the isolated
PostgreSQL harness now validates the boundary, while an operator migration
command must still be in place before it can become a readiness prerequisite.
Order and position repositories remain later slices.

M9b.1 adds a separate, schema-only version 2 without altering or backfilling
the legacy `trades` and `open_positions` tables. `position_streams` provides a
globally unique stable key and future per-position lock/version boundary.
`orders` stores a versioned `PositionInstruction` envelope and its payload hash
before external submission. One decision ID may reserve only one order within
an execution scope. `order_events` reserves a positive, per-position version
key for future causal allocation; `(client_order_id, trade_id)` remains the
confirmed-fill identity. The schema scopes venue identities by execution scope
and symbol, and stores versioned payloads as JSON objects.

The bounded indexed columns are a persistence representability contract, not a
new domain invariant. SQL rejects empty and ordinary space-padded identifiers;
the persistence boundary remains responsible for the domain's complete clean-
text and storage-length rules.

M9b.2 supplies a pure, version-1 codec for `PositionInstruction` and all seven
M8a lifecycle events. `trading.persistence.journal_codec` exposes immutable
`EncodedPositionInstruction` and `EncodedOrderLifecycleEvent` records plus
their explicit encode/decode functions. It serializes exact `Decimal` values
and leverage as strings, aware timestamps in canonical UTC with a `+00:00`
offset and six fractional digits, and enums and optional values under an exact
JSON object shape. Canonical JSON uses sorted keys, compact separators, and
ASCII escaping; its UTF-8 bytes determine the stored SHA-256 payload digest.
That digest is evidence of corruption or inconsistency, not an authenticity
guarantee or MAC. Unknown versions, malformed envelopes, and payload/hash or
duplicated-column conflicts fail closed with `JournalQuarantineError`; they are
quarantine inputs and are never coerced into domain values. Version 1 is
immutable. Any change to its keys, types, canonicalization, or hash contract
requires a new envelope version and a matching schema migration.

M9b.3 adds `PostgresOrderPositionJournal` as the only database consumer of the
codec and reducers. Each public operation obtains a fresh connection from an
injected factory and owns its transaction. Registration and append return only
after PostgreSQL acknowledges commit; an unacknowledged commit becomes the
typed `JournalCommitUnknown` outcome and is never reported as success. Exact
retries compare the complete canonical version-1 envelopes rather than domain
numeric equality, so `Decimal("1.0")` and `Decimal("1.00")` cannot silently
change a durable fact. Writers lock one `position_streams` row, replay and
validate the complete stream, resolve duplicate event and fill identities,
allocate `stream_version + 1`, update historical venue correlation, and append
the event in one transaction. Public replay uses one repeatable-read, read-only
snapshot and requires event versions to equal exactly `1..stream_version`.
Every event is decoded and applied first to its `OrderLifecycle`, then every
confirmed fill is applied to the M8b position reducer in global
`position_version` order. Corrupt, gapped, or reducer-invalid history fails the
whole replay with no partial projection.

The concrete repository remains an unwired persistence boundary. M9b.4 now
provides the single register-before-submit application owner through structural
ports, but no runtime module composes it and no startup readiness or
reconciliation path exists. Failed replay is detected and typed, but it is not
yet durably quarantined because the schema and operational workflow for
quarantine are deliberately deferred. Each append currently replays the
complete stream before validation and again after insertion; that cost is
acceptable for an unwired correctness slice, but bounded replay or snapshots
are required before activation. M9b.5 prevalidates and echoes the domain-sourced
client order ID at the legacy paper transport boundary and emits a bounded
opaque mock venue order ID. A future fill source must still guarantee clean
Unicode and the schema limit of 255 characters for every trade ID it emits;
detecting an unrepresentable identifier only after an external effect is not an
acceptable activation boundary. The codec rejects NUL characters, isolated
Unicode surrogates, and over-length identifiers, but this validation cannot
retroactively make an already accepted venue identifier safe. Later M9b slices
must add an atomic paper receipt, query/reconciliation, durable quarantine,
runtime composition, and the remaining activation gates before
`PositionService` can own the runtime boundary.

Read models for API/dashboard use separate repository methods or immutable
snapshots so presentation queries cannot mutate trading state.

## Events and observability

The current global event bus is not extended. During migration, a narrow
`TelemetrySink` receives immutable post-transition notifications. Failures in
metrics, logging, dashboard, or notification adapters are visible but cannot
change a completed trading decision or trigger a second submission.

If later profiling proves that multi-process isolation or an actor runtime is
needed, the typed domain messages and ports provide a clean boundary. That
decision is deferred until ELVIS has a measured throughput or fault-isolation
need.

## Test architecture

| Layer | Scope | External I/O |
|---|---|---|
| Domain unit | invariants, state transitions, pure calculations | none |
| Application unit | cycle, policy, risk, idempotency with fakes | none |
| Adapter contract | each exchange/repository/model against shared contract tests | controlled |
| Integration | PostgreSQL migration, paper broker, model artefact loading | ephemeral local services |
| Replay | same cycle over frozen market fixtures | none |
| Live canary | read-only market/account health, then owner-approved paper/live scope | explicit only |

Tests default to blocked network access. Time, IDs, model outputs, venue replies,
and database state are injected. Source-text tripwires are replaced gradually
with behaviour tests at the new application boundary.

## Performance budget

Initial budgets are tripwires, not marketing claims:

- domain validation and application orchestration overhead: p99 below 1 ms in
  an explicit 10,000-iteration warmed run against no-op in-memory adapters,
  excluding model and I/O work;
- no more than one account snapshot and one market snapshot per symbol cycle;
- no network or database call inside a pure policy;
- all external calls have explicit deadlines;
- per-stage monotonic timings on every cycle; and
- optimisation or Rust extraction only after a recorded profile identifies a
  CPU-bound stage whose cost matters relative to network latency.

This targets low latency by reducing I/O and uncertainty first, rather than by
introducing concurrency that makes order ownership harder to reason about.
Every recorded result includes CPU model, operating system, Python version,
sample count, warm-up count, clock (`perf_counter_ns`), and whether garbage
collection was enabled; it is a regression tripwire, not a cross-machine claim.

## Explicit non-goals

- no wholesale Rust rewrite;
- no microservice split during the current migration;
- no generic framework or plugin system for hypothetical exchanges;
- no replacement of working strategies solely for stylistic consistency;
- no automatic activation of live trading; and
- no claim that architectural quality creates trading alpha.
