# Reference architecture analysis

## Method and limits

The four repositories were cloned read-only and reviewed at the revisions in
the [migration index](README.md). The analysis covers their runtime topology,
domain and adapter boundaries, order/risk lifecycle, persistence, replay and
backtest model, configuration, tests, operations, and latency characteristics.

This is a source comparison, not a popularity comparison. A mature project can
contain useful patterns and unsuitable complexity at the same time. Concepts
were evaluated against ELVIS's current scale and constraints; no source code was
copied.

Local validation performed during the audit:

- OpenMarket: `cargo test --workspace` built the workspace; 126 tests passed,
  one was ignored, and one calibration test failed on an `Option::unwrap()`;
- NautilusTrader: targeted Rust tests passed for execution-failure taxonomy,
  order transitions, and risk configuration; the full Rust/PyO3 workspace was
  not built;
- Hummingbot: static source and test topology were inspected without building
  its Conda/Cython environment; and
- Freqtrade: `compileall` passed; its full Python >=3.11 dependency environment
  was not installed in the reference clone.

## Executive comparison

| Criterion | OpenMarket | NautilusTrader | Hummingbot | Freqtrade | ELVIS decision |
|---|---|---|---|---|---|
| Primary shape | Rust services/workspace | Rust-native event engine with Python API | layered Python/Cython monolith, V1+V2 | modular Python monolith | smaller modular Python monolith |
| Hot-path model | Tokio services over local WebSockets | single-process message bus and engines | clock ticks, pub/sub, controller/executor polling | candle-aligned polling loop | direct synchronous cycle; bounded async I/O only |
| Domain model | typed messages, uneven lifecycle | very rich typed orders/fills/positions | explicit in-flight orders and events | durable ORM trade/order model | small immutable domain contracts |
| Risk | thin and trusted upstream | mandatory pre-trade risk engine | budget checks; global risk incomplete | protections, locks, sizing, limits | mandatory fail-closed pre-trade risk service |
| Persistence/restart | raw market journal strong; live orders weak | cache/event store/persistence components | SQL recorder and order restore, singleton-heavy | mature orders/trades/migrations/reconcile | named repositories, migrations, idempotency |
| Research/live parity | duplicated and version-drifted | central design goal, same architecture | partial V2/backtest reuse | strong strategy-interface reuse, candle limitations | one application core, adapter-only mode changes |
| Timing/replay | source+ingest timestamps, datasets | clocks, deterministic simulation, event ordering | live/sim clocks but polling and wall time | OHLCV replay, not tick-low-latency | monotonic live clock + virtual replay clock |
| Extensibility | fixed venues/services | extensive typed adapter crates | very broad connector estate | dynamic resolvers/plugins | explicit registry for proven needs only |
| Operational maturity | mixed; archived and deploy docs drift | very high but correspondingly complex | mature CLI/connectors, mixed internals | mature lifecycle/API/packaging | adopt narrow lifecycle and health patterns |
| Main complexity risk | premature distribution | framework scale far beyond ELVIS | V1/V2 inheritance and singletons | large `FreqtradeBot` and config dictionaries | ports without framework-building |

None of the four repositories is a template to transplant. The target combines
NautilusTrader's contracts and clocks, Hummingbot's order tracking, Freqtrade's
durable recovery and operational lifecycle, and OpenMarket's data provenance
and artefact reproducibility.

## OpenMarket

### Observed architecture

OpenMarket is an Apache-2.0 Rust workspace of 13 crates. Separate Axum services
collect Binance and Polymarket streams, record and pair events, generate
signals, and optionally execute or paper-trade them. Local service integration
uses WebSocket JSON.

```text
Binance WS -> exchange-binance :8001 ---\
                                         -> signal-engine :8003 -> execution-engine
Polymarket -> exchange-polymarket :8002 -/
                    \
                     recorder :8005 -> SQLite / Parquet export
```

Evidence:

- workspace membership: `Cargo.toml:1-16`;
- collector/service startup: `crates/exchange-binance/src/main.rs:13-53` and
  `crates/exchange-polymarket/src/main.rs:11-49`;
- signal service: `crates/signal-engine/src/main.rs:147-185`;
- recorder: `crates/recorder/src/main.rs:6-70`; and
- reproducibility identifiers: `docs/reproducibility.md:3-8`.

### Useful ideas

- preserve both source and ingest timestamps;
- retain the raw payload beside the normalised market event;
- bound ingestion channels and batch writes into SQLite WAL;
- publish replayable Parquet data with dataset metadata;
- report source commit, dataset, model, and configuration for a result;
- let a model artefact declare ordered feature names and reject missing
  features (`crates/signal-engine/src/calibrated.rs:201-307`); and
- expose freshness/health per upstream source.

### Costs and defects not to inherit

The topology is distributed before its interfaces are fully reliable. It pays
JSON serialization and service failure costs on one host, duplicates some
Polymarket ingestion in the recorder, and has no end-to-end correlation or
proven benchmark despite performance targets.

Live order and position state is mainly in memory. There is no durable order
state machine or idempotency key. A failed take-profit sell can still mark the
local position closed (`crates/execution-engine/src/main.rs:873-899`). The
executor trusts upstream risk, example paper configuration is not consistently
implemented, and live, paper, and backtest sizing/signal versions have drifted.

The current Docker file does not reproduce the complete documented service
graph. The project is also explicitly an archived research record, and its own
reported out-of-sample economics are negative. It is valuable precisely because
it documents reproducibility and null results honestly, not because its live
executor should be reused.

### ELVIS decision

Adopt data provenance, replay discipline, bounded buffers, batch storage, and
feature metadata. Keep them in-process until profiling or fault-isolation needs
justify a service boundary. Reject in-memory-only order ownership, silent model
disablement, duplicate live/backtest logic, and a wholesale Rust rewrite.

## NautilusTrader

### Observed architecture

NautilusTrader is an LGPL-3.0 production trading framework. Its current core is
a Rust workspace exposed to Python through PyO3. Component crates separate the
domain model, common infrastructure, data, execution, risk, portfolio,
backtesting, live runtime, persistence, serialization, and venue adapters. The
container crate documents these boundaries in `crates/README.md:1-31`.

```text
TradingNode / BacktestEngine
           |
           v
     MessageBus + Clock + Cache
           |
   +-------+--------+----------+
   v                v          v
DataEngine      RiskEngine  ExecutionEngine
   |                |          |
adapters        Portfolio   execution clients
           \       |       /
              Strategy/Actor
```

Strategies are specialised data actors. Commands and events pass through
engines over a central in-process message bus. Backtest and live nodes compose
the same domain and engine concepts with different clients and clocks. Venue
adapters are separate crates rather than conditions inside strategies.

### Useful ideas

- rich, validated identifiers and domain types for instruments, orders, fills,
  accounts, and positions;
- a mandatory risk engine between strategy order commands and execution;
- distinguish a locally proven `NotSent`, a definite `VenueRejected`, and an
  `Ambiguous` submission which must be reconciled rather than retried blindly;
- separate data, execution, portfolio, and risk ownership;
- a cache/read model distinct from the execution clients;
- injectable real and test clocks, deterministic simulation, and explicit event
  ordering;
- the same strategy-facing semantics across backtest and live runtime;
- adapter isolation and contract-focused tests;
- high-performance Rust only underneath stable domain/application boundaries;
  and
- serious performance, supply-chain, deterministic-simulation, and
  multi-platform CI gates.

The risk engine explicitly describes pre-trade validation in
`crates/risk/src/engine/mod.rs:80-104`, and its validated builder configuration
is isolated in `crates/risk/src/engine/config.rs:47-102`. The backtest engine
requires a test clock rather than reading ambient wall time
(`crates/backtest/src/engine.rs:1882-1900`).

The three-way live submission taxonomy is defined in
`crates/live/src/execution/failure.rs`. It is more precise than a boolean:
`Ambiguous` means the venue may have applied the command, so the order remains
in flight until a private stream, query, or startup reconciliation resolves it.

### Costs and constraints not to inherit

NautilusTrader solves multi-asset, multi-venue, institutional-scale problems.
Its large domain taxonomy, actor lifecycle, serialization formats, cache, event
store, plugin system, dozens of adapter crates, Rust/PyO3 build pipeline, and
extensive macros would be a new platform project inside ELVIS.

The central strategy implementation and engine files are still large despite
the strong boundaries. A message bus can also hide control flow when used for a
simple single-process decision. ELVIS does not need wire formats, distributed
actors, or a universal instrument model before it has one reliable paper/live
order lifecycle.

LGPL-3.0 is also a reason to adopt concepts rather than copy code without a
deliberate licensing review.

### ELVIS decision

Adopt the dependency direction, domain invariants, exact values at order and
accounting boundaries, clock abstraction, submission-failure taxonomy,
mandatory risk boundary, and research/live semantic parity. Implement them as
a handful of Python dataclasses, protocols, and application services. Do not
introduce an actor runtime, universal cache, serialization framework, or Rust
extension at this stage.

## Hummingbot

### Observed architecture

Hummingbot is an Apache-2.0 Python/Cython trading framework with a classic V1
strategy layer and a newer Strategy V2 layer. The V2 flow separates controller
decisions from typed executor actions and specialised order/position executors.

```text
Clock -> StrategyV2Base -> MarketDataProvider -> Controllers
                                      |
                                      v
                           ExecutorAction queue
                                      |
                                      v
                           ExecutorOrchestrator
                                      |
                                      v
                    position/grid/DCA/TWAP/... executors
                                      |
                                      v
                                  Connectors
```

Evidence:

- public V1/V2/script taxonomy: `README.md:108-115`;
- V2 composition: `hummingbot/strategy/strategy_v2_base.py:256-289`;
- typed actions: `hummingbot/strategy_v2/models/executor_actions.py:10-36`;
- executor mapping: `hummingbot/strategy_v2/executors/executor_orchestrator.py:515-568`;
- connector template: `hummingbot/connector/exchange_py_base.py:38-85`; and
- in-flight order states: `hummingbot/core/data_type/in_flight_order.py:21-54`.

### Useful ideas

- normalise exchange connectors and their errors/capabilities;
- register the client order before submission;
- track explicit pending/open/partial/filled/cancelled/failed states;
- correlate client and venue IDs and deduplicate fills by trade ID;
- reconcile WebSocket updates with REST and quarantine lost orders;
- use one paper exchange through the same connector-facing contract;
- separate a controller's desired action from an executor that owns the order
  lifecycle; and
- provide bounded graceful shutdown that reports residual exposure.

### Costs and defects not to inherit

Strategy V2 still inherits V1 machinery. Global singletons, dynamic imports,
large configuration maps, reflection-based connector discovery, multiple
independent polling loops, synchronous SQL writes, and unbounded routing queues
remain. Default strategy/controller/executor timings are approximately 1 s,
1 s, and 0.5 s respectively; paper execution adds a 5 s delay. It is not an
ultra-low-latency reference merely because selected structures use Cython/C++.

Risk is not a mandatory central invariant: budget checks and an optional kill
switch do not cover stale data, exposure, leverage, daily loss, venue health,
or position divergence comprehensively. The full connector and MQTT/Gateway
surface would be unjustified for ELVIS.

### ELVIS decision

Adopt the order state machine, reconciliation, narrow connector contract,
typed controller-to-executor intent, and graceful shutdown semantics. Add a
stronger mandatory risk gate and durable idempotency. Reject V1/V2 compatibility
layers, global singletons, reflection-based discovery, autonomous polling per
component, and unmeasured Cython optimisation.

## Freqtrade

### Observed architecture

Freqtrade is a GPLv3 modular Python monolith. `Worker` owns a small lifecycle
state machine and throttled loop, while `FreqtradeBot` manually composes and
orchestrates exchange, strategy, wallets, database, data provider, pairlists,
protections, and RPC. The runtime processes persistent state, candle data,
signals, order reconciliation, exits, and then entries.

Evidence:

- lifecycle and throttle: `freqtrade/worker.py:26-141`;
- composition: `freqtrade/freqtradebot.py:73-179`;
- cycle ordering: `freqtrade/freqtradebot.py:257-311`;
- strategy interface: `freqtrade/strategy/interface.py:52-158`; and
- documented live/backtest flow: `docs/bot-basics.md:41-105`.

### Useful ideas

- a visible `STOPPED`/`RUNNING`/`PAUSED`/reload lifecycle and heartbeat;
- JSON Schema configuration whose requirements change by run mode;
- durable `Trade`, `Order`, `PairLock`, wallet, and key/value models with
  migrations and uniqueness constraints;
- startup recovery of open orders followed by exchange reconciliation;
- normalised exchange exceptions and capability checks;
- composable protections such as cooldown, stop-loss guard, drawdown, and
  low-profit locks;
- one strategy-facing interface reused by live and backtest modes; and
- lookahead and recursive-analysis tools for strategy bias detection.

### Costs and defects not to inherit

`FreqtradeBot` exceeds 2,400 lines and remains a god object. Configuration
dictionaries cross most boundaries, ORM sessions have global aspects, and
dynamic resolvers load arbitrary Python plugins. Its live loop usually runs in
seconds and its simulator is candle based, so fills, depth, latency, impact, and
intra-candle ordering remain approximations.

Its GPLv3 licence makes direct code copying a separate legal/product decision.
The comparison therefore transfers concepts only.

### ELVIS decision

Adopt lifecycle states, schema validation, migrations, durable order/trade
relations, startup reconciliation, protection chains, and bias tests. Use
explicit registries and constructor injection instead of dynamic plugin loading
or a second large orchestrator.

## Adopt / adapt / reject matrix

| Decision | Reference idea | ELVIS application |
|---|---|---|
| Adopt | Nautilus typed domain and engine ownership | immutable signal/order/fill/position contracts and one owner per transition |
| Adopt | Nautilus clock and simulation discipline | monotonic live clock, virtual replay clock, deterministic fixtures |
| Adopt | Nautilus `NotSent` / `VenueRejected` / `Ambiguous` taxonomy | never blindly retry a possibly accepted write |
| Adopt | Hummingbot in-flight order tracker | explicit state machine, client/venue IDs, fill dedupe, unknown quarantine |
| Adopt | Freqtrade persistence and restart reconcile | migrations, uniqueness, named records, exchange verification on startup |
| Adopt | OpenMarket source/ingest timestamps | data freshness and provenance on every market snapshot |
| Adopt | OpenMarket artefact-ordered features | versioned feature schemas validated at load and inference |
| Adopt | Freqtrade protection chain | named fail-closed policies with recorded rejection reasons |
| Adapt | controller/action/executor separation | three light boundaries, no inheritance framework |
| Adapt | venue adapter ecosystems | implement only Binance plus explicit paper/replay adapters first |
| Adapt | event-driven engines | direct calls for safety-critical flow; notifications after transitions |
| Adapt | raw market journal | bounded append-only capture without full CQRS/event sourcing |
| Adapt | Rust/Cython hot paths | profile first; move only proven CPU bottlenecks |
| Reject | wholesale framework adoption | migration remains native to ELVIS and incremental |
| Reject | local microservices now | no JSON/network hop inside one-host trading path |
| Reject | global service locators/singletons | explicit composition and injected ports |
| Reject | arbitrary dynamic plugins | allowlisted registry with explicit capabilities |
| Reject | fail-open or optional risk | every actionable intent passes mandatory risk checks |
| Reject | simulation as live proof | paper/replay results state their fill/latency assumptions |

## Resulting priority for ELVIS

1. typed signal/order/execution contracts and one submission path;
2. model feature-schema contract and fail-closed artefact loading;
3. mandatory pre-trade risk planning;
4. durable order/fill/position state and startup reconciliation;
5. injected clock and one strategy core across replay, paper, and live;
6. source/ingest market journal with bounded buffers;
7. typed immutable configuration and explicit lifecycle;
8. end-to-end freshness, decision, submission, acknowledgment, and divergence
   metrics; and
9. profiling before any Rust/Cython or service extraction.
