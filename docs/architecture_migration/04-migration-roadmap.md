# ELVIS migration roadmap

## Migration contract

Every slice must be independently reviewable and reversible:

1. state the invariant and blast radius;
2. add or adapt focused tests first where practical;
3. make the smallest production change;
4. run focused tests, formatting, and static checks;
5. run the broader non-performance suite when shared runtime code changes;
6. update this status ledger and relevant operator documentation; and
7. create one explicit commit containing only that slice.

No slice enables unattended live trading, changes secrets, deploys, pushes, or
deletes a legacy path before its replacement has passed parity checks.

## Roadmap

| ID | Atomic outcome | Verification gate | Rollback | Status |
|---|---|---|---|---|
| M0 | Pin source revisions; measure source, test, CI, model, and Docker baseline | method and measured results recorded; reproducible test command recorded | documentation-only | Implemented |
| M1 | Publish current map, reference comparison, target architecture, and this ledger | Markdown links and Mermaid blocks checked; docs review | revert docs commit | Implemented |
| M2 | Add immutable signal, order-intent, and submission-report domain contracts | domain unit tests; no I/O imports | remove new unused package | Implemented |
| M3 | Add a direct `OrderService` and narrow `ExecutionPort` with one adapter call and no internal retry | application unit tests; 10,000-call latency tripwire; no network | remove new unused service | Implemented |
| M4 | Add a typed adapter and acknowledged-success handler for the current executor; replace duplicated BUY/SELL submission in the multi-symbol paper path | adapter contract tests; main wiring test; full suite | revert typed wiring only; never restore duplicate direct-order paths | Implemented |
| M5 | Establish versioned feature schemas and validate model artefacts on load | 9/11/20-feature contract tests, incompatible artefact rejection, training/inference round trip | retain prior artefact and loader adapter | Planned |
| M6 | Introduce a fail-closed signal-policy pipeline and move filters one at a time | policy unit tests including exception/timeouts; shadow parity log | disable migrated policy adapter | Planned |
| M7 | Introduce pre-trade risk planning; move cooldown, sizing, leverage ceiling, and fee viability out of `main.py` | risk table tests, property tests, paper replay; no fallback order | feature flag selects legacy planner | Planned |
| M8 | Make one `PositionService` own fills, stops, take profit, and reconciliation; retire background/inline duplicate ownership | state-machine tests, restart/reconciliation integration test | select legacy position manager | Planned |
| M9 | Replace positional PostgreSQL tuples with repositories and migrations | ephemeral PostgreSQL from empty volume, upgrade test, transaction/idempotency tests | compatibility repository adapter | Planned |
| M10 | Parse configuration once; replace global service lookup at migrated boundaries | config validation matrix, startup failure tests | compose legacy services in adapter | Planned |
| M11 | Move API, dashboard, metrics, and notifications to read models/post-transition sinks | fault-injection tests prove trading result is unchanged | detach sink | Planned |
| M12 | Remove dead event handlers, duplicate modules, legacy execution branch, and global lookups after call-site audit | `rg` zero-reference proof, full suite, paper soak | deletion in separate commits | Planned |
| M13 | Profile replay and paper runtime; optimise only measured hot spots | recorded p50/p95/p99, CPU/RSS, deterministic replay checksum | revert isolated optimisation | Planned |

## First operational vertical slice

M2 through M4 form the first end-to-end slice:

```text
existing strategy and filters
        |
        v
typed Signal -> validated OrderIntent -> OrderService -> legacy executor adapter
                                                    -> typed SubmissionReport
                                                    -> existing success recorder
```

This slice removes duplicated BUY/SELL branching and makes submission outcomes
testable without changing the strategy, filter, sizing, database, or venue
algorithms. It is deliberately narrower than a new engine.

### M2 acceptance criteria

- invalid/non-finite price, quantity, confidence, and leverage cannot construct
  an actionable order;
- `HOLD` cannot become an `OrderIntent`;
- submission reports distinguish `NOT_SENT`, `VENUE_REJECTED`, `SUBMITTED`, and
  `AMBIGUOUS` without implying that submission is a fill; and
- the domain package has no environment, pandas, database, or Binance
  dependency.

### M2 implementation record

The new `trading.domain` package is unused by the legacy runtime and therefore
changes no order behaviour. It contains only standard-library and internal
imports, keeps `HOLD` out of `OrderSide`, limits the initial order type to
`MARKET`, and treats `SUBMITTED` as an acknowledgment rather than a fill.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/domain
.venv/bin/python -m pytest tests/test_domain_contracts.py -q
.venv/bin/python -m black --target-version py310 --check trading/domain tests/test_domain_contracts.py
.venv/bin/python -m isort --check-only trading/domain tests/test_domain_contracts.py
.venv/bin/python -m flake8 trading/domain tests/test_domain_contracts.py --max-line-length=88
```

The focused suite passed 85 tests. The tests include an import-purity gate and
explicitly reject the pre-existing `trading.orders.OrderSide` at the new domain
boundary; the later legacy adapter must map between the two enums deliberately.

### M3 acceptance criteria

- executor exceptions and malformed results become `AMBIGUOUS`;
- each invocation makes at most one adapter call and `OrderService` never
  retries;
- the service has no environment, pandas, database, or Binance dependency; and
- a focused latency test covers at least 10,000 fake executions without network
  I/O and guards against an accidental high-overhead design.

### M3 implementation record

`trading.application.OrderService` is stateless and still unused by the legacy
runtime. It makes one `ExecutionPort.submit()` call per invocation. Exceptions,
malformed responses, and mismatched client order IDs become `AMBIGUOUS` reports
whose exception details are not exposed. Expected adapter failures remain typed
return values. The service has no retry, recorder, telemetry, persistence,
environment, pandas, database, or Binance dependency.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/application
.venv/bin/python -m pytest tests/test_order_service.py -q
.venv/bin/python -m pytest tests/perf/test_order_service_latency.py -q -m perf -s
.venv/bin/python -m black --target-version py310 --check trading/application tests/test_order_service.py tests/perf/test_order_service_latency.py
.venv/bin/python -m isort --check-only trading/application tests/test_order_service.py tests/perf/test_order_service_latency.py
.venv/bin/python -m flake8 trading/application tests/test_order_service.py tests/perf/test_order_service_latency.py --max-line-length=88
```

The unit suite passed 17 tests. The warmed 10,000-sample in-memory run measured
p99 at 0.21 microseconds with `perf_counter_ns`, garbage collection enabled,
CPython 3.14.6, macOS 27.0 arm64, and an Apple M1 Max. This is an application-
overhead regression tripwire, not an end-to-end exchange-latency claim.

### M4 acceptance criteria

- both current `execute_buy` and `execute_sell` routes satisfy one adapter
  contract;
- the adapter declares the current executor as paper-only and a `live` runtime
  cannot accidentally activate venue submission;
- existing cooldown and model-vote recording happens once after acknowledged
  execution;
- no recording happens for a rejected or failed execution;
- multi-symbol paper behaviour remains enabled by default;
- `main.py` has no direct `place_order`, `execute_buy`, or `execute_sell` call,
  no environment escape can restore one, and an old strategy API fails closed;
- an acknowledged legacy fill must echo the exact symbol and side and contain a
  non-blank order ID;
- values that overflow or underflow during the `Decimal`-to-float boundary
  conversion are rejected before any executor call; and
- the full non-performance suite has no regression beyond a documented baseline
  environmental failure.

### M4 implementation record

The active multi-symbol branch now constructs a validated `Signal` and
`OrderIntent`, then calls one `OrderService` for both BUY and SELL. The
paper-only adapter converts `Decimal` to float at the legacy boundary and maps
only explicit responses: `FILLED` with the exact symbol, side, and a non-blank
order ID is acknowledged, `BLOCKED` is `NOT_SENT`, explicit `REJECTED` is
`VENUE_REJECTED`, and empty, malformed, incoherent, or unknown responses are
`AMBIGUOUS`. Non-representable or float-underflowed order values are rejected
before the executor boundary. Votes and cooldown are recorded once only for an
acknowledged report. The existing executor's own trade/database writes are not
duplicated.

`main(mode="live")` now fails before bootstrap, and the adapter independently
returns `NOT_SENT` outside paper mode. This prevents bootstrap from creating an
authenticated client or changing venue leverage under the unsupported live
mode. The environment escape for duplicate single-symbol execution was removed,
all direct placement calls were removed from `main.py`, and the old-strategy
fallback now logs and refuses actionable output instead of submitting it.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/execution/legacy_paper_adapter.py main.py
.venv/bin/python -m pytest tests/test_legacy_paper_adapter.py tests/test_main_order_submission.py -q
.venv/bin/python -m pytest tests/test_binance_executor.py tests/test_paper_fill_integrity.py tests/test_roadmap_wiring.py -q
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The M4-focused suite passed 31 tests; the cumulative M2--M4 contract suite
passed 133 tests; and the selected regression suite passed 25. The full result
was 796 passed, 9 skipped, 3 deselected, and the same one
baseline failure: the locally reachable PostgreSQL instance lacks
`np.trades`. No new failure was introduced.

## Cut-over policy

Each later behavioural migration has three modes:

1. **legacy** — current implementation is authoritative;
2. **shadow** — both implementations evaluate the same frozen input, only the
   legacy path may act, and differences are recorded; and
3. **active** — the new implementation acts, with a narrow switch back to
   legacy until the next stable checkpoint.

Shadow mode must never call a second executor or mutate cooldown, positions,
portfolio, model feedback, or persistence. A comparison is useful only if it is
side-effect free.

## Commit plan

Commits remain small and ordered. The initial sequence is:

1. `docs(architecture): record ELVIS migration design`
2. `feat(domain): add typed trading contracts`
3. `feat(execution): add deterministic order service`
4. `refactor(execution): route paper orders through order service`
5. `feat(models): enforce versioned feature schemas`

If a step exposes a pre-existing defect that blocks its gate, the fix receives
its own test-first commit unless it is inseparable from the new invariant.

## Definition of migration complete

The architecture migration is complete when:

- the runner is a small lifecycle shell around `TradingCycle`;
- domain/application packages do not import exchange, database, UI, environment,
  or global-container modules;
- paper, replay, and live use the same decision/risk/order/position services;
- every deployed model passes an explicit feature-schema contract;
- one component owns each order and position transition;
- a fresh ephemeral PostgreSQL database migrates and passes integration tests;
- unit tests make no uncontrolled network connections;
- dead event handlers and superseded duplicate implementations are removed;
- the complete test, lint, container, replay, and paper-soak gates pass; and
- operator and architecture documentation describe the code that is actually
  active, with no planned component presented as implemented.
