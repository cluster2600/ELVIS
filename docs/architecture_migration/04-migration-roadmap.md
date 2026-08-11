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
| M2 | Add immutable signal, order-intent, and submission-report domain contracts | domain unit tests; no I/O imports | remove new unused package | Planned |
| M3 | Add a direct `OrderService` and narrow `ExecutionPort` with one adapter call and no internal retry | application unit tests; 10,000-call latency tripwire; no network | remove new unused service | Planned |
| M4 | Add adapters for the current executor and success recorder; replace duplicated BUY/SELL submission in the multi-symbol paper path | adapter contract tests; main wiring test; full suite | one call-site revert restores legacy branch | Planned |
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

### M3 acceptance criteria

- executor exceptions and malformed results become `AMBIGUOUS`;
- each invocation makes at most one adapter call and `OrderService` never
  retries;
- the service has no environment, pandas, database, or Binance dependency; and
- a focused latency test covers at least 10,000 fake executions without network
  I/O and guards against an accidental high-overhead design.

### M4 acceptance criteria

- both current `execute_buy` and `execute_sell` routes satisfy one adapter
  contract;
- the adapter declares the current executor as paper-only and a `live` runtime
  cannot accidentally activate venue submission;
- existing cooldown and model-vote recording happens once after accepted
  execution;
- no recording happens for a rejected or failed execution;
- multi-symbol paper behaviour remains enabled by default;
- the disabled legacy single-symbol block is not re-enabled; and
- the full non-performance suite has no regression beyond a documented baseline
  environmental failure.

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
