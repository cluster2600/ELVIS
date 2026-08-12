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

No slice enables unattended live trading, changes secrets, deploys, or pushes.
A load-bearing legacy path is deleted only after its replacement passes parity
checks. A path proven synthetic, non-deployable, and inactive may instead be
retired after an explicit audit, zero-call regression tests, and a documented
rollback decision that does not restore unsafe behaviour.

## Roadmap

| ID | Atomic outcome | Verification gate | Rollback | Status |
|---|---|---|---|---|
| M0 | Pin source revisions; measure source, test, CI, model, and Docker baseline | method and measured results recorded; reproducible test command recorded | documentation-only | Implemented |
| M1 | Publish current map, reference comparison, target architecture, and this ledger | Markdown links and Mermaid blocks checked; docs review | revert docs commit | Implemented |
| M2 | Add immutable signal, order-intent, and submission-report domain contracts | domain unit tests; no I/O imports | remove new unused package | Implemented |
| M3 | Add a direct `OrderService` and narrow `ExecutionPort` with one adapter call and no internal retry | application unit tests; 10,000-call latency tripwire; no network | remove new unused service | Implemented |
| M4 | Add a typed adapter and acknowledged-success handler for the current executor; replace duplicated BUY/SELL submission in the multi-symbol paper path | adapter contract tests; main wiring test; full suite | revert typed wiring only; never restore duplicate direct-order paths | Implemented |
| M5 | Establish versioned feature schemas and validate model artefacts on load | 9/11-feature contracts, incompatible artefact rejection, invalid Ensemble members retired, training/inference round trip | revert only the current contract adapter; never restore invalid loaders | Implemented |
| M6 | Introduce a fail-closed signal-policy pipeline and move filters one at a time | policy unit tests including exception/timeouts; shadow parity log | disable migrated policy adapter | In progress (M6a core) |
| M7 | Introduce pre-trade risk planning; move cooldown, sizing, leverage ceiling, and fee viability out of `main.py` | risk table tests, property tests, paper replay; no fallback order | feature flag selects legacy planner | In progress (M7h fee-regime cut-over) |
| M8 | Make one `PositionService` own fills, stops, take profit, and reconciliation; retire background/inline duplicate ownership | state-machine tests, restart/reconciliation integration test | select legacy position manager | In progress (M8b pure position reducer; no runtime cut-over) |
| M9 | Replace positional PostgreSQL tuples with repositories and migrations | ephemeral PostgreSQL from empty volume, upgrade test, transaction/idempotency tests | compatibility repository adapter | In progress (M9b.6 unresolved-submission read model) |
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

### M5a implementation record

The first model slice defined immutable, Python 3.10-compatible feature schemas
without changing runtime model selection. Research and Bonenkamp each received
distinct 9- and 11-feature identities because their indicator implementations
and social inputs are not interchangeable. It also captured the incompatible
orders of the then-present YDF and CoreML 20-feature paths. M5d subsequently
used that evidence to retire both synthetic placeholders and removed their
unused runtime schemas; the Research and Bonenkamp contracts remain active.

Each schema owns its ordered names, logical dtypes, and preprocessing version.
Vectorization rejects missing, boolean, non-numeric, NaN, or infinite values;
unrelated context keys are allowed. Fitted sklearn-like components must declare
the exact input dimension and, when available, exact ordered feature names.
There is no padding, truncation, or implicit default in the contract.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/models/feature_schema.py trading/models/feature_schemas.py
.venv/bin/python -m pytest tests/test_feature_schema_contracts.py -q
.venv/bin/python -m black --target-version py310 --check trading/models/feature_schema.py trading/models/feature_schemas.py trading/models/__init__.py tests/test_feature_schema_contracts.py
.venv/bin/python -m isort --check-only trading/models/feature_schema.py trading/models/feature_schemas.py trading/models/__init__.py tests/test_feature_schema_contracts.py
.venv/bin/python -m flake8 trading/models/feature_schema.py trading/models/feature_schemas.py trading/models/__init__.py tests/test_feature_schema_contracts.py --max-line-length=88
```

All 21 M5a contract tests passed. M5 remains in progress until manifests are
validated before deserialization and the active producers/consumers use these
schemas.

### M5b implementation record

The second model slice adds a model-local feature manifest. It records the exact
feature schema, model kind, library and library version, a timezone-aware
creation time, component filenames, and SHA-256 digests. Validation rejects an
unknown format, another schema or runtime version, malformed data, missing or
tampered files, symbolic links, components outside the manifest directory, and
duplicate component paths.

The manifest is written atomically and last. A loader can therefore validate
schema, runtime, path, and hash before invoking a pickle/joblib deserializer.
This contract complements `core.models.ModelRegistry`: the existing registry is
an unused approval catalogue, whereas the sibling manifest is an enforced
compatibility boundary for all components of one model. They are not treated as
two deployment authorities.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/models/artifact_manifest.py
.venv/bin/python -m pytest tests/test_feature_artifact_manifest.py -q
.venv/bin/python -m black --target-version py310 --check trading/models/artifact_manifest.py trading/models/__init__.py tests/test_feature_artifact_manifest.py
.venv/bin/python -m isort --check-only trading/models/artifact_manifest.py trading/models/__init__.py tests/test_feature_artifact_manifest.py
.venv/bin/python -m flake8 trading/models/artifact_manifest.py trading/models/__init__.py tests/test_feature_artifact_manifest.py --max-line-length=88
```

All 10 M5b tests passed, including a fitted sklearn
training--persistence--validation--inference round trip. Runtime loaders remain
unchanged until M5c, so this commit cannot activate or reject a production
model by itself.

### M5c Research implementation record

`ResearchBasedStrategy` now selects a distinct 9- or 11-feature schema once at
construction. Training and inference vectorize the same named mapping in that
schema's order. A missing value, non-finite value, scaler mismatch, unfitted
preprocessor, or wrong shape raises a feature-contract error; the former
pad/truncate and unscaled-prediction paths are gone. Feature preparation is
performed only for an activated model; the untrained RSI fallback does not
pretend to be model inference.

Training fits cloned model/scaler candidates and activates them together only
after cross-validation, fitted dimensions, concrete implementation types,
binary class order, persistence, and manifest creation all succeed. A failed
retraining therefore leaves the active in-memory pair untouched. Loading
validates the manifest, exact scikit-learn version, and component hashes before
joblib; treats `InconsistentVersionWarning` as an incompatibility; validates
both loaded objects in locals; and assigns them together only after every check
passes. Unmanifested ignored local pickles are therefore not activated. The old
procedural mismatch test was replaced with isolated assertions, and the
remaining research integration test now writes only under `tmp_path`.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/models trading/strategies/research_based_strategy.py
.venv/bin/python -m pytest tests/test_research_feature_schema.py tests/test_research_strategy_features.py tests/test_feature_fix.py tests/test_research_strategy.py -q
.venv/bin/python -m pytest tests/test_feature_schema_contracts.py tests/test_feature_artifact_manifest.py tests/test_research_feature_schema.py tests/test_research_strategy_features.py tests/test_feature_fix.py -q
```

The Research-focused suite passed 22 tests and the cumulative M5 contract suite
passed 52. The full non-performance suite reached 839 passed, 9 skipped, 3
deselected, and the same baseline PostgreSQL failure (`np.trades` is absent).
At this Research checkpoint, M5 remained in progress for Bonenkamp and the
active Ensemble model adapters. Signal-policy fallbacks are intentionally left
to M6.

### M5c Bonenkamp implementation record

`BonenkampHFTStrategy` now applies its own 9- or 11-feature schema at both
training and inference. It no longer returns an all-zero vector, predicts on
unscaled data, or prepares a model vector while operating its untrained RSI
fallback. A trained instance requires the declared fitted scaler and returns
`HOLD` without invoking the classifier when the contract fails.

Construction now attempts to load the model through the strict sibling
manifest. Schema, exact scikit-learn version, component hashes, concrete
Random-Forest/StandardScaler types, fitted dimensions, and binary class order
must all pass before the pair is assigned. Consequently, the ignored local
pickles left by the baseline are not activated without a compatible manifest.

Retraining uses cloned candidates and a `TimeSeriesSplit`; it persists the
validated pair before changing any active model, scaler, timestamp, or score
history. Cross-validation, mono-class, and persistence failures therefore
preserve the prior in-memory state. The former procedural test file was reduced
to isolated pytest assertions under `tmp_path`, so it no longer rewrites the
working-copy artefacts or reports return-value tests as passing.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/strategies/bonenkamp_hft_strategy.py
.venv/bin/python -m pytest tests/test_bonenkamp_feature_schema.py tests/test_bonenkamp_strategy.py -q
.venv/bin/python -m pytest tests/test_feature_schema_contracts.py tests/test_feature_artifact_manifest.py tests/test_research_feature_schema.py tests/test_research_strategy_features.py tests/test_feature_fix.py tests/test_bonenkamp_feature_schema.py tests/test_bonenkamp_strategy.py -q
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The Bonenkamp-focused suite passed 24 tests and the cumulative M5 contract
suite passed 76. The full non-performance suite reached 856 passed, 9 skipped,
3 deselected, and the same baseline PostgreSQL failure (`np.trades` is absent).
M5 remains in progress for the active Ensemble model adapters. Transactional
multi-file artefact publication remains a later persistence hardening slice;
the current manifest is atomic and written last, so interrupted bundles reject
closed on the next load.

### M5d Ensemble placeholder retirement implementation record

The 20-feature investigation established that neither legacy Ensemble member
was a deployable model. The tracked YDF directory was a provenance-less
synthetic classifier: its own OOB accuracy was 37.1%, below the 39.3%
majority-class baseline, while the configured TensorFlow-DF loader could not
load its native YDF bundle. Even if loaded with the correct API, the runtime
mistook a three-probability output for a class index and fabricated confidence.
The CoreML artefact expected by runtime did not exist; its only producer trained
on random data, exposed different input/output names, and wrote to a developer
absolute path.

Both members were therefore retired instead of repaired. Their imports,
constructor options, loaders, prediction branches, unused feature schemas,
configuration keys, random-data generator, and tracked 5.5 MB placeholder were
removed. `DataProcessor` and the legacy DataFrame adapter no longer invent
order, future-price, sentiment, social, or order-book inputs. Legitimate
features such as a computed rolling volume mean remain available when an
actual producer creates them.

The unused YDF/CoreML packages and provisioning hooks were also removed. The
optional Python 3.10 training image remains for the real unified training entry
point and now explicitly installs pinned CPU PyTorch, TensorFlow, and seaborn;
Ansible installs only runtime requirements into its deployment venv. The
container entry point now runs its child from the repository root, preserves
Compose-provided database settings, propagates child failures, and cleans up
without turning a successful run into a failure.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/models trading/data/data_processor.py trading/strategies/ensemble_strategy.py scripts/train_no_vault.py tests/test_ensemble_placeholder_retirement.py tests/test_training_entrypoint.py
.venv/bin/python -m pytest -q tests/test_ensemble_placeholder_retirement.py tests/test_feature_schema_contracts.py tests/test_main_retrain_hook.py tests/test_training_entrypoint.py
.venv/bin/python -m pytest -q tests/test_feature_schema_contracts.py tests/test_feature_artifact_manifest.py tests/test_research_feature_schema.py tests/test_research_strategy_features.py tests/test_feature_fix.py tests/test_research_strategy.py tests/test_bonenkamp_feature_schema.py tests/test_bonenkamp_strategy.py tests/test_ensemble_placeholder_retirement.py tests/test_main_retrain_hook.py tests/test_training_entrypoint.py
ansible-playbook --syntax-check ansible/playbook.yml -i 'localhost,' --connection=local
docker compose --profile ml config --no-env-resolution --no-path-resolution --no-interpolate -q
docker compose --profile ml build elvis-ml-trainer
docker run --rm --network none --entrypoint python \
  elvis-architecture-migration-elvis-ml-trainer:latest \
  -c "import torch, tensorflow, seaborn; from training.models.model_trainer import ModelTrainer; print(torch.__version__, tensorflow.__version__, seaborn.__version__, ModelTrainer.__name__)"
docker run --rm --network none --entrypoint python \
  elvis-architecture-migration-elvis-ml-trainer:latest \
  training/train_models.py --help
docker run --rm --network none -v "$PWD/data:/app/data:ro" \
  elvis-architecture-migration-elvis-ml-trainer:latest \
  bash scripts/run_training.sh --quick --no-vault
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused retirement/contracts/entry-point suite passed 35 tests and the
cumulative M5 suite passed 91. The rebuilt Python 3.10 image imported CPU
PyTorch 2.10.0, TensorFlow 2.16.2, seaborn 0.13.2, and `ModelTrainer`; the
checked-in sample data completed the real no-Vault training command with exit
code zero. A no-data smoke reached the same child and returned its failure
without printing the global success banner. The full non-performance suite
reached 863 passed, 9 skipped, 3 deselected, and the same baseline
PostgreSQL failure (`np.trades` is absent).
No replacement model will be admitted without causal data, a reproducible
producer, a manifest round trip, and out-of-sample evidence above a declared
baseline.

### M5e invalid trade-learned path retirement implementation record

The remaining `trade_learned` vote was not a direction model. Its producer
selected only completed SELL rows and labelled the same row from realised net
PnL. Eleven of its twenty inputs were constants, symbol was absent, and its
random split reused future observations during cross-validation. Runtime then
constructed a different distribution from wall-clock time and simulated
defaults, mapped "profitable SELL" to BUY, and emitted a three-value vector
whose sum could exceed one. A manifest would have made this invalid meaning
reproducible, not correct.

The loader, constructor option, runtime vote, synthetic feature extractor, and
isolated producer were therefore removed. Configuration arguments after
`symbols` are keyword-only so an old positional model path fails immediately
instead of being reinterpreted as the MLX URL. Research and Bonenkamp remain
the only sklearn artefact-backed strategy members wired into Ensemble, and
retain their strict schemas, manifests, hashes, version checks, and atomic
activation. Ignored local pickle files are also excluded from Docker build
contexts; no user-owned local artefact was deleted. A future learned component
must start as either a causal future-return direction model or a meta-policy
that can veto a proposed direction but cannot invent the opposite side.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_ensemble_placeholder_retirement.py tests/test_fixed_strategies.py tests/test_signal_generation.py
.venv/bin/python -m pytest -q tests/test_feature_schema_contracts.py tests/test_feature_artifact_manifest.py tests/test_research_feature_schema.py tests/test_bonenkamp_feature_schema.py tests/test_ensemble_placeholder_retirement.py
git grep -n -E 'trade_learned|TradeBasedTrainer|trade_based_trainer' -- core trading training scripts
docker compose --profile ml build elvis-ml-trainer
docker run --rm --network none --entrypoint python \
  elvis-architecture-migration-elvis-ml-trainer:latest \
  -c "from pathlib import Path; from trading.strategies.ensemble_strategy import EnsembleStrategy; import inspect; assert 'trade_learned_model_path' not in inspect.signature(EnsembleStrategy).parameters; assert not Path('/app/training/trade_based_trainer.py').exists(); assert not list(Path('/app').rglob('*.pkl')); print('retired-paths-absent')"
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The retirement tests prove that even an injected legacy object receives zero
calls and that the unchanged technical fallback still supplies the only vote
when no validated optional strategy is available. The focused strategy suite
passed 9 tests (with 4 pre-existing return-value warnings), the cumulative
feature-contract suite passed 61 tests (with 118 joblib/NumPy compatibility
warnings), and an expanded Ensemble call-site suite passed 36 tests with one
skip. The rebuilt image printed `retired-paths-absent`; it contained neither
the producer nor any ignored pickle, while deployment can still mount validated
artifacts under `/app/models`. The executable zero-reference check returned no
matches. The full non-performance suite reached 863 passed, 9 skipped, 3
deselected, and only the same local PostgreSQL baseline failure because
`np.trades` is absent. The pinned baseline audit and historical changelog remain
unchanged.

### M6a fail-closed signal-policy core implementation record

`trading.application.SignalPolicyPipeline` is a pure, deliberately unused
application primitive. It applies an immutable tuple of named policies exactly
once and in order. A policy returns only `SignalPolicyResult`: it may attach
reasons, adjust confidence, or veto. It cannot return an action or a replacement
signal, so promotion from HOLD and reversal between BUY and SELL are
unrepresentable. The pipeline alone rebuilds the validated `Signal`, preserving
its decision ID, symbol, side, price, timestamp, strategy, and prior reasons.

Policy IDs are validated and snapshotted at composition time. A malformed
return, `TimeoutError`, or other `Exception` becomes HOLD at confidence zero,
stops later policies, and adds a stable `policy:<id>:...` reason without the
exception message. `BaseException` is not swallowed. An incoming HOLD and an
empty pipeline return the exact input object and perform no calls.

This slice has no `main.py` wiring and therefore changes no trading behaviour.
It intentionally does not claim to interrupt blocking calls: the synchronous
core can only convert a `TimeoutError` already raised by a policy. Network or
database-backed policies remain excluded until a bounded adapter exists. The
next slice will add one pure RSI policy and compare it in shadow mode before it
can become authoritative.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/application tests/test_signal_policy.py
/usr/local/bin/python3.10 -m pytest -q tests/test_signal_policy.py tests/test_order_service.py
.venv/bin/black --target-version py310 --check trading/application tests/test_signal_policy.py tests/test_order_service.py
.venv/bin/isort --check-only trading/application tests/test_signal_policy.py tests/test_order_service.py
.venv/bin/flake8 trading/application tests/test_signal_policy.py tests/test_order_service.py --max-line-length=88
git grep -n 'SignalPolicyPipeline' -- main.py core trading | grep -v 'trading/application'
```

The focused suite covers validation boundaries, immutability, fixed order,
confidence propagation, reason attribution, identity preservation, veto
short-circuiting, malformed returns, timeout/exception sanitisation, HOLD
short-circuiting, import purity, and non-swallowing of `KeyboardInterrupt`.
It passed 50 policy tests, 67 combined application tests, and 152 combined
domain/application tests. The executable wiring check returned no matches in
M6a. The full non-performance suite reached 913 passed, 9 skipped, 3
deselected, and only the unchanged local PostgreSQL baseline failure because
`np.trades` is absent.

### M6b1 pure RSI gate implementation record

`RsiGatePolicy` captures one immutable RSI observation and implements the
legacy gate's strict valid-data boundaries independently: BUY is vetoed only
above 70, SELL only below 30, and the boundary values pass. It returns only the
restricted `SignalPolicyResult`, so it cannot reverse or promote a side. Its
threshold configuration is finite, bounded, ordered, and immutable.

Representative finite observations in `[0, 100]`, including both strict
boundaries and values immediately on either side, are compared against the
legacy `rsi_gate` and prove action/confidence parity. Missing, boolean,
non-numeric, non-finite, or out-of-range observations deliberately diverge
fail-closed to HOLD. This safer behaviour remains candidate-only until the next
shadow slice measures it on the exact per-symbol input. The policy has no
pandas, NumPy, logging, environment, database, clock, or network dependency and
is not yet imported by `main.py`.

Verification at implementation time:

```bash
/usr/local/bin/python3.10 -m compileall -q trading/application/rsi_gate_policy.py tests/test_rsi_gate_policy.py
/usr/local/bin/python3.10 -m pytest -q tests/test_rsi_gate_policy.py tests/test_signal_policy.py tests/test_order_service.py
.venv/bin/black --target-version py310 --check trading/application tests/test_rsi_gate_policy.py tests/test_signal_policy.py tests/test_order_service.py
.venv/bin/isort --check-only trading/application tests/test_rsi_gate_policy.py tests/test_signal_policy.py tests/test_order_service.py
.venv/bin/flake8 trading/application tests/test_rsi_gate_policy.py tests/test_signal_policy.py tests/test_order_service.py --max-line-length=88
```

The focused suite passed 104 tests, including all valid RSI boundaries and the
deliberate invalid-data divergence. The full non-performance suite reached 950
passed, 9 skipped, 3 deselected, and only the unchanged local PostgreSQL
baseline failure because `np.trades` is absent. Runtime authority remains
wholly legacy.

### M6b2 RSI shadow wiring implementation record

The loop now normalizes the current symbol's RSI gate observation once before
the strategy's neutral `50.0` fallback. The authoritative
`apply_signal_filters` call runs first and receives that same normalized value.
Only after it completes does the shadow observer compare the legacy RSI-specific
veto reason with `SignalPolicyPipeline((RsiGatePolicy(...),))`; HOLD decisions
from momentum, squeeze, trading hours, or MACD do not contaminate the
comparison.

`ELVIS_RSI_POLICY_MODE=shadow` enables observation; the default and every other
value remain legacy-only. The observer returns `None`, its call is a standalone
expression, and it has no executor, order service, cooldown, model-feedback,
portfolio, or database dependency. It catches ordinary candidate and logging
failures after the authoritative filter has run. Structured match/divergence
records carry a fresh shadow evaluation ID, current symbol, strategy, stage,
legacy and candidate action/confidence, and reason codes, but no candle or
order-book payload. The evaluation ID is deliberately not presented as the
later order correlation ID. Keys which the global logging context overwrites
are avoided.

Valid observations match at the strict 70/30 boundaries. Missing and NaN values
produce the expected candidate-only fail-closed divergence; some invalid values
already veto in both paths depending on side. These differences are recorded,
never applied. Activation remains a later commit: it requires a measured paper
shadow window, disables only `rsi_gate` inside the legacy composite, and leaves
the single downstream `OrderService` path unchanged.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_rsi_policy_shadow.py tests/test_rsi_gate_policy.py tests/test_signal_policy.py tests/test_main_order_submission.py
.venv/bin/black --target-version py310 --check tests/test_rsi_policy_shadow.py tests/test_rsi_gate_policy.py
.venv/bin/isort --check-only tests/test_rsi_policy_shadow.py tests/test_rsi_gate_policy.py
.venv/bin/flake8 tests/test_rsi_policy_shadow.py tests/test_rsi_gate_policy.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q main.py trading/application tests/test_rsi_policy_shadow.py
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused suite passed 121 tests. It includes AST gates proving one
post-legacy, non-assigned shadow call, the same `_filter_rsi` input on both
paths, and no second submission API. The full-suite result is recorded after
the final review: 976 passed, 9 skipped, 3 deselected, and only the unchanged
local PostgreSQL baseline failure because `np.trades` is absent.

### M7a risk-decision contract implementation record

`trading.domain.RiskDecision` is an immutable, infrastructure-free boundary
between pre-trade planning and execution. It contains one clean decision ID, a
strict approval boolean, immutable reasons, and an optional
`OrderIntent`. Approval requires exactly one typed intent carrying the same
decision ID. Rejection forbids an intent and requires at least one reason, so a
denied plan cannot cross accidentally into `OrderService`.

This slice is deliberately unused by `main.py`. It adds no portfolio snapshot,
configuration, sizing algorithm, fallback, database read, or feature flag. The
legacy planner remains authoritative until a later M7 slice can compare a
complete candidate plan without inheriting the known cross-symbol data and
leverage-configuration defects.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_risk_decision.py tests/test_domain_contracts.py
.venv/bin/black --target-version py310 --check trading/domain/risk.py trading/domain/__init__.py tests/test_risk_decision.py
.venv/bin/isort --check-only trading/domain/risk.py trading/domain/__init__.py tests/test_risk_decision.py
.venv/bin/flake8 trading/domain/risk.py trading/domain/__init__.py tests/test_risk_decision.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q trading/domain tests/test_risk_decision.py
```

The focused contract suite passed 22 tests and the cumulative domain suite
passed 107. The full suite passed 998 tests, skipped 9, deselected 3, and kept
only the unchanged local PostgreSQL baseline failure because `np.trades` is
absent. The existing domain import-purity gate covers the new module.

### M7b canonical bootstrap leverage implementation record

The pre-trade audit found two different `TRADING_CONFIG` dictionaries. The
bootstrap imported the package-level legacy mapping, which has no
`DEFAULT_LEVERAGE`, and silently supplied `50` to the executor while the
documented canonical value in `config.config` is `3`. This made startup depend
on an unsafe fallback and on `OVERRIDE_HIGH_LEVERAGE` rather than the declared
configuration.

Bootstrap now aliases `config.config.TRADING_CONFIG` explicitly for the
leverage field, reads `DEFAULT_LEVERAGE` with a required-key lookup, and passes
that one value to both the first executor and its paper fallback. A missing key
aborts construction before either executor exists. Its existing package-level
mapping still governs mode and other legacy settings, so paper does not become
Futures testnet as a side effect. Merging those consumers remains M10. This
slice also does not enable live trading, which remains rejected before
bootstrap.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_bootstrap_leverage_config.py
.venv/bin/python -m pytest -q tests/test_bootstrap_leverage_config.py tests/test_bootstrap_exchange_config.py tests/test_binance_executor.py tests/test_legacy_paper_adapter.py tests/test_main_order_submission.py
.venv/bin/black --target-version py310 --check core/bootstrap.py tests/test_bootstrap_leverage_config.py
.venv/bin/isort --check-only core/bootstrap.py tests/test_bootstrap_leverage_config.py
.venv/bin/flake8 tests/test_bootstrap_leverage_config.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q core/bootstrap.py tests/test_bootstrap_leverage_config.py
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The three leverage-source tests pass, including primary/fallback parity and a
missing-key fail-closed case. The focused bootstrap/execution regression suite
passes 51 tests. The full suite passes 1,001 tests, skips 9, deselects 3, and
keeps only the unchanged local PostgreSQL baseline failure because `np.trades`
is absent.

### M7c contract-quantity cost semantics implementation record

The fee-gate audit found that its `quantity` is already the contract/base
quantity used by the executor. Multiplying `price * quantity`, fees, and PnL by
leverage a second time produced incorrect amounts and would corrupt every
future positive profit threshold. The corrected convention is:

```text
entry_notional = entry_price * quantity
exit_notional  = exit_price * quantity
initial_margin = entry_notional / leverage
gross_pnl      = direction * (exit_price - entry_price) * quantity
```

`all_in_cost` now requires `expected_exit_price` by keyword, charges each taker
fee on its own fill notional, and funds the entry notional. Leverage is absent
from both cost/viability APIs; for a fixed quantity it changes margin, not PnL
or exchange fees. The keyword-only exit price also makes the removed legacy
three-positional-argument form fail instead of silently remapping values.

Invalid, boolean, negative, non-finite, conversion-overflowed, or
calculation-overflowed cost inputs remain distinguishable from a deliberate
zero-fee schedule. They return a finite non-viable result. A negative minimum
profit is rejected. In `main.py`, an unexpected fee-gate exception now forces
HOLD instead of leaving BUY/SELL actionable. This slice does not yet reserve
margin, reconcile exposure, or claim a venue-accurate future funding schedule;
those remain in the pre-trade snapshot/service work.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_fee_gate.py tests/test_roadmap_wiring.py tests/test_main_order_submission.py
.venv/bin/python -m doctest trading/fees/fee_gate.py
.venv/bin/black --target-version py310 --check trading/fees/fee_gate.py tests/test_fee_gate.py tests/test_roadmap_wiring.py
.venv/bin/isort --check-only trading/fees/fee_gate.py tests/test_fee_gate.py tests/test_roadmap_wiring.py
.venv/bin/flake8 trading/fees/fee_gate.py tests/test_fee_gate.py tests/test_roadmap_wiring.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q trading/fees/fee_gate.py main.py tests/test_fee_gate.py tests/test_roadmap_wiring.py
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused suite passes 84 tests, including cross-checks against
`BinanceFeeCalculator` at 1x/3x/10x and an AST gate for exception-to-HOLD. The
full suite passes 1,038 tests, skips 9, deselects 3, and keeps only the
unchanged local PostgreSQL baseline failure because `np.trades` is absent.

### M7d per-symbol market-input isolation implementation record

The active multi-symbol loop previously enriched only the primary BTC frame,
then reused that frame for every symbol's high-win-rate analysis, regime
detection, roadmap filters, and volume sizing. A BNB decision could therefore
be qualified and sized from BTC candles and volume. The temporary
`filter_result` and `regime_result` values were also guarded through
`locals()`, so a result from an earlier symbol or cycle could remain eligible.

`trading.data.market_frames.enrich_symbol_frames` now deep-copies and enriches
every fetched frame independently. The primary BTC view remains reserved for
the dashboard. Emergency fallback data remains outside the symbol mapping, so
it cannot become tradeable accidentally. At the start of each symbol pass,
the loop takes one local 100-row `symbol_history` copy and resets both temporary
filter results. High-win-rate analysis, regime detection, roadmap filters, and
volume sizing all consume that same symbol-local history; no load of the global
`data` alias remains inside the loop.

This slice does not change flags, thresholds, filter order, the per-symbol
regime cache used by open positions, or order submission. It also deliberately
does not repair the separate MACD-histogram alias or regime-label vocabulary;
those defects need their own tests and commits.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_symbol_market_data.py tests/test_technical_indicators_module.py tests/test_winrate_regime_units.py tests/test_signal_filters.py tests/test_position_sizing.py tests/test_rsi_policy_shadow.py tests/test_roadmap_wiring.py tests/test_main_order_submission.py
.venv/bin/black --target-version py310 --check trading/data/market_frames.py tests/test_symbol_market_data.py
.venv/bin/isort --check-only trading/data/market_frames.py tests/test_symbol_market_data.py
.venv/bin/flake8 trading/data/market_frames.py tests/test_symbol_market_data.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q trading/data/market_frames.py tests/test_symbol_market_data.py main.py
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused suite passes 182 tests. Its behavioural cases prove independent
RSI and volume outcomes from opposed BTC/BNB frames while preserving the raw
inputs. Missing, non-finite, short, partially enriched, or failed symbol frames
are omitted locally without suppressing a healthy peer. AST gates prove all
four consumers receive `symbol_history`, both temporary results reset before
use, and the symbol loop contains no read of the global BTC `data` alias. The
full suite passes 1,052 tests, skips 9, deselects 3, and keeps only the unchanged
local PostgreSQL baseline failure because `np.trades` is absent.

### M7e MACD histogram producer-contract implementation record

The roadmap divergence filter has always consumed a `macd_histogram` candle
column, but the active technical-indicator producer emitted only `macd` and
`signal_line`. The filter therefore treated every runtime frame as missing the
input and silently returned no divergence. The producer now emits
`macd_histogram` directly from `ta.trend.MACD.macd_diff()`. The per-symbol
market-frame boundary requires the new column, its latest finite value, and two
finite `close`/`macd_histogram` observations because the divergence consumer
compares a two-candle window. A partial calculation therefore cannot
reintroduce the silent fallback.

The change does not alter filter thresholds, override defaults, signal-policy
authority, or filter order. It only completes the producer/consumer contract
that the active roadmap gate already declared.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_technical_indicators_module.py tests/test_symbol_market_data.py tests/test_signal_filters.py tests/test_roadmap_wiring.py tests/test_rsi_policy_shadow.py
.venv/bin/black --target-version py310 --check trading/analysis/technical_indicators.py trading/data/market_frames.py tests/test_technical_indicators_module.py tests/test_symbol_market_data.py
.venv/bin/isort --check-only trading/analysis/technical_indicators.py trading/data/market_frames.py tests/test_technical_indicators_module.py tests/test_symbol_market_data.py
.venv/bin/flake8 trading/analysis/technical_indicators.py trading/data/market_frames.py tests/test_technical_indicators_module.py tests/test_symbol_market_data.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q trading/analysis/technical_indicators.py trading/data/market_frames.py tests/test_technical_indicators_module.py tests/test_symbol_market_data.py
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused suite passes 122 tests. It verifies the complete output schema and
the numerical identity `macd_histogram == macd - signal_line`, alongside the
existing divergence matrix, two-candle input boundary, and per-symbol wiring
gates. The full suite passes 1,054 tests, skips 9, deselects 3, and keeps only
the unchanged local PostgreSQL baseline failure because `np.trades` is absent.

### M7f take-profit regime producer-contract implementation record

The live cache currently stores `MarketRegimeDetector.regime.class`, whose
values (`optimal`, `favorable`, `neutral`, and `unfavorable`) combine trend,
volatility, momentum, liquidity, and trading-session quality. They cannot be
mapped honestly to the topological labels expected by `dynamic_take_profit`;
all four therefore fell through to its `RANGING` fallback. The detector's own
trend calculation also has a separate unit inconsistency and is not activated
as a take-profit source in this slice.

`HighWinRateFilter` already calculates a per-symbol topology on the same
`symbol_history` using the corrected whole-window trend strength. Its result
now includes a dedicated `take_profit_regime`, produced by a small pure
classifier: dailyized one-minute volatility at or above 5% yields `CHOPPY`;
strong or moderate trends yield `TRENDING`; known weak or ranging observations
yield `RANGING`;
unknown, malformed, boolean, non-finite, or negative observations yield
`None`. `REVERSAL` remains supported by the exit function but is deliberately
not produced until ELVIS has an explicit reversal detector.

This commit adds only the producer contract. `main.py` still uses its legacy
cache, so no fee-gate or exit target changes authority yet. The next slice can
observe the new field in shadow mode, clear stale values on missing analysis,
and make the fee gate fail closed before a separately reviewed cut-over.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_winrate_regime_units.py tests/test_roadmap_wiring.py tests/test_exits.py
.venv/bin/black --target-version py310 --check trading/analysis/high_winrate_filter.py tests/test_winrate_regime_units.py
.venv/bin/isort --check-only trading/analysis/high_winrate_filter.py tests/test_winrate_regime_units.py
.venv/bin/flake8 tests/test_winrate_regime_units.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q trading/analysis/high_winrate_filter.py tests/test_winrate_regime_units.py
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused suite passes 69 tests. It covers the complete profile matrix,
strict volatility boundary, malformed and non-finite inputs, the producer
field, short-history failure, and the invariant that the classifier never
emits `REVERSAL` or a quality label. The full suite passes 1,072 tests, skips 9,
deselects 3, and keeps only the unchanged local PostgreSQL baseline failure
because `np.trades` is absent.

### M7g take-profit regime shadow implementation record

`ELVIS_TP_REGIME_MODE=shadow` now compares the purpose-specific
`take_profit_regime` produced by `HighWinRateFilter` with the effective legacy
behaviour. The legacy cache continues to receive the quality label, so labels
outside the supported TP vocabulary resolve to the existing `RANGING`
fallback. The observer records that effective profile, the candidate profile,
availability, and match status in a bounded structured event. Its evaluation
identifier is explicitly shadow-only and is not presented as an order ID.

The default is `legacy`; unknown values also keep legacy behaviour. The shadow
call is a standalone expression after the authoritative high-win-rate analyses.
It returns `None`, swallows ordinary calculation and logging failures, and has
no reference to executors, order submission, cooldown, feedback, persistence,
the fee gate, the open-position cache, or exit state. This slice deliberately
does not rename or update `_last_regime`, change a take-profit target, or make a
missing candidate reject an order. Stale-cache removal and fail-closed active
consumption remain the next separately reviewed cut-over.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_take_profit_regime_shadow.py tests/test_winrate_regime_units.py tests/test_roadmap_wiring.py tests/test_main_order_submission.py
.venv/bin/black --target-version py310 --check tests/test_take_profit_regime_shadow.py tests/test_winrate_regime_units.py
.venv/bin/isort --check-only tests/test_take_profit_regime_shadow.py tests/test_winrate_regime_units.py
.venv/bin/flake8 tests/test_take_profit_regime_shadow.py tests/test_winrate_regime_units.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q main.py tests/test_take_profit_regime_shadow.py
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused suite passes 52 tests. Behavioural tests cover legacy matches,
expected TRENDING/CHOPPY divergences, unavailable candidates, sanitized logger
failure, and exact structured fields. AST gates prove the call is opt-in,
non-assigned, before the sole typed submission, incapable of adding a stateful
dependency, and leaves the authoritative quality-label cache unchanged. The
full suite passes 1,087 tests, skips 9, deselects 3, and keeps only the unchanged
local PostgreSQL baseline failure because `np.trades` is absent.

### M7h fee-gate take-profit regime cut-over implementation record

`ELVIS_TP_REGIME_MODE=active` now makes the pre-submission fee gate consume
only the purpose-specific profile produced during the current symbol
evaluation. The active boundary accepts exactly `TRENDING`, `RANGING`, or
`CHOPPY`; it rejects `REVERSAL` because no current producer can establish that
state. Missing, stale, malformed, differently cased, boolean, or otherwise
unproduced profiles yield `None`, set the signal to `HOLD`, and never reach
`dynamic_take_profit` or `is_trade_viable`.

The active branch does not read `_last_regime`. Consequently, disabling the
high-win-rate analysis, entering below its initial confidence guard, or hitting
an analysis exception leaves the per-symbol profile at its explicit `None`
reset and blocks a later submission even if another downstream component makes
the signal actionable. The existing fee-gate exception handler remains
fail-closed. `legacy`, `shadow`, unknown modes, and the default value retain the
legacy fee-gate behaviour for rollback.

This slice changes only the new-order fee gate. The quality-label cache and the
open-position exit block remain legacy and are not silently recast as a
position-bound regime. M8b now represents an immutable entry-time profile in a
pure position instruction. Resolving and associating that profile in the live
submission/fill path remains part of the future `PositionService` cut-over.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_take_profit_regime_cutover.py tests/test_take_profit_regime_shadow.py tests/test_winrate_regime_units.py tests/test_roadmap_wiring.py tests/test_main_order_submission.py tests/test_fee_gate.py
.venv/bin/black --target-version py310 --check tests/test_take_profit_regime_cutover.py tests/test_take_profit_regime_shadow.py
.venv/bin/isort --check-only tests/test_take_profit_regime_cutover.py tests/test_take_profit_regime_shadow.py
.venv/bin/flake8 tests/test_take_profit_regime_cutover.py tests/test_take_profit_regime_shadow.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q main.py tests/test_take_profit_regime_cutover.py
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused suite passes 135 tests. Unit tests lock the exact active vocabulary
and reject every unproduced input. AST gates prove lazy cache access, current
local provenance, `HOLD` before either fee calculation on invalid input, the
single downstream typed submission, and an unchanged legacy exit cache. The
full suite passes 1,101 tests, skips 9, deselects 3, and keeps only the unchanged
local PostgreSQL baseline failure because `np.trades` is absent.

### M8a pure order-lifecycle implementation record

`trading.domain.order_lifecycle` now defines one immutable projection for what
ELVIS knows about an order. Its explicit states are `PENDING`, `RECONCILING`,
`OPEN`, `PARTIAL`, `CANCEL_PENDING`, `CANCELLED`, `FILLED`, and `FAILED`.
Submission acknowledgement remains distinct from a fill: the pure mapper turns
a `SubmissionReport.SUBMITTED` into `SubmissionAcknowledged` even when the
legacy `venue_status` string says `FILLED`. An ambiguous submission becomes
`RECONCILING`; a proven `NOT_SENT` or `VENUE_REJECTED` result becomes `FAILED`
while retaining the original status and orthogonal retry-safety value.

Confirmed fills carry correlated client, venue, trade, symbol, and side IDs,
exact quantity/price/fee `Decimal` values, and an injected aware venue
timestamp. A positive fee requires its asset; a zero fee may omit it. Fill
totals and remaining quantity use an isolated, explicitly dimensioned decimal
context with inexact arithmetic trapped. They therefore do not inherit a
caller's decimal precision and cannot silently round an overfill into an
accepted quantity. Fills are deduplicated by `trade_id`; an exact duplicate is
an identity no-op, while a conflicting payload or overfill raises
`InvalidOrderTransition`. Canonical trade-ID ordering makes supported event
permutations converge without sorting or rejecting by venue timestamps.

Cancellation events carry a stable `cancel_request_id`. The lifecycle retains
that ID only while `CANCEL_PENDING`, so a delayed response for attempt A cannot
clear a newer attempt B. Partial fills during cancellation remain
cancel-pending; late fills after confirmed cancellation are still counted, and
an exact full fill wins over cancellation. The reducer has no clock, UUID,
logger, environment read, I/O, database, executor, or runtime consumer. M8b's
pure position reducer and M9b.2/M9b.3 persistence modules are its infrastructure
consumers; M9b.4's coordinator is its only application consumer. None is wired
to the runtime.

This is deliberately not a runtime cut-over. ELVIS does not yet provide a
reliable fill stream, a truthful paper cancellation adapter, or startup
reconciliation. M8b now supplies explicit `OPEN`/`REDUCE_ONLY` position effects
and entry-time exit context, still without runtime wiring. M9b.1 prepares the
correlated order/event schema and uniqueness constraints, M9b.2 adds strict
lossless codecs, and M9b.3 adds committed repository writes plus reducer-based
restart replay. M9b.4 adds an unwired register-before-submit owner; later M9b
slices must add reconciliation, durable quarantine, and safe runtime composition
before lifecycle ownership can be described as complete.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_order_lifecycle.py tests/test_domain_contracts.py tests/test_order_service.py tests/test_legacy_paper_adapter.py
.venv/bin/black --target-version py310 --check trading/domain/_validation.py trading/domain/order_lifecycle.py trading/domain/__init__.py tests/test_order_lifecycle.py
.venv/bin/isort --check-only trading/domain/_validation.py trading/domain/order_lifecycle.py trading/domain/__init__.py tests/test_order_lifecycle.py
.venv/bin/flake8 trading/domain/_validation.py trading/domain/order_lifecycle.py trading/domain/__init__.py tests/test_order_lifecycle.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q trading/domain tests/test_order_lifecycle.py
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The M8a implementation snapshot recorded 262 focused tests. It covered every
non-fill transition, exact and conflicting duplicates, large-precision full
fills and overfills under different ambient decimal contexts, correlation
mismatches, state-construction invariants, stale cancellation responses, late
fills, and ACK/fill/cancel permutations. Its full-suite snapshot recorded 1,238
passes, 9 skips, 3 deselections, and only the unchanged local PostgreSQL
baseline failure because `np.trades` was absent. M8b subsequently became the
first internal consumer. M9b.2/M9b.3 permit the exact persistence codec and
repository modules as infrastructure consumers, and M9b.4 permits the exact
application coordinator; there is still no runtime consumer.

### M8 prerequisite: make legacy exits one-shot and transactional

The still-active paper exit path now closes one legacy position row under a
row lock. Its closing-trade insert and exact-ID position delete use the same
PostgreSQL connection and transaction; a delete count other than one, an
invalid value, a missing row, or any database error rolls the transaction back
and returns `False`. A successful commit returns `True`, and later diagnostic
logging cannot change that committed outcome. This removes the earlier split
transaction where a trade could survive without its position deletion, or a
position could be deleted after the trade helper silently failed.

The inline stop-loss, trailing-stop, and take-profit branches now end the
current position iteration only after that committed `True` result. Their
success messages use values that exist in the position tuple. The optional
Balanced Starter owner also increments its close counters and reports success
only when the same helper returns `True`; its six direct callers no longer turn
a database failure into a reported close. A concurrent PostgreSQL regression
proves that two closers produce one committed trade and one `True` result, and
a foreign-key fault proves that a failed delete rolls the preceding insert
back.

This is a containment fix, not the M8 cut-over. The legacy tables still use
`REAL`, carry no position key/effect or entry-time exit context, and cannot be
replayed from confirmed fills. Balanced Starter and the inline loop remain
separate position owners. M8b defines those missing pure domain contracts;
M9b.1 prepares their journal tables and M9b.3 now appends and replays directives
and confirmed fills transactionally. A single future `PositionService` must
still replace these owners. A lost connection while `COMMIT` is being
acknowledged also remains an ambiguous outcome that only durable reconciliation
can resolve; this boolean helper is not an exactly-once protocol.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_position_exit_control_flow.py tests/test_paper_db_schema.py tests/test_paper_fill_integrity.py tests/test_stop_loss_threshold.py tests/test_exits.py tests/test_main_order_submission.py tests/test_roadmap_wiring.py tests/test_take_profit_regime_cutover.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<admin-dsn> ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres tests/postgres
.venv/bin/black --target-version py310 --check utils/paper_trade_db.py trading/strategies/balanced_starter.py tests/test_position_exit_control_flow.py tests/postgres/test_position_close_postgres.py
.venv/bin/isort --check-only utils/paper_trade_db.py trading/strategies/balanced_starter.py tests/test_position_exit_control_flow.py tests/postgres/test_position_close_postgres.py
.venv/bin/flake8 tests/test_position_exit_control_flow.py tests/postgres/test_position_close_postgres.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q main.py utils/paper_trade_db.py trading/strategies/balanced_starter.py tests/test_position_exit_control_flow.py tests/postgres/test_position_close_postgres.py
env -u ELVIS_TEST_POSTGRES_ADMIN_DSN -u ELVIS_TEST_POSTGRES_REQUIRED \
  .venv/bin/python -m pytest -q tests/ -m 'not perf and not postgres'
env -u ELVIS_TEST_POSTGRES_ADMIN_DSN -u ELVIS_TEST_POSTGRES_REQUIRED \
  TZ=UTC .venv/bin/python -m pytest -q tests/ -m 'not perf and not postgres'
```

The focused legacy suite passes 97 tests and skips one optional integration
case. The isolated PostgreSQL 15 suite passes all 9 tests, including the three
close-transaction cases. The local non-PostgreSQL suite passes 1,238 tests,
skips 49, and deselects 12. Its only failure at 00:15 CEST is the pre-existing
time-dependent `test_check_new_day_same_day`: it subtracts two hours from wall
time across midnight and still asserts that the date did not change. The same
suite under UTC passes 1,239 tests with the same 49 skips and 12 deselections;
this slice does not alter the risk-manager clock.

### M8b pure position-effect implementation record

`trading.domain.positions` now defines the immutable position projection that
the future `PositionService` must own. A `PositionInstruction` binds one stable
`position_key`, an explicit `OPEN` or `REDUCE_ONLY` effect, and the approved
`OrderIntent` before submission. `OPEN` also captures a typed entry-time
`PositionExitContext`; `REDUCE_ONLY` cannot replace it. `PositionSide.LONG` and
`SHORT` remain distinct from order direction, so `OPEN BUY` creates or scales a
long position, `OPEN SELL` creates or scales a short position, and a reduction
must use the opposite order side. A reduction never creates a position, clips
an over-reduction, or flips the remainder.

The `position_fill_from_lifecycle` boundary binds only a `ConfirmedFill` already
present in the matching `OrderLifecycle`; replay may reconstruct the same
validated immutable value from a durable confirmed event. Position-local fill
identity is the composite `(client_order_id, trade_id)`; exact duplicates are
identity no-ops, including after closure, while conflicting payloads fail.
Partial fills for one client order must retain the exact instruction and venue
order ID and cannot exceed that intent's quantity. Canonical composite-ID
ordering makes independent arrival orders converge. Opened, reduced, and
remaining quantities reuse M8a's isolated exact-`Decimal` arithmetic, so
ambient precision, rounding mode, and traps cannot hide a per-order overfill or
global over-reduction.

The aggregate retains the opening leverage and exit context. Scale-ins must
match both; reductions deliberately do not compare their order leverage because
it is not a position quantity effect and may differ after configuration or
venue changes. An exact reduction closes the key; a new fill after closure is a
conflict rather than an implicit reopen. M8b does not calculate average entry,
realised PnL, fee conversion, lot allocation, or exit triggers.

This slice remains pure and unused by the runtime. The package facade re-exports
the contracts, so loading another `trading.domain` submodule may load the file;
M9b.2's codec, M9b.3's repository, and M9b.4's application coordinator are the
only production modules outside `trading.domain` allowed to reference its
journal values. An import-aware AST
gate covers direct, facade, relative, aliased, and literal dynamic imports
without confusing the generic name `Position` with unrelated classes. The
legacy inline and Balanced Starter owners, paper executor, and positional
PostgreSQL tables remain authoritative. M9b.1 prepares the journal schema,
M9b.2 encodes its instruction and event payloads, M9b.3 appends and replays them,
and M9b.4 coordinates a committed reservation and one submission without runtime
wiring; later slices must reconcile ambiguous submissions before any cut-over.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_position_lifecycle.py tests/test_order_lifecycle.py tests/test_domain_contracts.py tests/test_risk_decision.py
.venv/bin/black --target-version py310 --check trading/domain tests/test_position_lifecycle.py tests/test_order_lifecycle.py
.venv/bin/isort --check-only trading/domain tests/test_position_lifecycle.py tests/test_order_lifecycle.py
.venv/bin/flake8 trading/domain/_decimal.py trading/domain/positions.py tests/test_position_lifecycle.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q trading/domain tests/test_position_lifecycle.py
env -u ELVIS_TEST_POSTGRES_ADMIN_DSN -u ELVIS_TEST_POSTGRES_REQUIRED \
  TZ=UTC .venv/bin/python -m pytest -q tests/ -m 'not perf and not postgres'
```

The M8b implementation snapshot recorded 322 focused tests. It covered exact
Decimal boundaries, open/scale/reduce/close transitions, long and short
directions, fill and lifecycle correlation, per-order caps, canonical
permutations, duplicate and terminal precedence, direct-construction
invariants, and zero runtime consumers. Its isolated non-PostgreSQL snapshot
recorded 1,317 passes, 49 skips, and 12 deselections under UTC. M9b.2/M9b.3 now
permit the exact codec and repository consumers, and M9b.4 permits the exact
application coordinator, while preserving the zero-runtime-consumer gate.

### M9 prerequisite: preserve open positions during initialization

`utils.paper_trade_db.init_db()` no longer drops `np.open_positions` during
ordinary executor construction. Within an already existing `np` schema, its
table setup remains additive through `CREATE TABLE IF NOT EXISTS`; only the
explicit paper-reset workflow calls `clear_open_positions()`. This closes the
immediate restart-data-loss hazard without claiming that the legacy initializer
can bootstrap the schema, or yet introducing a repository or changing the
positional legacy schema.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_paper_db_schema.py tests/test_paper_fill_integrity.py
.venv/bin/black --target-version py310 --check utils/paper_trade_db.py tests/test_paper_db_schema.py
.venv/bin/isort --check-only utils/paper_trade_db.py tests/test_paper_db_schema.py
.venv/bin/flake8 tests/test_paper_db_schema.py --max-line-length=88
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused suite passes 10 tests. A PostgreSQL 15 container smoke test creates
and seeds `np.open_positions`, invokes `init_db()` a second time, and confirms
that the row remains. The container is isolated from operator databases and is
removed after the check. The full suite passes 1,239 tests, skips 9, deselects
3, and keeps only the unchanged local PostgreSQL baseline failure because
`np.trades` is absent. The versioned runner is introduced in M9a below; the
order journal still requires transaction, uniqueness, replay, and idempotency
tests now that the M8b position-effect contract is stable.

### M9a versioned PostgreSQL baseline

`trading.persistence.migration_runner` loads ordered SQL through
`importlib.resources`, calculates an immutable SHA-256 digest, rejects an empty
catalogue and missing, duplicate, reordered, unknown, modified, or non-prefix
versions, and applies pending work under a PostgreSQL transaction advisory lock.
It requires a dedicated psycopg2 connection in the ready/`IDLE` state and
establishes `READ COMMITTED` before acquiring that lock, so a concurrent waiter
observes the winner's commit instead of an older snapshot. Migration metadata,
every pending DDL statement, and its version record commit together; any failure
rolls back the complete pending sequence. The runner refuses autocommit, a
prepared connection, or an existing caller transaction and never closes a
caller-owned connection. A small SQL lexer rejects top-level transaction or
session-setting commands while correctly ignoring quoted text, CR/LF comments,
nested block comments, and dollar-quoted procedural bodies, so packaged SQL
cannot commit outside the runner's boundary. Standard string syntax is restored
immediately before every migration.

The runner creates `np` and its migration ledger as transaction metadata; the
packaged `0001_legacy_baseline.sql` then creates the current legacy tables
additively without a `DROP`, rename, or destructive backfill. Existing
positional column order and `REAL`/naive timestamp types are preserved for
compatibility. Before recording version 1, the migration validates exact column
order/types/nullability/default semantics, ordinary permanent relations,
required non-deferrable constraints, and the two contract indexes. It refuses
an incompatible legacy or migration-ledger layout, including unlogged state,
non-durable serial sequences, triggers, rules, RLS, policies, or inheritance.
Each version record uses `INSERT ... RETURNING`; after deferred constraints are
forced, the ledger contract and complete ordered history are revalidated before
commit. This
baseline is schema-only: it never inserts USDT, BNB, or other business state.
Later repositories will use typed exact storage instead of silently changing
these consumers. The SQL file is explicit package data and was verified inside
the built wheel. Nothing imports the runner from `main.py`, bootstrap, or the
legacy database helper in this slice.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_migration_runner.py
.venv/bin/black --target-version py310 --check trading/persistence tests/test_migration_runner.py
.venv/bin/isort --check-only trading/persistence tests/test_migration_runner.py
.venv/bin/flake8 trading/persistence tests/test_migration_runner.py --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q trading/persistence tests/test_migration_runner.py
/usr/local/bin/python3.10 -m pip wheel --no-deps --wheel-dir <temporary-directory> .
.venv/bin/python -m pytest tests/ -q -m 'not perf'
```

The focused suite passes 35 tests. An isolated PostgreSQL 15 smoke applies the
baseline from an empty database, verifies an idempotent second run and no balance
seed, serializes two concurrent runners, and proves that rejecting a caller's
active transaction neither commits nor rolls it back. It upgrades an exact
legacy schema while preserving an `open_positions` sentinel; incompatible
defaults, indexes, deferrable constraints, `UNLOGGED` state, and ledger layouts
are rejected without recording version 1. A suppressed ledger insert and a
top-level `COMMIT` attempt are also rejected. A failing second migration rolls
back both its DDL and version. A post-baseline `orders` table and `position_key`
extension survive the next no-op version-1 run, proving that M9a does not freeze
future namespaces. M9b.1 now claims its three new table names explicitly and
rejects any pre-existing collision instead of silently adopting it. The full
suite passes 1,274 tests, skips 9, deselects 3,
and keeps only the known ambient-database test failure; the next atomic
test-infrastructure
commit moves that test to a required isolated PostgreSQL CI job rather than
masking it.

### M9a isolated PostgreSQL test gate

Database-backed tests now require the explicit
`ELVIS_TEST_POSTGRES_ADMIN_DSN` contract. The harness creates a fresh
`elvis_pytest_<uuid>` database for each test and drops only that exact database
afterward. It never reads `DATABASE_URL`, the runtime `POSTGRES_*`/`DB_*`
variables, `.env`, or Vault. An absent DSN skips PostgreSQL tests locally; with
`ELVIS_TEST_POSTGRES_REQUIRED=1`, absence or connection/setup failure fails the
suite. Merely adding the `postgres` marker is insufficient: the test must also
consume the isolated database fixture. Pytest also forces `VAULT_ENABLED=false`
and `PYTHON_DOTENV_DISABLED=1` before application modules are imported.
Collection-time and background-thread database connections fail closed with
`OperationalError`; during an ordinary test, a main-thread connection becomes a
visible skip. Legacy diagnostics that used to catch an error and return `False`
can therefore no longer contact an operator database or count that attempt as a
passing database check.

The PostgreSQL suite applies the packaged baseline on an empty database, checks
its no-op replay and lack of balance seed, adopts an exact unversioned legacy
layout without losing an open-position sentinel, rolls back a failing second
migration with its ledger, and serializes two concurrent runners. The adaptive
feedback SQL regression now inserts deterministic rows directly into the
disposable database and proves a zero-PnL close is returned before a later
profitable close; it no longer discovers or deletes rows in the ambient local
database. CI separates the fast unit job (`not perf and not postgres`) from a
required PostgreSQL 15/Python 3.10 job, and image publication depends on both.
The migration runner remains unused by production startup.

Verification at implementation time:

```bash
env -u ELVIS_TEST_POSTGRES_ADMIN_DSN -u ELVIS_TEST_POSTGRES_REQUIRED \
  .venv/bin/python -m pytest -q -ra tests/postgres
ELVIS_TEST_POSTGRES_ADMIN_DSN=<admin-dsn> ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres tests/postgres
.venv/bin/python -m pytest -q tests/ -m 'not perf and not postgres'
/usr/local/bin/python3.10 -m compileall -q tests/postgres tests/conftest.py
```

The first command skips explicitly without a DSN. The second command runs only
against a disposable PostgreSQL database supplied by the test harness. Legacy
script-like diagnostics that still have no assertions are quarantined as
visible skips when they attempt PostgreSQL; converting or removing those files
is a separate cleanup and does not weaken this connection boundary.
The boolean-returning diagnostics in `test_database_connection.py`,
`test_database_integration.py`, and `test_vault_integration.py` are no longer
collected as pytest tests. The similarly procedural `test_vault_connection.py`,
which can write a Vault test secret, is also manual-only. All four retain their
existing `__main__` entry points.
The six PostgreSQL tests pass both in the project environment and in a minimal
Python 3.10 environment against PostgreSQL 15; teardown leaves zero
`elvis_pytest_*` databases. With no test DSN, the full non-performance suite
passes 1,225 tests, explicitly skips 49, and deselects 9. Setting the required
flag without a DSN fails setup as designed.

### M9b.1 prepared order/position journal schema

The forward-only `0002_order_position_journal.sql` migration adds three
separate, durable tables without altering, backfilling, or reading the legacy
`np.trades` and `np.open_positions` layouts. `np.position_streams` owns a
globally unique, stable `position_key`, execution scope, and future stream
version.
`np.orders` records the correlated client/decision IDs, position effect,
execution scope, and a versioned `PositionInstruction` JSON envelope plus its
payload hash. A decision ID may reserve only one order per execution scope.
`np.order_events` records the seven M8a lifecycle fact types under
a `position_version` intended for monotonic allocation, with stable event and
confirmed-fill identities. There is deliberately no mutable `np.positions`
table: the M8b position remains a projection to rebuild by replay.

Indexed identifiers are bounded; M9b.2's codec rejects the domain-sourced values
it cannot represent, while M9b.3's repository enforces its own scope and event
identifiers before persistence. SQL rejects empty and
ordinary space-padded identifiers, while the codec enforces the complete domain
clean-text rule. Venue order IDs are unique only within `(execution_scope,
symbol)`, rather than incorrectly assumed global across accounts or adapters.
Event identity is unique per client order, and confirmed-fill identity is the
M8b composite `(client_order_id, trade_id)`.
Version columns reject unknown envelope versions, payload columns require JSON
objects, and foreign keys prevent events from escaping their order and position
stream. The JSON objects are storage envelopes, not a permissive serializer.
M9b.2's strict codec encodes every exact `Decimal` as a JSON string and validates
the full typed payload and SHA-256 digest before M9b.3 writes.
PostgreSQL `NUMERIC` and JSON numbers are intentionally absent because their
range is narrower than the already accepted pure-domain Decimal range.

M9b.1 itself prepares storage only, and the migration runner remains unwired to
startup. M9b.2 supplies the lossless codecs, and M9b.3 supplies the transaction
boundary, committed reservation, per-position locking/version allocation, and
reducer-based replay. M9b.4 now supplies the unwired application owner for
register-before-submit and submission-observation persistence. Later slices
must add reconciliation, durable quarantine, startup readiness, and safe runtime
composition. Until those gates pass, the legacy executor and positional tables
remain authoritative. Rolling application code back leaves these unused,
additive tables in place; there is no destructive down migration.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q tests/test_migration_runner.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<admin-dsn> ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres tests/postgres
.venv/bin/black --target-version py310 --check trading/persistence tests/test_migration_runner.py tests/postgres
.venv/bin/isort --check-only trading/persistence tests/test_migration_runner.py tests/postgres
.venv/bin/flake8 trading/persistence tests/test_migration_runner.py tests/postgres --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q trading/persistence tests/test_migration_runner.py tests/postgres
env -u ELVIS_TEST_POSTGRES_ADMIN_DSN -u ELVIS_TEST_POSTGRES_REQUIRED \
  TZ=UTC .venv/bin/python -m pytest -q tests/ -m 'not perf and not postgres'
/usr/local/bin/python3.10 -m pip wheel --no-deps --wheel-dir <temporary-directory> .
unzip -l <temporary-directory>/elvis_trading_bot-*.whl | rg '000[12]_.*\.sql'
```

The migration unit suite passes 35 tests. The isolated PostgreSQL 15 suite
passes 21 tests from fresh and version-1 databases. It preserves a legacy
sentinel during the version-2 upgrade, rejects a pre-existing incompatible
`np.orders` relation with a complete version-2 rollback, and validates scoped
venue identity, per-position version keys, correlated fills, and timezone-aware
timestamps. It also demonstrates an exact Decimal string beyond the PostgreSQL
numeric scale round-tripping unchanged. These are schema constraints, not
repository idempotency, concurrency, or replay claims. The UTC non-PostgreSQL
suite passes 1,317 tests, skips 49, and deselects 24. The Python 3.10 wheel
contains both immutable SQL migrations; its generated local build tree was
removed after inspection.

### M9b.2 pure lossless journal codecs

M9b.2 adds `trading.persistence.journal_codec`, a pure persistence codec for the
version-1 `PositionInstruction` envelope and each of the seven M8a lifecycle
event types. Its public records are the frozen, slotted
`EncodedPositionInstruction` and `EncodedOrderLifecycleEvent`; explicit
`encode_position_instruction`, `decode_position_instruction`,
`encode_order_lifecycle_event`, and `decode_order_lifecycle_event` functions
translate domain values and the corresponding untrusted envelope columns
without a generic dataclass serializer. The encode functions return the named
records; the keyword-only decode functions validate the version, payload,
digest, and duplicated indexed columns before returning a domain value.
`JournalCodecError` is their common failure boundary:
`JournalEncodeError` rejects a domain value before persistence, and
`JournalQuarantineError` rejects unknown, corrupt, or inconsistent stored data.
The codec does not serialize `RiskDecision`, `SubmissionReport`,
`OrderLifecycle`, `Position`, or either reducer: reports are translated to typed
events before the journal boundary, while order and position projections remain
replay products. The codec has no database, executor, environment, clock, UUID,
or runtime dependency. M9b.3 is now its only database consumer.

The version-1 wire contract uses exact JSON object shapes, sorted keys, compact
separators, and ASCII escaping. Every `Decimal` and the positive integer
leverage are strings, so values such as `1E-20000` and significant trailing
zeroes survive encode/decode without passing through a JSON or PostgreSQL
number. Aware datetimes are normalized to UTC and encoded with six fractional
digits and the explicit `+00:00` offset. Enum values, optional nulls, and
duplicated relational fields are validated rather than coerced. The SHA-256
value covers the UTF-8 canonical payload bytes: it detects corruption and
inconsistency, but is not an authenticity proof, signature, or MAC.

Decode rejects unknown envelope or event versions, unknown event types,
malformed or non-object JSON, missing or additional fields, invalid scalar
types, non-canonical exact values, payload/hash mismatches, and conflicts
between payload and duplicated columns. These observations are quarantine
inputs and must never be replayed as partial or best-effort domain values. The
version-1 format is immutable; changing its keys, types, canonicalization, or
digest contract requires a new version and a matching schema migration.

Encode also enforces the schema bounds for domain-sourced client order,
decision, position, symbol, venue-order, and trade identifiers. NUL characters
and isolated Unicode surrogates are not representable in PostgreSQL JSONB and
fail before persistence. The codec deliberately invents no storage limit for
unindexed payload-only reason, cancellation-request, or fee-asset text beyond
clean Unicode representability.

M9b.2 itself remains a representation slice. M9b.3 supplies the repository,
stream lock, `position_version` allocation, append transaction, and replay;
M9b.4 adds the unwired register-before-submit owner. There is still no
reconciliation path, durable quarantine, startup readiness gate, or runtime
cut-over. Activation has an additional adapter gate: every
`venue_order_id` and `trade_id` emitted by an execution adapter must already be
clean Unicode and no longer than the schema limit of 255 characters. A codec
failure discovered only after the venue effect or fill observation would be too
late to make the active path safe.

Verification must cover golden version-1 JSON and SHA-256 vectors, lossless
round trips for `OPEN`, `REDUCE_ONLY`, and all seven event types, malformed and
tampered envelopes, metadata correlation, storage bounds, Python 3.10
compatibility, and an import-aware AST gate. M9b.3 advances that gate to permit
the exact codec and repository modules; M9b.4 permits the exact application
coordinator while proving that no adapter, composition root, or runtime module
uses the migrated boundary. PostgreSQL
integration remains the M9b.1 schema gate; M9b.2 adds no SQL or database I/O.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q \
  tests/test_journal_codec.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py \
  tests/test_domain_contracts.py
/usr/local/bin/python3.10 -m pytest -q tests/test_journal_codec.py
.venv/bin/python -m pytest -q tests/test_migration_runner.py
.venv/bin/black --target-version py310 --check \
  trading/persistence/journal_codec.py \
  tests/test_journal_codec.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py
.venv/bin/isort --check-only \
  trading/persistence/journal_codec.py \
  tests/test_journal_codec.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py
.venv/bin/flake8 \
  trading/persistence/journal_codec.py \
  tests/test_journal_codec.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/persistence/journal_codec.py \
  tests/test_journal_codec.py
env -u ELVIS_TEST_POSTGRES_ADMIN_DSN \
  -u ELVIS_TEST_POSTGRES_REQUIRED \
  TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The focused domain/codec suite passes 412 tests, and the codec-only suite
passes 95 tests on Python 3.10. The unchanged migration-runner regression suite
passes 35 tests. Black, isort, flake8, Python 3.10 compilation, and the diff
check pass. The isolated non-PostgreSQL suite passes 1,429 tests, skips 49, and
deselects 24 under `Pacific/Honolulu`. A separate UTC run before 02:00 UTC
reports the pre-existing time-boundary failure
`tests/test_risk_manager.py::TestRiskManager::test_check_new_day_same_day`
because that test subtracts two hours and still expects the same calendar day;
it otherwise reports 1,428 passes, 49 skips, and 24 deselections. M9b.2 does
not change that test or the risk manager. No PostgreSQL run is required for
this pure slice because neither migration nor schema changed.

### M9b.3 transactional order/position repository and replay

M9b.3 adds the unwired `trading.persistence.order_position_journal` boundary.
`PostgresOrderPositionJournal` receives only an injected connection factory and
offers three explicit operations: `reserve_instruction`, `append_event`, and
`replay_position`. Every call obtains a fresh, idle, `autocommit=False`
connection. Writes use `READ COMMITTED`, own rollback/commit/close, and return a
`ReservationCommit` or `EventCommit` only after PostgreSQL acknowledges the
commit. A lost commit acknowledgement raises `JournalCommitUnknown`; it is never
reported as a reservation or append success. Repeating the same stable identity
on a fresh connection resolves whether the fact exists without an automatic
external retry.

Reservation creates or locks one globally stable position stream, verifies its
execution scope, decodes and replays any existing rows, and inserts the exact
version-1 instruction envelope. `CREATED` is the only disposition M9b.4's
register-before-submit owner treats as a new reservation. `EXISTING` means a
canonical reservation already exists and must be reconciled rather than blindly
resubmitted. Client-order and scoped decision conflicts roll the transaction
back, including any transient stream row, so a losing registration cannot
commit an empty stream. The insert uses targetless conflict handling because two
concurrent exact reservations may race on either the primary key or the
equivalent scoped identity constraint; the locked replay still verifies that
the winning stream has the exact execution scope.

Append locks the `position_streams` row before resolving identities or
allocating a version. It compares complete canonical envelopes, not domain
numeric equality: significant Decimal representation such as `1.0` versus
`1.00` cannot mutate an existing fact. The stable `(client_order_id, event_id)`
identity deduplicates every observation, while confirmed fills also use
`(client_order_id, trade_id)`; an exact fill observed under another event ID
returns the original durable event/version, and a different envelope conflicts.
After the current stream passes replay, the candidate is applied to the M8a
order reducer and, for a confirmed fill, the M8b position reducer. Historical
venue correlation, event insert, and `stream_version + 1` update commit in one
transaction. Allocation never uses `MAX(position_version)`.

Replay uses one `REPEATABLE READ READ ONLY` snapshot, requires versions to be
the exact prefix `1..stream_version`, decodes every instruction and event, and
applies facts in global `position_version` order even when client IDs or venue
timestamps sort differently. It returns frozen, named `ReplayedOrder`,
`JournalEventRecord`, and `PositionStreamProjection` values only after the whole
history succeeds. Bad hashes, missing versions, conflicting venue identities,
unknown orders, or reducer-invalid history raise `JournalReplayError`; no row is
skipped and no partial projection escapes.

This is still not a runtime cut-over. The repository is not exported through
the persistence facade, and M9b.4 depends only on its structural application
port rather than importing this PostgreSQL module. No adapter, composition root,
startup, or runtime module imports it. There is no durable quarantine record yet;
M9b.3 promises typed fail-closed detection only. A forward migration and a
separate post-rollback workflow are required before the word "quarantined" can
describe durable state. Before activation, a pre-submission validator must also
reject incompatible scale-ins and unavailable `REDUCE_ONLY` capacity, and the
application owner must prevent or reserve collectively unsafe concurrent
orders, reconcile unresolved reservations, and guarantee the adapter's
venue/trade identifier bounds. Append currently performs two complete stream
replays under the position lock, so snapshots or another bounded replay strategy
are also required before activation; M9b.3 treats correctness as authoritative
and makes no production-throughput claim.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q \
  tests/test_order_position_journal.py \
  tests/test_journal_codec.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py \
  tests/test_domain_contracts.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres \
  tests/postgres/test_order_position_repository_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres tests/postgres
.venv/bin/black --target-version py310 --check \
  trading/persistence/order_position_journal.py \
  tests/test_order_position_journal.py \
  tests/postgres/test_order_position_repository_postgres.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py
.venv/bin/isort --check-only \
  trading/persistence/order_position_journal.py \
  tests/test_order_position_journal.py \
  tests/postgres/test_order_position_repository_postgres.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py
.venv/bin/flake8 \
  trading/persistence/order_position_journal.py \
  tests/test_order_position_journal.py \
  tests/postgres/test_order_position_repository_postgres.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/persistence/order_position_journal.py \
  tests/test_order_position_journal.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_order_position_journal.py
env -u ELVIS_TEST_POSTGRES_ADMIN_DSN \
  -u ELVIS_TEST_POSTGRES_REQUIRED \
  TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The focused repository, codec, lifecycle, position, and domain-contract suite
passes 446 tests. The repository unit suite passes 34 tests on Python 3.10.
The repository PostgreSQL 15 matrix passes 10 tests, including concurrent
distinct/exact appends and competing `REDUCE_ONLY` fills; the complete isolated
PostgreSQL suite passes 31 tests. Black, isort, flake8, Python 3.10 compilation,
and the diff check pass. The isolated non-PostgreSQL suite passes 1,463 tests,
skips 49, and deselects 34 under `Pacific/Honolulu`.

### M9b.4 unwired journaled submission coordinator

M9b.4 adds `trading.application.journaled_order_service` as the single
application owner of the prepared register-before-submit sequence. It accepts a
complete `PositionInstruction` and depends on a storage-neutral
`OrderJournalPort`, the existing single-call `OrderService`, and an injected
wall clock. The application package has no PostgreSQL or persistence import.
The concrete M9b.3 `ReservationCommit` exposes only the structural
`is_created` receipt needed by this boundary.

Only a committed receipt whose `is_created` value is exactly `True` authorizes
an external call. A reservation error propagates before the clock or executor.
An exact existing reservation returns `EXISTING_RESERVATION` and
`requires_reconciliation=True` without reading the clock, calling the executor,
or appending another event. This conservative result applies regardless of the
stored lifecycle because M9b.4 deliberately does not couple the application
port to a PostgreSQL projection. A created reservation followed by an invalid
or failed clock also makes no external call and remains work for the future
reconciler rather than becoming permission to resubmit.

For a new reservation, the aware observation time is validated before the
external effect. `OrderService` then invokes its adapter once, and its exact
`SubmissionReport` is translated to one M8a submission event. The stable event
identity is `submission-attempt-1` within the client order; a reported
`venue_status=FILLED` still produces only `SubmissionAcknowledged`, never a
`ConfirmedFill`. The immutable `JournaledSubmissionResult` independently
reconstructs the expected event from its report and timestamp, rejects a
different status or venue identity, requires the stable event ID and a positive
position version, and reports `RECORDED` only after those facts are returned by
the journal.

If append fails, if its commit acknowledgement is lost, or if its receipt is
malformed, `SubmissionObservationNotRecorded` retains the exact report, event,
stable event ID, and original cause. The service never retries the external
call. A later exact append can discover an already committed event, while a
second service invocation discovers the existing reservation and still does not
resubmit. `requires_reconciliation` on a recorded result retains the narrower
`SubmissionReport` meaning: it is true for an ambiguous submission; an
acknowledgement is known submission state but remains distinct from any future
fill observation.

The PostgreSQL composition test verifies that another connection can see the
committed reservation before the executor is called. Two simultaneous exact
service invocations produce one `RECORDED`, one `EXISTING_RESERVATION`, and
exactly one adapter call. That test exposed a real M9b.3 insertion race: the
stream insert previously named only its primary-key conflict target while the
equivalent scoped-identity unique constraint could win concurrently. The insert
now uses targetless `ON CONFLICT DO NOTHING`; locked replay still verifies the
winning scope and instruction before any permission to submit.

This remains an unwired safety slice. The application facade re-exports the
service, so another application submodule import may load it, but an
import-aware gate finds no reference or invocation from an adapter, composition
root, startup path, or other runtime module. M8a/M8b consumer gates allow only
this exact application module and the two exact persistence modules. The gates
cover direct, facade, relative, aliased, assignment, and literal dynamic import
forms across all production roots.

Activation remains blocked until the runtime can construct a truthful
pre-submit `PositionInstruction`, query/reconcile the propagated client order
identity from an atomic durable paper receipt, and expose an independent
confirmed-fill observation. It also needs collective position-capacity
reservation, scope-wide unresolved-order inventory, durable quarantine, one
database configuration/factory, migrations and reconciliation before
readiness, bounded replay, and retirement of the executor, inline exit, and
Balanced Starter position authorities. The current paper executor's `FILLED`
response is not promoted to a confirmed fill.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q \
  tests/test_journaled_order_service.py \
  tests/test_order_position_journal.py \
  tests/test_journal_codec.py \
  tests/test_order_service.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py \
  tests/test_domain_contracts.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_journaled_order_service.py \
  tests/test_order_position_journal.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres \
  tests/postgres/test_journaled_order_service_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres tests/postgres
.venv/bin/black --target-version py310 --check \
  trading/application/journaled_order_service.py \
  trading/application/__init__.py \
  trading/persistence/order_position_journal.py \
  tests/test_journaled_order_service.py \
  tests/postgres/test_journaled_order_service_postgres.py \
  tests/test_order_position_journal.py \
  tests/test_order_service.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py
.venv/bin/isort --check-only <same-files>
.venv/bin/flake8 <same-files> --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/journaled_order_service.py \
  trading/persistence/order_position_journal.py \
  tests/test_journaled_order_service.py \
  tests/postgres/test_journaled_order_service_postgres.py
env -u ELVIS_TEST_POSTGRES_ADMIN_DSN \
  -u ELVIS_TEST_POSTGRES_REQUIRED \
  TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The focused application, repository, codec, lifecycle, position, and domain
suite passes 539 tests. The application/repository subset passes 95 tests on
Python 3.10. The new PostgreSQL 15 composition matrix passes 4 tests, and the
complete isolated PostgreSQL suite passes 35. The isolated non-PostgreSQL suite
passes 1,539 tests, skips 49, and deselects 38 under `Pacific/Honolulu`. Black,
isort, flake8, Python 3.10 compilation, and the diff check pass.

### M9b.5 correlated legacy paper response

M9b.5 closes one concrete transport gap in the active legacy paper path without
activating the journal. `LegacyPaperExecutionAdapter` passes the
`OrderIntent.client_order_id` as an explicit keyword-only argument through
`execute_buy` or `execute_sell`. `BinanceExecutor` validates a supplied client
ID before fee calculation, position reads, or database effects, forwards it to
the paper execution owner, and echoes it as `clientOrderId` in the response.
The adapter treats a nominal `FILLED` response as `SUBMITTED` only when this
echo exactly matches the intent; a missing or different echo is
`AMBIGUOUS/UNSAFE`, never acknowledged by symbol and side alone.

Mock venue order IDs no longer use `int(time.time())`, which could collide for
two same-symbol calls in one second. Each accepted paper invocation emits a
bounded opaque UUID-based `orderId`. Direct legacy callers that provide no
client ID remain compatible and receive no correlated echo; only the typed
adapter requires the echo. The change neither retries nor deduplicates an
order, and it does not make a paper response a `ConfirmedFill`.

This slice deliberately adds no SQL, lookup cache, or `get_order_status`
implementation. The legacy trade and position helpers still use separate
connections, commit independently, and can swallow write failures; neither
table stores the client or venue order ID. Therefore the response is transport
correlation only, not a durable receipt, idempotency guarantee, restart proof,
or reconciliation source. A forward migration and one transaction-owning paper
repository must persist the instruction identity, execution outcome, and fill
observation before M9b.4 can be composed into runtime.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q \
  tests/test_legacy_paper_adapter.py \
  tests/test_binance_executor.py \
  tests/test_paper_fill_integrity.py \
  tests/test_main_order_submission.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_legacy_paper_adapter.py \
  tests/test_binance_executor.py
.venv/bin/black --target-version py310 --check \
  trading/execution/legacy_paper_adapter.py \
  trading/execution/binance_executor.py \
  tests/test_legacy_paper_adapter.py \
  tests/test_binance_executor.py
.venv/bin/isort --check-only \
  trading/execution/legacy_paper_adapter.py \
  trading/execution/binance_executor.py \
  tests/test_legacy_paper_adapter.py \
  tests/test_binance_executor.py
.venv/bin/flake8 \
  trading/execution/legacy_paper_adapter.py \
  tests/test_legacy_paper_adapter.py \
  tests/test_binance_executor.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/execution/legacy_paper_adapter.py \
  trading/execution/binance_executor.py \
  tests/test_legacy_paper_adapter.py \
  tests/test_binance_executor.py
env -u ELVIS_TEST_POSTGRES_ADMIN_DSN \
  -u ELVIS_TEST_POSTGRES_REQUIRED \
  TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The focused suite passes 60 tests and skips 4; the adapter/executor subset
passes 47 tests and skips 3 on Python 3.10. Black and isort pass for all four
changed Python files. Flake8 passes for the new adapter and tests; the complete
legacy `binance_executor.py` still has its documented pre-existing lint debt,
so this slice does not claim a clean whole-file flake8 result for it. Python
3.10 compilation and the diff check pass. The isolated non-PostgreSQL suite
passes 1,552 tests, skips 50, and deselects 38 under `Pacific/Honolulu`.

### M9b.6 unresolved-submission read model

M9b.6 reuses the existing M9b.1 tables and adds no migration. The unwired
PostgreSQL repository exposes two read-only operations:
`replay_order(execution_scope, client_order_id)` and
`list_unresolved_submissions(execution_scope)`. The first resolves the order's
position key, replays the complete stream, and returns only the exact correlated
order. The second finds all position streams in one execution scope, replays
each of them inside the same `REPEATABLE READ READ ONLY` transaction, retains
only orders in `PENDING` or `RECONCILING`, and returns them in canonical client
order ID order.

The inventory is all-or-nothing. A missing order raises a typed not-found
error; a scope mismatch, corrupt envelope, version gap, invalid reducer
transition, or inconsistent position stream fails closed without returning a
partial list. Both reads roll back their read-only snapshot and close their
owned connection; they never call a write operation or assemble results from
multiple snapshots.

This does not make a `PENDING` order safe to resubmit. Reservations created by
M9b.4 or found during an in-place migration may represent either no external
call or an external effect whose observation was lost. Only a future
transaction-owning paper executor, operating after legacy writers are fenced,
can define `PENDING` as no committed paper effect and resolve a commit-unknown
outcome by exact event replay. M9b.6 supplies recovery visibility, not a retry
policy, durable quarantine, startup readiness, or runtime wiring.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q \
  tests/test_order_position_journal.py \
  tests/test_journaled_order_service.py \
  tests/test_journal_codec.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py
/usr/local/bin/python3.10 -m pytest -q tests/test_order_position_journal.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres \
  tests/postgres/test_order_position_repository_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres tests/postgres
.venv/bin/black --target-version py310 --check \
  trading/persistence/order_position_journal.py \
  tests/test_order_position_journal.py \
  tests/postgres/test_order_position_repository_postgres.py
.venv/bin/isort --check-only \
  trading/persistence/order_position_journal.py \
  tests/test_order_position_journal.py \
  tests/postgres/test_order_position_repository_postgres.py
.venv/bin/flake8 \
  trading/persistence/order_position_journal.py \
  tests/test_order_position_journal.py \
  tests/postgres/test_order_position_repository_postgres.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/persistence/order_position_journal.py \
  tests/test_order_position_journal.py \
  tests/postgres/test_order_position_repository_postgres.py
git diff --check
```

The focused repository/domain suite passes 443 tests, and the repository unit
suite passes 45 tests under Python 3.10. PostgreSQL 15 passes all 15 repository
integration tests and all 40 tests in the isolated PostgreSQL suite. Black,
isort, flake8, Python 3.10 compilation, and the diff check pass. The complete
non-PostgreSQL suite passes 1,563 tests, skips 50, deselects 43, and passes 7
subtests under `Pacific/Honolulu`.

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
