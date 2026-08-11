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
| M7 | Introduce pre-trade risk planning; move cooldown, sizing, leverage ceiling, and fee viability out of `main.py` | risk table tests, property tests, paper replay; no fallback order | feature flag selects legacy planner | In progress (M7a contract) |
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
