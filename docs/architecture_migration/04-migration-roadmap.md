# ELVIS V2 migration roadmap

> **Authoritative programme status.** V2 is being built incrementally on
> `codex/elvis-architecture-migration`; it is not a released or deployed runtime.
> Implemented-but-dormant components do not hold production authority. The
> compatibility paper runtime remains authoritative and `ACTIVE` remains a
> **NO-GO** until the cut-over gates in this ledger are complete.

## Migration contract

Every slice must be independently reviewable and reversible:

1. state the invariant and blast radius;
2. add or adapt focused tests first where practical;
3. make the smallest production change;
4. run focused tests, formatting, and static checks;
5. run the broader non-performance suite when shared runtime code changes;
6. update this status ledger and relevant operator documentation; and
7. create one explicit commit containing only that slice.

No implementation slice enables unattended live trading, changes secrets, or
deploys. Git publication is a separate, explicit repository action. A
load-bearing legacy path is deleted only after its replacement passes parity
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
| M8 | Make one `PositionService` own fills, stops, take profit, and reconciliation; retire background/inline duplicate ownership | state-machine tests, restart/reconciliation integration test | select legacy position manager | In progress (M8b position reducer; M9b.8 FIFO economics; M9b.9 quote settlement; M9b.11 pure paper accounting; no runtime cut-over) |
| M9 | Replace positional PostgreSQL tuples with repositories and migrations | ephemeral PostgreSQL from empty volume, upgrade test, transaction/idempotency tests | compatibility repository adapter | In progress (M9b.14c3c3b read-only imported-vs-operator-hypothesis review implemented locally; source/runtime provenance, coherent review, V2 opening/replay, pull-request CI, and runtime cut-over remain pending) |
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

### M9b.7 pure durable-submission owner contract

M9b.7 adds only `trading.application.durable_submission`, a pure contract for
the future transaction-owning paper executor. It does not add SQL, a repository
implementation, venue I/O, or runtime wiring. One immutable attempt context
binds the complete `PositionInstruction`, execution scope, timezone-aware
observation time, and durable submission event identity before any side effect.
A future owner must reuse that exact context for the initial operation,
commit-unknown recovery, and replay instead of sampling a new time or inventing
a new event identity.

The contract distinguishes `COMMITTED`, meaning the current call durably
established the returned facts, from `REPLAYED`, meaning an exact prior commit
was rediscovered without another execution effect. Both dispositions carry the
same canonical durable event/report meaning. The lifecycle event is the durable
source of truth, so canonical report reconstruction intentionally loses the raw
transport `venue_status`; callers must not present that non-persisted field as
a replayed fact. In particular, a canonical acknowledgement remains only an
ACK. It does not create a `ConfirmedFill`, change filled quantity, or project a
position.

The future concrete owner is expected to validate the stable attempt context,
serialize on the relevant stream, apply the paper execution effect, and append
the canonical submission and independently confirmed fill facts in one database
transaction. Only after commit may it return `COMMITTED`; an exact existing
durable attempt returns `REPLAYED`, while mismatched or indeterminate state
remains reconciliation work. This slice does not yet implement that transaction
or its repository.

Activation remains blocked on the economic side of the operation: balances,
fees, realised PnL, legacy trade rows, and open-position projections are not
part of this pure contract. M9b.7 also does not make the future owner the sole
writer and does not fence the existing executor, inline exit path, Balanced
Starter, or other legacy database writers. Those ownership and parity gates
must be implemented and verified before runtime composition.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py \
  tests/test_domain_contracts.py \
  tests/test_journaled_order_service.py
/usr/local/bin/python3.10 -m pytest -q tests/test_durable_submission.py
.venv/bin/black --target-version py310 --check \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py
.venv/bin/isort --check-only \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py
.venv/bin/flake8 \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The focused application/domain suite passes 497 tests, and the pure contract
passes 109 tests under Python 3.10. Black, isort, flake8, Python 3.10
compilation, and the diff check pass. The complete non-PostgreSQL suite passes
1,672 tests, skips 50, deselects 43, and passes 7 subtests under
`Pacific/Honolulu`.

### M9b.8 pure FIFO paper economics

M9b.8 adds `trading.domain.paper_economics`, extends the shared exact-Decimal
helper with exact multiplication, re-exports the new values through the domain
facade, and adds pure unit and no-runtime-consumer gates. It does not add SQL, a
repository, a schema migration, a clock, environment reads, or runtime wiring.
A `PaperFillRecord` couples one validated `PositionFill` to its durable
`position_version` and per-order `(client_order_id, event_id)` identity;
`PaperEconomics` folds those records into immutable FIFO cost lots, exact open
quantity and cost, cumulative gross realised PnL, and exact fee totals grouped
by asset.

`position_version` is the only causal ordering input. Fill-record versions must
increase strictly, but need not be consecutive: submission and other non-fill
lifecycle events may occupy the intervening versions. The full journal replay,
not this fill-only reducer, must prove that the complete position stream has no
gap. An exact replay of the same record returns the existing projection. A
conflicting composite event identity, fill identity, or position version, or a
new fill whose version does not advance, fails closed; the same bare `event_id`
may recur under a different client order. Identity ordering inside
`Position.fills`, event timestamps, legacy row IDs, and database collation are
not substitutes for causal version order.

`PaperLotMethod` admits only `FIFO`. Every `OPEN` confirmed fill creates one lot
using its exact price and quantity; a scale-in appends a later lot.
`REDUCE_ONLY` consumes the oldest remaining lots first, leaves an exact
partial-lot remainder when necessary, and closes the projection only when no
quantity remains. Open cost is the exact sum of each
surviving lot's `remaining_quantity * entry_price`. Gross realised PnL is
`(exit_price - entry_price) * matched_quantity` for a long and
`(entry_price - exit_price) * matched_quantity` for a short. Leverage does not
multiply notional, fee, or gross PnL for an already fixed contract/base
quantity. The reducer uses confirmed fill price, quantity, and fee facts only;
it does not use intent reference prices, hardcoded mock prices, epsilon
tolerances, or binary floats.

FIFO is a deliberate target policy, not a claim of behavioural parity with the
legacy paper path. The current executor and database helpers mix incompatible
models: opens insert independent rows, Balanced Starter and the inline exit path
close one row by ID, while executor netting selects an unordered first opposite
row and then deletes every row for that symbol and side. Its helpers also use
separate transactions and `REAL` columns. That history cannot define stable lot
allocation, exact cost basis, or replay semantics for the new reducer.

Positive confirmed fees remain separate exact totals for each `fee_asset`, and
zero fees are omitted; fees are not converted or subtracted from gross PnL.
Balance and cash accounting, margin reservation, net PnL, fee-asset conversion,
funding, borrowing, liquidation, unrealised mark-to-market PnL, price/fill
simulation, tick and lot quantisation, exit selection, legacy-table projection,
and historical legacy adoption are explicitly deferred. M9b.8 also does not
make the future owner the sole writer or fence `BinanceExecutor`, Balanced
Starter, the inline main exits, or any other legacy database writer.

Activation therefore still requires an exact durable fill ledger, atomic
economic and compatibility projections where those projections remain needed,
one PostgreSQL transaction owner, deterministic stream locking and replay,
commit-unknown recovery, startup readiness that proves legacy writers are
fenced, and reconciliation/quarantine of unsupported legacy state. The pure
reducer establishes arithmetic and allocation semantics only.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py \
  tests/test_order_lifecycle.py \
  tests/test_domain_contracts.py \
  tests/test_durable_submission.py \
  tests/test_order_position_journal.py
/usr/local/bin/python3.10 -m pytest -q tests/test_paper_economics.py
.venv/bin/black --target-version py310 --check \
  trading/domain/_decimal.py \
  trading/domain/paper_economics.py \
  trading/domain/__init__.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py
.venv/bin/isort --check-only \
  trading/domain/_decimal.py \
  trading/domain/paper_economics.py \
  trading/domain/__init__.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py
.venv/bin/flake8 \
  trading/domain/_decimal.py \
  trading/domain/paper_economics.py \
  trading/domain/__init__.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/domain/_decimal.py \
  trading/domain/paper_economics.py \
  trading/domain/__init__.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The pure paper-economics suite passes 75 tests under both supported Python
interpreters, and the focused domain/application/persistence regression suite
passes 556 tests. Black, isort, flake8, Python 3.10 compilation, and the diff
check pass. The complete non-PostgreSQL suite passes 1,747 tests, skips 50,
deselects 43, and passes 7 subtests under `Pacific/Honolulu`.

### M9b.9 pure linear quote settlement

M9b.9 adds `trading.domain.paper_settlement` and re-exports its public contract
through the domain facade. `PaperLinearInstrument` makes the instrument model
explicit: one symbol, distinct base and quote assets, a multiplier of one, and
linear settlement in the quote asset. This deliberately excludes inverse,
quanto, multiplier-bearing, and implicitly parsed symbol contracts.

`settle_paper_fill(instrument, before, record)` consumes one already confirmed,
causally versioned `PaperFillRecord` and an optional prior
factory-created `PaperSettlementCheckpoint`. That compact checkpoint binds the
instrument to the prior FIFO economics without retaining a recursive chain of
settlement results, so a later fill cannot silently renominate the same
position stream. It neither executes an order nor creates an ACK or fill. The
returned immutable `PaperSettlement` contains the exact M9b.8 projection at
`after.economics` and three explicitly denominated delta views:

- `gross_realized_pnl_delta` is the change in cumulative FIFO gross realised
  PnL and is always a `PaperAssetAmount` in the instrument's quote asset;
- `fee_debits` contains the positive confirmed fee amount in the fill's exact
  `fee_asset`; and
- `cash_deltas` combines those terms by asset as signed amounts: realised PnL
  in quote and a negative fee in its own asset, with exact zero totals omitted.

No fee-asset conversion occurs. A fee in the quote asset may combine
algebraically with quote-settled realised PnL; a fee in another asset remains a
separate debit. Neither case authorises a synthetic cross-asset net PnL, an FX
rate of one, or a balance mutation. All arithmetic and direct-construction
validation retain exact `Decimal` payload identity rather than using float,
tolerance, or numeric equality that discards quantum.

`PaperSettlementDisposition.APPLIED` means a new causal fill was applied. It is
still `APPLIED` when an opening fill realises zero PnL and has no fee.
Reapplying the exact record to its existing checkpoint yields `REPLAYED`,
retains the same checkpoint and `PaperEconomics` object, emits a zero
quote-denominated gross-PnL delta, and emits no repeated fee debit or cash
delta. Symbol mismatch, causal conflict, or unrepresentable exact arithmetic
fails closed as `InvalidPaperSettlement`.
Direct `PaperSettlement` construction re-derives the after projection,
disposition, and every delta, preventing a caller from forging the result.
The M8/M9 causal values reachable from its checkpoint reject generated
`__setstate__` mutation after construction. Standard copy and pickle
restoration target a fresh object, revalidate every field, and clean up a
failed partial restore.

This is a pure semantic slice only. It does not maintain cash or asset balances,
reserve or release margin, decide account admission, enforce buying power,
model funding, borrowing, liquidation, or unrealised mark-to-market PnL, or
select/observe fills. It adds no SQL, migration, repository, PostgreSQL lock,
transaction owner, legacy `trades`/`open_positions` projection, runtime wiring,
sole-writer fence, readiness policy, or reconciliation workflow. M9b.11 now
defines the pure account fold over these deltas, but they still cannot become
durable postings or authorise execution until a later transactional owner
persists and serializes those invariants.

Verification at implementation time:

```bash
.venv/bin/python -m pytest -q \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py \
  tests/test_order_lifecycle.py \
  tests/test_domain_contracts.py
/usr/local/bin/python3.10 -m pytest -q tests/test_paper_settlement.py
.venv/bin/black --target-version py310 --check \
  trading/domain/paper_settlement.py \
  trading/domain/paper_economics.py \
  trading/domain/positions.py \
  trading/domain/order_lifecycle.py \
  trading/domain/orders.py \
  trading/domain/_validation.py \
  trading/domain/__init__.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py \
  tests/test_order_lifecycle.py
.venv/bin/isort --check-only \
  trading/domain/paper_settlement.py \
  trading/domain/paper_economics.py \
  trading/domain/positions.py \
  trading/domain/order_lifecycle.py \
  trading/domain/orders.py \
  trading/domain/_validation.py \
  trading/domain/__init__.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py \
  tests/test_order_lifecycle.py
.venv/bin/flake8 \
  trading/domain/paper_settlement.py \
  trading/domain/paper_economics.py \
  trading/domain/positions.py \
  trading/domain/order_lifecycle.py \
  trading/domain/orders.py \
  trading/domain/_validation.py \
  trading/domain/__init__.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py \
  tests/test_order_lifecycle.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/domain \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py \
  tests/test_order_lifecycle.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The focused M9b.9/domain regression suite passes 452 tests. The settlement
contract itself passes 50 tests under both supported Python interpreters.
Black, isort, flake8, Python 3.10 compilation, and the diff check pass. The
complete non-PostgreSQL suite passes 1,797 tests, skips 50, deselects 43, and
passes 7 subtests under `Pacific/Honolulu`.

### M9b.10a pure terminal paper-submission plan

M9b.10a extends only the pure `trading.application.durable_submission`
boundary and its package exports. `PaperPlannedFill(event_id, fill)` couples one
non-durable `ConfirmedFill` candidate to the event identity it would receive in
the journal. `PaperSubmissionPlan(attempt, submission, fills)` contains the
exact `SubmissionAttemptContext`, exactly one `SubmissionAcknowledged`, and a
non-empty tuple containing only exact `PaperPlannedFill` values. It is a set of
candidate facts, not a receipt, execution result, or persistence claim.

The plan validates the complete terminal full-fill batch before the M9b.10b
owner may write it. The acknowledgement preserves the attempt's client order
ID and exact observation time. Its event ID and every fill event ID are
distinct. Each fill preserves the intent's client order ID, symbol, and side,
does not predate the acknowledgement, and shares one venue order ID with the
acknowledgement.
Trade IDs are unique. Exact-`Decimal` fill quantities must sum to exactly the
intent quantity, so an empty, partial, or over-filled candidate batch fails
closed. An ambiguous or failed submission cannot carry planned fills.

`PaperSubmissionPlanner.plan(attempt, /) -> PaperSubmissionPlan` is the narrow
M9b.10b owner dependency. It supplies already stable, precomputed candidate
facts and promises to retain the exact attempt object. Its data must not depend
on a hidden clock, random draw, network/database read, mutable market snapshot,
or a price inferred from `OrderIntent.reference_price`. The protocol cannot by
itself prove those properties; the later owner must call it only after proving
under the stream lock that the order is genuinely new, require
`plan.attempt is attempt`, and cover the concrete planner/composition with
determinism tests. Replay and reconciliation paths must never invoke it.

M9b.10a itself deliberately implements no SQL, repository method, migration,
transaction, paper simulator, venue/market I/O, clock, runtime consumer, or
composition-root wiring. It does not alter or consume migration
`0002_order_position_journal.sql`. M9b.10b now implements the first narrow
PostgreSQL owner with one fresh connection and transaction that locks and
replays the stream, reserves a genuinely new instruction, appends the ACK and
all terminal full fills at consecutive versions, and commits once. It does not
compose the existing public `reserve_instruction` and `append_event` methods,
because each owns a separate transaction. An exact already-terminal batch may
return `REPLAYED` without planning; an existing `PENDING`, ACK-only, partial,
interleaved, mismatched, gapped, or corrupt history requires reconciliation or
fails closed with no planner call, guessed suffix, append, or resubmission. A
lost commit acknowledgement preserves `SubmissionCommitUnknown`, and a fresh
retry replays before any possible plan.

Migration `0002` is sufficient only for that journal-only ACK/full-fill batch.
It has no batch manifest and proves no durable balance, posting, margin,
instrument-snapshot, or legacy-projection invariant. M9b.10b does not change
that migration. Activation remains blocked on all of the following:

- durable account-version, balance, reservation, and posting ownership that
  applies the M9b.11 margin/admission policy atomically with the journal batch;
- persisted instrument identity/version and explicit price, fee, tick, and
  lot-rule snapshot provenance;
- funding, borrowing, liquidation, and unrealised mark-to-market rules;
- generation and execution-scope provenance;
- durable reconciliation and quarantine, including pre-atomic `PENDING`,
  ACK-only, and partial histories;
- startup migration/readiness verification and an explicit repository factory;
- bounded replay or snapshots;
- a proved sole-writer fence over the legacy executor, inline exits, Balanced
  Starter, and every other legacy database writer;
- side-effect-free shadow parity followed by an explicit cut-over decision.

Verification commands for this slice:

```bash
.venv/bin/python -m pytest -q \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py \
  tests/test_domain_contracts.py \
  tests/test_journaled_order_service.py
/usr/local/bin/python3.10 -m pytest -q tests/test_durable_submission.py
.venv/bin/black --target-version py310 --check \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/domain/order_lifecycle.py \
  trading/domain/orders.py \
  tests/test_durable_submission.py
.venv/bin/isort --check-only \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/domain/order_lifecycle.py \
  trading/domain/orders.py \
  tests/test_durable_submission.py
.venv/bin/flake8 \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/domain/order_lifecycle.py \
  trading/domain/orders.py \
  tests/test_durable_submission.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/domain/order_lifecycle.py \
  trading/domain/orders.py \
  tests/test_durable_submission.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The focused application/domain regression suite passes 523 tests. The durable
submission contract itself passes 135 tests under both supported Python
interpreters. Black, isort, flake8, Python 3.10 compilation, and the diff check
pass. The complete non-PostgreSQL suite passes 1,823 tests, skips 50, deselects
43, and passes 7 subtests under `Pacific/Honolulu`.

### M9b.10b unwired atomic terminal paper-submission owner

M9b.10b adds the concrete
`trading.persistence.atomic_paper_submission_owner.PostgresAtomicPaperSubmissionOwner`
without wiring it into startup or the trading runtime. Its exact entry point is
`execute(attempt: SubmissionAttemptContext, /) -> DurableSubmissionReceipt`.
The constructor receives an injected connection factory and the M9b.10a
`PaperSubmissionPlanner`.

The owner supports one deliberately narrow paper path: a genuinely new
instruction whose stable precomputed plan contains exactly one ACK followed by
one or more `ConfirmedFill` facts that fully fill the order immediately. Every
fill is persisted; the exact quantities must sum to the intent quantity. It
does not admit an ambiguous, failed, ACK-only, partial, delayed, cancelled, or
externally submitted order, and it never manufactures a missing suffix.

`execute()` obtains one fresh connection, establishes one write transaction,
inserts or locks the position stream, and replays the complete locked stream.
Before a new reservation can be created, every existing sibling order on that
stream must itself be an exact supported terminal ACK/full-fill batch. The
owner then inserts the instruction reservation, invokes the planner exactly
once under the same stream lock, requires the returned plan to retain the exact
attempt object, validates the lifecycle and position transition, records the
venue correlation, advances the stream version for the complete batch, inserts
the ACK and fills at consecutive versions, replays the resulting stream, and
commits once. It does not compose
`PostgresOrderPositionJournal.reserve_instruction()` with `append_event()`;
those public operations commit independently and cannot provide this invariant.

When the exact durable instruction and exact attempt already identify a
supported terminal batch, the owner reconstructs a canonical `REPLAYED`
receipt directly from journal order, without calling the planner or changing
row contents or metadata. This remains true when a later terminal sibling batch
exists on the position stream. An instruction mismatch, including a changed
`Decimal` quantum, is a journal identity conflict. A `PENDING`, ACK-only,
partial, interleaved, contradictory, unresolved-sibling, gapped, or corrupt
history fails closed before planning as reconciliation work or a typed journal
error. It is never permission to append, infer a fill, or resubmit.

This replay recognition is deliberately shape-based. Migration `0002` has no
atomic-owner generation or batch-provenance marker, so a complete ACK and exact
full-fill suffix previously written by separate journal commits is also
adopted as `REPLAYED`; the owner does not falsely claim to prove its origin.
That adoption is only a journal no-op. It cannot authorize runtime activation,
legacy compatibility state, or economic/accounting effects. A later durable
generation and execution-scope fence must distinguish eligible atomic-owner
history before cut-over.

The receipt is returned as `COMMITTED` only after PostgreSQL acknowledges the
single commit. If that acknowledgement is lost, the owner raises
`SubmissionCommitUnknown(attempt)` without claiming rollback or success. A
fresh call locks and replays before any possible planner invocation, so a
transaction that actually committed is recovered as `REPLAYED` and is not
duplicated. Failures before commit roll back the new stream, reservation,
venue correlation, stream-version advance, and every event together.

This slice reuses `0002_order_position_journal.sql` unchanged; there is no new
migration. Its SQL is limited to transaction state and the M9b.1
`np.position_streams`, `np.orders`, and `np.order_events` journal. It does not
call a venue, discover or infer a price, sample a clock, run FIFO economics or
settlement, write balances/postings/margin, update legacy trades or open
positions, publish telemetry, construct a production repository, or activate a
runtime consumer. The module is protected by a zero-consumer gate. Rolling it
back therefore removes only the unused owner and its tests; the additive
unchanged journal schema remains available to the earlier repository.

PostgreSQL 15 tests cover exact two-fill quantity and version preservation, one
connection and one commit, read-only exact replay, same-attempt and distinct-
batch concurrency, non-interleaving stream blocks, failure injection after
each SQL mutation, planner failure, lost commit acknowledgement, incomplete or
contradictory histories, shape-based adoption of an exact terminal history,
corrupt/gapped streams, exact `Decimal` identity, and a
statement trace proving zero legacy/account-table access.

Verification commands for this slice:

```bash
.venv/bin/python -m pytest -q \
  tests/test_durable_submission.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/test_order_position_journal.py \
  tests/test_journal_codec.py \
  tests/test_order_lifecycle.py \
  tests/test_position_lifecycle.py \
  tests/test_domain_contracts.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_durable_submission.py \
  tests/test_atomic_paper_submission_owner.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra \
  tests/postgres/test_atomic_paper_submission_owner_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  /usr/local/bin/python3.10 -m pytest -q -ra \
  tests/postgres/test_atomic_paper_submission_owner_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q -ra -m postgres tests/postgres
.venv/bin/black --target-version py310 --check \
  trading/application/__init__.py \
  trading/application/durable_submission.py \
  trading/domain/order_lifecycle.py \
  trading/persistence/atomic_paper_submission_owner.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/postgres/test_atomic_paper_submission_owner_postgres.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_position_lifecycle.py
.venv/bin/isort --check-only \
  trading/application/__init__.py \
  trading/application/durable_submission.py \
  trading/domain/order_lifecycle.py \
  trading/persistence/atomic_paper_submission_owner.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/postgres/test_atomic_paper_submission_owner_postgres.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_position_lifecycle.py
.venv/bin/flake8 \
  trading/application/__init__.py \
  trading/application/durable_submission.py \
  trading/domain/order_lifecycle.py \
  trading/persistence/atomic_paper_submission_owner.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/postgres/test_atomic_paper_submission_owner_postgres.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_position_lifecycle.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/__init__.py \
  trading/application/durable_submission.py \
  trading/domain/order_lifecycle.py \
  trading/persistence/atomic_paper_submission_owner.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/postgres/test_atomic_paper_submission_owner_postgres.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_position_lifecycle.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The focused application, owner, repository, codec, lifecycle, position, and
domain suite passes 626 tests. The application/owner subset passes 159 tests on
Python 3.10. The isolated atomic-owner PostgreSQL 15 matrix passes 23 tests
under each supported Python interpreter, and the complete isolated PostgreSQL
15 suite passes 63 tests in the project environment. Black, isort, flake8,
Python 3.10 compilation, and the diff check pass. The complete non-PostgreSQL
suite passes 1,847 tests, skips 50, deselects 66, and passes 7 subtests under
`Pacific/Honolulu`.

### M9b.11 pure global paper-account accounting

M9b.11 adds `trading.domain.paper_accounting` and its domain-facade exports.
The slice is an immutable, pure fold from exact M9b.9 `PaperSettlement` facts
to account admission, postings, balances, margin reservations, and solvency. It
does not extend migration `0002`, add another migration, or wire a production
consumer.

`PaperAccountPolicy(account_key, collateral_asset, margin_quantum)` fixes the
identity, denomination, and exact positive reservation quantum for one isolated
paper account. `PaperAccountBalance(asset, available, reserved)` separates
available funds from non-negative reserved margin. `PaperMarginReservation`
binds the current positive collateral requirement to one position key.
`new_paper_account(policy, opening_balances)` requires one exact, unique,
asset-sorted tuple of solvent, unreserved opening balances containing the
collateral asset. The resulting `PaperAccount` starts `ACTIVE` with no synthetic
settlement record or reservation.

`PaperAccountSettlementRecord(account_version, settlement)` couples each newly
applied settlement to a positive account-global version. Applied versions form
the exact contiguous prefix `1..len(records)` across every position sharing the
account. This `account_version` is independent of each fill's durable
`position_version`: the latter orders a single position stream, whereas the
former serializes collateral consumption globally. Rejected candidates do not
consume the next account version.

`admit_paper_settlement(account, account_version, settlement)` returns a
`PaperAccountAdmission` with three exact dispositions:
`PaperAccountAdmissionDisposition.APPLIED`, `REPLAYED`, or `REJECTED`. An exact
settlement found at its existing account version replays against the current
account with no posting or duplicate effect, including after later records.
Conflicting data at an existing version, a reused event/fill identity at
another version, a sequence gap/regression, or a broken per-position settlement
chain fails closed as `InvalidPaperAccountTransition`. A funding/admission
rejection returns the same account object, no postings, explicit `reasons`, and
leaves that version available for another candidate.

For each admitted settlement, the target reservation is computed from the
complete after-settlement FIFO projection:

```text
target_margin = ceil(open_cost / leverage / margin_quantum) * margin_quantum
margin_delta  = target_margin - current_position_reservation
```

The ceiling is implemented as an exact bounded integer ratio and therefore
does not inherit ambient `Decimal` precision, rounding mode, or traps. Every
non-zero requirement rounds upward to the explicit policy quantum. Scale-in and
reduction recompute the target from surviving FIFO lots, avoiding accumulated
rounding drift; an exact close releases the full reservation and removes its
`PaperMarginReservation`.

`PaperAccountPosting(asset, bucket, amount)` records only non-zero exact signed
movements. Settlement `cash_deltas` post to `AVAILABLE` in their own asset.
Margin movement creates equal and opposite collateral postings between
`PaperAccountPostingBucket.AVAILABLE` and `RESERVED_MARGIN`. For every asset,
the total posting amount therefore equals the corresponding settlement cash
delta; internal reservation movement conserves the asset total. A quote fee and
margin reservation are considered together. For `OPEN`, a foreign-asset fee
must be funded; every such fee is debited in its explicit asset without
conversion. Reserved balances cannot become negative and zero movements are
omitted.

An `OPEN` settlement is applied only when the prior account is `ACTIVE` and all
resulting available asset balances stay non-negative. Otherwise it returns
`REJECTED` without mutation. A `REDUCE_ONLY` settlement may still apply so a
position can close and its exact realised loss or fees can be recognized. If
that produces any negative available balance, the derived state becomes
`PaperAccountState.INSOLVENT`; subsequent `OPEN` exposure is rejected. Direct
`PaperAccount` and `PaperAccountAdmission` construction replays and re-derives
all balances, reservations, records, postings, disposition, state, and reasons,
so a caller cannot forge an applied outcome. The policy retains the supplied
exact `Decimal` quantum rather than silently rewriting it.

This is not yet the durable account owner required by M9b.10b activation. The
module performs no I/O and imports only standard-library and domain code. It
contains no SQL, schema, repository, lock, transaction, price source, clock,
venue call, legacy-table projection, telemetry, or runtime composition. A
future PostgreSQL slice must make account-version allocation, journal facts,
settlement records, postings, balances, and reservations one atomic transaction
under a single account writer. Until then, two concurrent pure evaluations can
both see the same available collateral and neither may authorize execution.

Activation also remains blocked on persisted instrument/version and price,
fee, tick, lot, and opening-capital provenance; funding, borrowing,
liquidation, and unrealised mark-to-market policy; generation/scope fences;
durable reconciliation and quarantine; startup migration/readiness and a
repository factory; bounded replay or snapshots; atomic compatibility
projections if retained; a proved legacy sole-writer fence; shadow parity; and
an explicit cut-over decision.

Verification commands for this slice:

```bash
.venv/bin/python -m pytest -q tests/test_paper_accounting.py
.venv/bin/python -m pytest -q \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py \
  tests/test_order_lifecycle.py \
  tests/test_domain_contracts.py
/usr/local/bin/python3.10 -m pytest -q tests/test_paper_accounting.py
.venv/bin/black --target-version py310 --check \
  trading/domain/paper_accounting.py \
  trading/domain/__init__.py \
  tests/test_paper_accounting.py
.venv/bin/isort --check-only \
  trading/domain/paper_accounting.py \
  trading/domain/__init__.py \
  tests/test_paper_accounting.py
.venv/bin/flake8 \
  trading/domain/paper_accounting.py \
  trading/domain/__init__.py \
  tests/test_paper_accounting.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/domain/paper_accounting.py \
  trading/domain/__init__.py \
  tests/test_paper_accounting.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The dedicated paper-accounting suite passes 63 tests under both supported
Python interpreters. The exact focused accounting, settlement, economics,
position, order, and domain command above passes 515 tests. Black, isort,
flake8, Python 3.10 compilation, and the diff check pass. The complete M9b.11
non-PostgreSQL gate passes 1,910 tests, skips 50, deselects 66, and passes 7
subtests under `Pacific/Honolulu`.

### M9b.12a compact paper-account journal codec

M9b.12a adds the pure, unwired
`trading.persistence.paper_account_journal_codec` contract before migration
`0003` fixes the durable account schema. It is imported through its direct
module and is deliberately not added to the lightweight `trading.persistence`
migration facade.

The opening envelope is produced by
`encode_paper_account_opening(execution_scope, owner_generation, account)` and
decoded by `decode_paper_account_opening(...)`. Encoding requires the exact
empty M9b.11 account: current balances equal its explicit opening balances, and
there are no settlement records or margin reservations. The canonical payload
binds the execution scope and positive owner generation to
`PaperAccountPolicy` and every opening balance. The indexed columns repeat the
scope, account key, generation, and collateral asset. Decode verifies version
1, strict payload shape and SHA-256, reconstructs the empty `PaperAccount`, and
cross-checks all indexed columns. No capital is seeded, defaulted, backfilled,
or represented as a fake fill.

`encode_paper_account_settlement(admission)` accepts only a newly
`PaperAccountAdmissionDisposition.APPLIED` result. The compact version-1
payload contains:

- account key and collateral asset;
- account version plus position key/version, client order ID, event ID, and
  trade ID;
- versioned `LINEAR_QUOTE_MULTIPLIER_ONE` symbol/base/quote identity;
- exact realised-PnL, fee-debit, and per-asset cash deltas;
- the derived non-zero postings, resulting account state, and resulting margin
  for the affected position.

It intentionally omits the recursive `before`/`after` settlement chain, FIFO
lots, cumulative economics, prior account records, and full account projection.
`decode_paper_account_settlement(before, settlement, ...)` receives the trusted
domain inputs needed to replay causality, calls M9b.11 admission at the indexed
account version, requires a newly applied result, regenerates the payload, and
cross-checks every denormalized account, journal, and instrument column. A
standalone stored row cannot manufacture a settlement or establish the global
account prefix without repository replay.

The owner-provenance envelope is built from
`PaperAccountBatchManifest` and `PaperAccountBatchFill`. The manifest binds
execution scope, account, positive owner generation, position/order identity,
and instruction SHA-256 to the ACK's event ID, position version, canonical
observed time, and event SHA-256. Every fill reference binds the same
position/order to event and trade IDs, position and account versions, the
journal event SHA-256, and the corresponding account-settlement SHA-256. The
manifest requires a non-empty fill tuple, consecutive fill position versions
immediately after the ACK, consecutive account versions, and unique event and
trade identities. `encode_paper_account_batch(...)` additionally exposes the
first/last account versions, last position version, and fill count as indexed
range columns. `decode_paper_account_batch(...)` reconstructs the manifest and
cross-checks that range as well as every other indexed value.

Every envelope is a frozen, slotted, validated value using canonical sorted,
compact, ASCII JSON and lowercase SHA-256 over its UTF-8 bytes. Exact finite
`Decimal` values remain canonical strings, preserving quantum, and the batch
timestamp is normalized and stored as canonical UTC with six fractional
digits, both in the indexed `submission_observed_at` and the payload. Decoders
treat JSON text or PostgreSQL JSON objects as untrusted input: duplicate keys,
JSON constants, extra/missing keys, non-canonical scalars, unknown versions,
domain violations, hash changes, and payload/index disagreement fail closed
with `JournalQuarantineError`. The hash detects drift and binds identities; it
is not authentication or evidence that one database transaction produced the
facts.

This slice adds no SQL or migration and does not change migration `0002`. It
implements no repository, PostgreSQL lock, transaction owner, batch write,
account provisioning command, execution call, clock sampling, runtime wiring,
readiness gate, reconciliation/quarantine workflow, legacy compatibility write,
or sole-writer fence. It also adds no funding, borrowing, liquidation,
unrealised mark-to-market, or price/tick/lot policy. Shape-compatible terminal
history created before this codec still lacks owner/account provenance and
requires reconciliation. M9b.12b now supplies the next additive slice:
migration `0003` stores the envelopes and projections dormantly before a future
owner can lock the account first, lock the position second, and commit journal
plus account facts atomically. Its exact schema contract and deliberate
row-level completeness boundary are recorded below.

Verification commands for this slice:

```bash
.venv/bin/python -m pytest -q tests/test_paper_account_journal_codec.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_journal_codec.py
.venv/bin/python -m pytest -q \
  tests/test_paper_account_journal_codec.py \
  tests/test_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_position_lifecycle.py \
  tests/test_order_lifecycle.py \
  tests/test_domain_contracts.py
.venv/bin/black --target-version py310 --check \
  trading/persistence/paper_account_journal_codec.py \
  tests/test_paper_account_journal_codec.py
.venv/bin/isort --check-only \
  trading/persistence/paper_account_journal_codec.py \
  tests/test_paper_account_journal_codec.py
.venv/bin/flake8 \
  trading/persistence/paper_account_journal_codec.py \
  tests/test_paper_account_journal_codec.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/persistence/paper_account_journal_codec.py \
  tests/test_paper_account_journal_codec.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The dedicated codec suite passes 136 tests under each supported Python
interpreter. The exact focused codec, journal-codec, accounting, settlement,
economics, position, order, and domain command above passes 746 tests. Black,
isort, flake8, Python 3.10 compilation, and the diff check pass. The complete
non-PostgreSQL gate passes 2,045 tests, skips 50, deselects 66, and passes 7
subtests under `Pacific/Honolulu`. No PostgreSQL test is required for M9b.12a
because it adds no schema or I/O.

### M9b.12b dormant paper-account ledger schema

M9b.12b adds the checksummed, forward-only
`0003_paper_account_ledger.sql` migration. Its immutable SHA-256 is
`6d7b99ed9cfa3480a12c550736e6bc914320fd0785d07fd1e48a8e37b912e081`.
The migration is additive and CREATE-only: four unique indexes make exact
composite references into the unchanged version-2 order journal possible, and
six new relations remain empty after migration:

- `paper_account_streams` holds the provisioned opening envelope and the future
  account-global lock/version/state row;
- `paper_account_balances` and `paper_margin_reservations` hold current
  exact-Decimal-text projections;
- `paper_account_batch_manifests` holds the owner-batch envelope and its exact
  opening, instruction, and ACK references;
- `paper_account_settlements` holds each compact settlement and simultaneously
  serves as the relational manifest-fill row; and
- `paper_account_postings` holds the per-settlement projection entries.

There is intentionally no separate batch-fill relation. Each settlement copies
its manifest's first account version, ACK position version, fill count, and its
one-based ordinal. CHECK constraints derive the exact account and position
versions from that ordinal. A deferred composite foreign key binds the row to
the manifest range, while another binds its position/order/event/trade/type/hash
to the exact version-2 `CONFIRMED_FILL`. The manifest binds the exact immutable
opening scope/account/provisioned generation/version/hash, order instruction
hash, and `SUBMISSION_ACKNOWLEDGED` identity/time/hash. It also prevents one
client order from being claimed by two accounts. Instrument checks require
distinct base and quote assets and require the quote asset to be the account's
collateral. `owner_generation` is opening provenance, not runtime fencing.

These constraints prove row-level ownership, identity, denomination, ordinal,
and envelope-reference consistency at commit. They do not prove that the
number of settlement rows equals `fill_count`, that every declared ordinal or
JSON manifest member is present, or that stored JSON/hash and exact Decimal
text are canonical. A manifest with zero settlement rows is explicitly valid
in this dormant schema. Nor does SQL replay M9b.11 causality, prove global
inter-batch account contiguity, synchronize the stream tail, or validate
balance/reservation/posting conservation. The future account repository must
decode and recompute all M9b.12a envelopes, cross-check complete manifest
membership and projections, and quarantine any mismatch. Its atomic owner must
lock account first and position second before it can own journal plus account
facts.

The migration performs no seed, default-capital provisioning, backfill, DML,
`ALTER`, destructive statement, trigger, or legacy-table write. It adds no
repository, transaction owner, execution call, runtime wiring, readiness gate,
reconciliation workflow, rotating generation, or sole-writer fence. Existing
version-2 histories are not adopted or blessed. The legacy runtime remains
authoritative, so rollback of this slice is application-code rollback with the
unused additive relations left in place.

Verification commands for this slice:

```bash
.venv/bin/pytest -q tests/test_migration_runner.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/pytest -q \
  tests/postgres/test_paper_account_ledger_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  /usr/local/bin/python3.10 -m pytest -q \
  tests/postgres/test_paper_account_ledger_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/pytest -q \
  tests/postgres/test_migration_runner_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/pytest -q tests/postgres
.venv/bin/pytest -q \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_order_position_journal.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_order_position_journal.py
.venv/bin/black --target-version py310 --check \
  tests/test_migration_runner.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_ledger_postgres.py
.venv/bin/isort --check-only \
  tests/test_migration_runner.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_ledger_postgres.py
.venv/bin/flake8 \
  tests/test_migration_runner.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_ledger_postgres.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/persistence \
  tests/test_migration_runner.py \
  tests/postgres
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The migration unit suite passes 35 tests. The dedicated dormant-ledger
PostgreSQL 15 suite passes 17 tests under each supported Python interpreter;
the migration PostgreSQL suite passes 9 tests, and the complete PostgreSQL 15
suite passes 82 tests. The exact adjacent migration/codec/accounting suite
passes 404 tests under each interpreter. Black, isort, flake8, Python 3.10
compilation, and the diff check pass. The complete non-PostgreSQL gate is also
complete for this slice: 2,046 tests pass, 50 skip, 85 deselect, and 7 subtests
pass under `Pacific/Honolulu`. Pytest exits successfully; the legacy background
threads still emit their known post-success logging errors after the result.

### M9b.12c strict paper-account provision/replay repository

M9b.12c adds the direct, unwired
`trading.persistence.paper_account_journal.PostgresPaperAccountJournal`. It
owns exactly three operations:

- `provision_account(*, execution_scope, owner_generation, account)` creates
  one explicit empty-account opening or returns its exact durable retry;
- `replay_account(*, execution_scope, account_key)` reconstructs one account;
  and
- `list_accounts(*, execution_scope)` replays the complete scoped inventory,
  sorted by account key and all-or-nothing.

Provision returns frozen `ProvisionedPaperAccount` and
`ReplayedPaperAccount` values. `ProvisionDisposition.CREATED` distinguishes the
single opening insert from `EXISTING` exact retry. Invalid scope, generation,
or non-empty account inputs fail before connecting. The stream and every
opening balance are inserted and strictly replayed inside one `READ COMMITTED`
transaction and exposed only after one successful commit. Existing rows are
locked and must match scope, immutable provisioning generation, and the exact
opening envelope. Concurrent exact callers converge on the same account;
conflicts leave it unchanged. A lost commit acknowledgement raises
`PaperAccountCommitUnknown`, including the scope, account key, and generation,
rather than reporting false success or retrying a write internally.

The exception vocabulary separates invalid input, known pre-commit storage
failure, unknown commit outcome, not-found, replay quarantine, and immutable
opening conflicts. `PaperAccountConflictKind` identifies `EXECUTION_SCOPE`,
`OWNER_GENERATION`, and `OPENING_IDENTITY`; all repository exceptions derive
from `PaperAccountJournalError`.

`replay_account` and `list_accounts` each use one `REPEATABLE READ READ ONLY`
snapshot. The repository decodes and rehashes the opening, manifests, and
settlements; proves exact manifest cardinality, ordinals, ranges, and contiguous
account versions; and fully replays each referenced order/position journal to
cross-check the instruction, terminal ACK/full-fill history, event identities,
trade identities, and payload hashes. It reconstructs each position fill,
paper-economic checkpoint, quote settlement, and account admission causally,
then requires exact agreement from settlement envelopes, postings, balances,
reservations, and stream version/state. Non-canonical Decimal text, any
projection drift, orphan/incomplete facts, or an account with an unclaimed old
position history raises `PaperAccountReplayError`; a scoped list returns no
partial result if one account fails.

The repository is intentionally not exported from the lightweight persistence
facade and has no runtime consumer. Provision accepts opening-only accounts and
does not adopt old version-2 histories. There is no settlement/posting write,
integrated account-first/position-second transaction owner, execution call,
readiness gate, legacy fence, or durable quarantine workflow.
`owner_generation` is immutable opening provenance, not a rotating runtime
fence. Full referenced-position replay and the current N+1 query shape are
acceptable at this unwired checkpoint; bounded replay and snapshots remain a
measured later optimization.

Verification commands for this slice:

```bash
.venv/bin/python -m pytest -q tests/test_paper_account_journal.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_journal.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q \
  tests/postgres/test_paper_account_repository_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  /usr/local/bin/python3.10 -m pytest -q \
  tests/postgres/test_paper_account_repository_postgres.py
.venv/bin/python -m pytest -q \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_order_position_journal.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_order_position_journal.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q tests/postgres
.venv/bin/black --target-version py310 --check \
  trading/persistence/paper_account_journal.py \
  tests/test_paper_account_journal.py \
  tests/postgres/test_paper_account_repository_postgres.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_economics.py \
  tests/test_paper_settlement.py \
  tests/test_position_lifecycle.py
.venv/bin/isort --check-only \
  trading/persistence/paper_account_journal.py \
  tests/test_paper_account_journal.py \
  tests/postgres/test_paper_account_repository_postgres.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_economics.py \
  tests/test_paper_settlement.py \
  tests/test_position_lifecycle.py
.venv/bin/flake8 \
  trading/persistence/paper_account_journal.py \
  tests/test_paper_account_journal.py \
  tests/postgres/test_paper_account_repository_postgres.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_economics.py \
  tests/test_paper_settlement.py \
  tests/test_position_lifecycle.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/persistence/paper_account_journal.py \
  tests/test_paper_account_journal.py \
  tests/postgres/test_paper_account_repository_postgres.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_economics.py \
  tests/test_paper_settlement.py \
  tests/test_position_lifecycle.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The dedicated repository unit suite passes 44 tests under each supported
Python interpreter. The dedicated PostgreSQL 15 suite passes 15 tests under
each interpreter. The exact adjacent repository/codec/accounting/settlement/
order-journal command passes 338 tests under each interpreter, and the complete
PostgreSQL 15 suite passes 97 tests. Black, isort, flake8, Python 3.10
compilation, and the diff check pass. The complete non-PostgreSQL gate passes
2,090 tests, skips 50, deselects 100, and passes 7 subtests under
`Pacific/Honolulu`. Pytest exits successfully; the legacy background threads
still emit their known post-success logging errors after the result.

### M9b.12d atomic paper-account submission owner

M9b.12d adds the application vocabulary for the first transaction boundary that
can own both migration `0002` journal facts and migration `0003` account facts:

- `PaperAccountSubmissionContext` binds one exact `SubmissionAttemptContext`,
  durable account key, and version-1 `PaperLinearInstrument` snapshot;
- `DurablePaperAccountSubmissionReceipt` binds the context to the exact durable
  submission receipt and one positive consecutive account version per fill;
- `PaperAccountSubmissionRejected` carries the rejected fill event identity and
  non-empty derived account-admission reasons;
- `PaperAccountSubmissionResult` is exactly the receipt-or-rejection union and
  `PaperAccountSubmissionOwner.execute(context, /)` is its positional-only port;
  and
- `PaperAccountSubmissionCommitUnknown` and
  `PaperAccountSubmissionReconciliationRequired` preserve the full context and
  require explicit reconciliation.

The concrete, still-unwired adapter is
`trading.persistence.atomic_paper_account_owner.PostgresAtomicPaperAccountOwner`.
It receives one fresh-connection factory and a `PaperSubmissionPlanner`; it does
not call the earlier public atomic owner's `execute` method. The account must be
provisioned first. Each invocation owns one `READ COMMITTED` transaction and
uses the single global lock order: strict `paper_account_streams` replay under
`FOR UPDATE`, followed by creation/lock and strict replay of the target
`position_streams` row.

If the client order already has an exact account manifest, the owner validates
the complete opening generation, context, instruction, instrument, ACK/fill
history, contiguous position/account ranges, settlement hashes, postings, and
materialized projections. It then returns a `REPLAYED` receipt with zero planner
calls and zero DML. Any migration-`0002` order history without that manifest,
including the terminal shape formerly replayable by the position-only owner,
raises `PaperAccountSubmissionReconciliationRequired` before planning. The same
fail-closed result applies to missing, incomplete, corrupt, or incompatible
manifest history; old facts are never adopted as atomically accounted facts.

For a new order, the planner runs once while both rows remain locked. The owner
validates the journal/position transition, builds each `PaperFillRecord`, FIFO
economic transition, quote settlement, and account admission in causal order,
and performs no externally visible action. If any admission is `REJECTED`, it
rolls back the whole transaction and returns `PaperAccountSubmissionRejected`;
all journal and account tables remain byte-for-byte unchanged. If all admissions
are `APPLIED`, the same transaction writes the order, ACK and confirmed fills,
exact batch manifest, compact settlements, postings, balances, margin
reservations, and both stream tails. Deferred constraints are forced immediate,
and a strict account/journal replay must match before the transaction commits
once.

Pre-commit failures roll back and remain known failures. If PostgreSQL may have
committed but its acknowledgement is lost, the owner raises
`PaperAccountSubmissionCommitUnknown` with the complete context; it never reports
success or retries planning internally. A later exact call can resolve the
outcome only through the no-DML manifest replay path. Concurrent exact calls must
therefore yield one commit and one replay, and every failure injected after a DML
statement must leave the complete journal/account snapshot unchanged unless the
single commit itself became unknown.

This slice is deliberately not exported from the lightweight persistence facade
and has no runtime consumer. It adds no venue execution, legacy-table SQL,
compatibility projection, rotating owner generation, readiness check, durable
quarantine workflow, sole-writer fence, shadow mode, or cut-over. The legacy
runtime remains authoritative.

Verification commands for this slice:

```bash
.venv/bin/python -m pytest -q tests/test_paper_account_submission.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_submission.py
.venv/bin/python -m pytest -q tests/test_atomic_paper_account_owner.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_atomic_paper_account_owner.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q \
  tests/postgres/test_atomic_paper_account_owner_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  /usr/local/bin/python3.10 -m pytest -q \
  tests/postgres/test_atomic_paper_account_owner_postgres.py
.venv/bin/python -m pytest -q \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/test_order_position_journal.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/test_order_position_journal.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=<disposable-postgres-15-admin-dsn> \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q tests/postgres
.venv/bin/black --target-version py310 --check \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/persistence/atomic_paper_account_owner.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/postgres/test_atomic_paper_account_owner_postgres.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_economics.py \
  tests/test_paper_settlement.py \
  tests/test_position_lifecycle.py
.venv/bin/isort --check-only \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/persistence/atomic_paper_account_owner.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/postgres/test_atomic_paper_account_owner_postgres.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_economics.py \
  tests/test_paper_settlement.py \
  tests/test_position_lifecycle.py
.venv/bin/flake8 \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/persistence/atomic_paper_account_owner.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/postgres/test_atomic_paper_account_owner_postgres.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_economics.py \
  tests/test_paper_settlement.py \
  tests/test_position_lifecycle.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/persistence/atomic_paper_account_owner.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/postgres/test_atomic_paper_account_owner_postgres.py \
  tests/test_atomic_paper_submission_owner.py \
  tests/test_durable_submission.py \
  tests/test_order_lifecycle.py \
  tests/test_order_position_journal.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_economics.py \
  tests/test_paper_settlement.py \
  tests/test_position_lifecycle.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The pure account-submission contract suite passes 22 tests and the owner-unit
suite passes 12 tests under each supported Python interpreter. The PostgreSQL
15 owner matrix passes 13 tests under each interpreter, including exact replay
with zero DML, a rejection at the second fill with zero durable mutation,
account-before-position locking, exact-call and cross-position concurrency,
mandatory reconciliation for old/incomplete/corrupt or context-incompatible
manifest history, rollback after every traced DML mutation, deferred-constraint
flushing, commit-unknown recovery, and absence of legacy-table SQL. The exact
adjacent command passes 469 tests under each interpreter, and the complete
PostgreSQL 15 suite passes 110 tests. Black, isort, flake8, Python 3.10
compilation, and the diff check pass. The complete non-PostgreSQL gate passes
2,124 tests, skips 50, deselects 113, and passes 7 subtests under
`Pacific/Honolulu`. Pytest exits successfully; the legacy background threads
still emit their known post-success logging errors after the result.

### M9b.13a pure pre-fence assessment contract

M9b.13a introduces the application-only contract for describing evidence that
may later support a sole-writer fence. `PaperAccountReadinessContext` carries an
approved execution scope, account key, immutable provisioning generation, and
opening-envelope hash. `MigrationIdentity` records the exact expected and
applied migration prefixes. `LegacyRelationWatermark` inventories every
migration-`0001` table without adopting or deleting its rows.

`PaperAccountReadinessFindingKind` distinguishes migration absence, pending
versions and drift; missing, unexpected, insolvent, or provenance-incompatible
accounts; account and position replay failures; unresolved submissions and
unaccounted orders; margin reservations; and durable or legacy open positions.
`PaperAccountReadinessAssessment` sorts and deduplicates stable findings,
derives a legacy-open-position blocker from the corresponding watermark, and
returns exactly `PREPARED_FOR_FENCE`, `BLOCKED`, or
`RECONCILIATION_REQUIRED`. Reconciliation findings take precedence over ordinary
blockers.
Malformed raw migration rows are not coerced into `MigrationIdentity` values:
the evidence retains only the decodable prefix and canonicalizes an explicit
`MIGRATION_DRIFT` blocker.

Every assessment reports `snapshot_authoritative == False`. In particular,
`PREPARED_FOR_FENCE` is not permission to start or activate trading. This slice
has no SQL, repository, migration, runtime consumer, health/readiness endpoint,
generation fence, trigger, role change, shadow execution, or cut-over. M9b.13b
now collects the evidence in one `REPEATABLE READ READ ONLY` PostgreSQL
snapshot; the eventual activation transaction must independently re-check it
under the global lock order `fence -> account -> position`.

Verification commands for this slice:

```bash
.venv/bin/python -m pytest -q tests/test_paper_account_readiness.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_readiness.py
.venv/bin/python -m pytest -q \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_order_position_journal.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_accounting.py \
  tests/test_paper_settlement.py \
  tests/test_paper_economics.py \
  tests/test_order_position_journal.py
.venv/bin/black --target-version py310 --check \
  trading/application/paper_account_readiness.py \
  trading/application/__init__.py \
  tests/test_paper_account_readiness.py
.venv/bin/isort --check-only \
  trading/application/paper_account_readiness.py \
  trading/application/__init__.py \
  tests/test_paper_account_readiness.py
.venv/bin/flake8 \
  trading/application/paper_account_readiness.py \
  trading/application/__init__.py \
  tests/test_paper_account_readiness.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/paper_account_readiness.py \
  trading/application/__init__.py \
  tests/test_paper_account_readiness.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The dedicated contract suite passes 122 tests under each supported Python
interpreter, and the exact adjacent command passes 569 tests under each
interpreter. Black, isort, flake8, Python 3.10 compilation, and the diff check
pass. The complete non-PostgreSQL gate passes 2,244 tests, skips 50, deselects
113, and passes 7 subtests under `Pacific/Honolulu`. Pytest exits successfully;
the legacy background threads still emit their known post-success logging
errors after the result.

### M9b.13b dormant global PostgreSQL pre-fence assessment

M9b.13b implements the M9b.13a port through the direct, non-facade
`trading.persistence.paper_account_readiness.PostgresPaperAccountReadiness`.
The constructor accepts one injected fresh-connection factory and
`assess(context, /)` returns a complete `PaperAccountReadinessAssessment` or
raises a typed error. `PaperAccountReadinessError` is the common boundary,
`PaperAccountReadinessInputError` rejects a non-exact context before connection,
and `PaperAccountReadinessStorageError` reports connection, query, packaged
migration, replay-boundary, or snapshot-finalization failure without returning a
partial assessment.

Each call obtains one connection and one cursor, starts a `REPEATABLE READ`,
`READ ONLY` transaction, and rolls it back after the report is built. It never
commits.
The adapter first loads the packaged migration identities and proves the
physical authority of `np.schema_migrations`. It must be one ordinary permanent
table with exactly, in order, `version integer NOT NULL`, `name text NOT NULL`,
`checksum character(64) NOT NULL`, and
`applied_at timestamp with time zone NOT NULL DEFAULT now()`; its only
constraint must be the non-deferrable, initially immediate, validated primary
key on `version`. Rules, triggers, row-level security, forced row-level
security, inheritance, and policies are forbidden. Only then does
the adapter read and compare the contiguous migration prefix. A missing ledger,
incomplete prefix, identity drift, malformed raw row, physical metadata drift,
or behavior overlay produces the canonical migration blocker and stops before
business inventory. A malformed trailing row preserves only its decodable
prefix and is `MIGRATION_DRIFT`, not `MIGRATION_PENDING`.

At the M9b.13b checkpoint, an exact ledger was followed by a second authority
gate covering these sixteen business relations: `np.account_balances`,
`np.liquidations`,
`np.margin_history`, `np.model_predictions`, `np.open_positions`, `np.trades`,
`np.trading_session_resets`, `np.order_events`, `np.orders`,
`np.position_streams`, `np.paper_account_streams`,
`np.paper_account_balances`, `np.paper_margin_reservations`,
`np.paper_account_batch_manifests`, `np.paper_account_settlements`, and
`np.paper_account_postings`. Each must be an ordinary permanent table with no
rules, user triggers, row-level security, forced row-level security,
inheritance, or policies. Missing relations, views or other relation kinds,
non-permanent persistence, and behavior overlays produce `MIGRATION_DRIFT`
before a business row is trusted. M9b.14a extends that gate with the seventeenth
control relation and exact legacy-trigger authority described below.

The M9b.14a legacy fence is global because none of the seven migration-`0001`
tables carries an execution scope. The assessment therefore scans every account
and position identity across every stored scope, not only the requested scope.
It passes the same cursor and each identity's stored scope into strict account
and position replay with `lock=False`; it neither calls a public repository
method nor opens a nested connection. The requested account key and scope must
match exactly. Missing or wrong-scope expected state, extra same- or
foreign-scope accounts, provenance mismatch, insolvency, margin reservations,
replay failures, and durable open positions all remain stable typed findings.

Before replay, the adapter reads the raw global relational claims without
collapsing multiplicity. Orders retain
`(position_key, execution_scope, client_order_id)` and manifests retain
`(account_key, execution_scope, position_key, client_order_id)`. Repeated global
client-order claims fail closed. The complete raw order multiset must equal the
strictly replayed order multiset, and the complete raw manifest multiset must
equal the strictly replayed manifest multiset. Orphan rows, duplicate claims,
rows outside every stream/account, and replay omissions therefore remain
visible.

The adapter also compares raw order claims with raw manifest claims after
normalizing away only the manifest account key. A missing manifest claim is
`UNACCOUNTED_ORDER`; a missing order claim is `ACCOUNT_REPLAY_FAILED`.
Raw-versus-replayed mismatches separately identify `np.orders` as
`POSITION_REPLAY_FAILED` or `np.paper_account_batch_manifests` as
`ACCOUNT_REPLAY_FAILED`. All five non-terminal lifecycle states are
`UNRESOLVED_SUBMISSION`, and an empty, orphaned, corrupt, or foreign-scope
position stream that cannot replay is `POSITION_REPLAY_FAILED`. The adapter then
captures exact row-count and maximum-ID watermarks for
`np.account_balances`, `np.liquidations`, `np.margin_history`,
`np.model_predictions`, `np.open_positions`, `np.trades`, and
`np.trading_session_resets`. A non-empty `np.open_positions` watermark derives
`LEGACY_OPEN_POSITION`; other legacy rows are inventoried, not adopted or
reconciled.

All adapter SQL is read-only: no DML, DDL, `LOCK`, `FOR UPDATE`, or commit is
permitted. The adapter is not exported by `trading.persistence` and has no
production runtime consumer. The result remains explicitly stale-on-return and
`snapshot_authoritative == False`; the PostgreSQL matrix includes a concurrent
legacy commit that is intentionally absent from the already-open repeatable-read
snapshot. Consequently, `PREPARED_FOR_FENCE` is evidence for a later locked
transition, not permission to start trading.

This slice adds no migration, fence record, runtime generation, trigger,
database-role restriction, legacy-writer shutdown, startup or health wiring,
reconciliation mutation, shadow execution, or cut-over. A later activation
boundary must lock the M9b.14a global fence, wait out in-flight legacy
writes, repeat this assessment under `fence -> account -> position`, and prevent
old binaries from writing after activation. Full global replay is currently
unbounded, so bounded replay or snapshots, operational timeouts, soak evidence,
and an explicit rollback decision remain mandatory.

Verification commands for this slice:

```bash
.venv/bin/python -m pytest -q \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q \
  tests/postgres/test_paper_account_readiness_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  /usr/local/bin/python3.10 -m pytest -q \
  tests/postgres/test_paper_account_readiness_postgres.py
.venv/bin/python -m pytest -q \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal.py \
  tests/test_order_position_journal.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal.py \
  tests/test_order_position_journal.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q tests/postgres
.venv/bin/black --target-version py310 --check \
  trading/application/paper_account_readiness.py \
  trading/application/__init__.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_paper_account_readiness_postgres.py
.venv/bin/isort --check-only \
  trading/application/paper_account_readiness.py \
  trading/application/__init__.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_paper_account_readiness_postgres.py
.venv/bin/flake8 \
  trading/application/paper_account_readiness.py \
  trading/application/__init__.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/paper_account_readiness.py \
  trading/application/__init__.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_paper_account_readiness_postgres.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The literal M9b.13b verification commands above passed the contract-plus-
repository pair with 184 tests under each Python interpreter, and the exact
adjacent five-file set with 308 tests under each interpreter. The focused
PostgreSQL 15 readiness suite passed 22 tests under each interpreter; the full
PostgreSQL 15 `tests/postgres` suite passed 132 tests under `.venv/bin/python`.
The full non-PostgreSQL Honolulu run passed 2,308 tests, with 50 skipped, 135
deselected, and 7 subtests. Black, isort, flake8, Python 3.10 compilation, and
the diff check were green.

### M9b.14a dormant global legacy-writer fence

M9b.14a adds the forward-only
`0004_paper_runtime_control.sql` migration without wiring the migration runner
or any new owner into production startup. Its immutable SHA-256 is
`869b015928c8cba7e60838ee1fbeb0006ce4647ef003c936c4f6a354e0306edb`.
It creates the ordinary permanent
singleton table `np.paper_runtime_control` with this exact initial authority:

- `control_key boolean NOT NULL PRIMARY KEY DEFAULT TRUE`, constrained to
  `TRUE` so at most one canonical row can exist;
- `mode text NOT NULL`, constrained to `LEGACY`, `SHADOW`, `PAUSED`, or
  `ACTIVE`;
- `runtime_generation bigint NOT NULL`, constrained non-negative; and
- `updated_at timestamptz NOT NULL DEFAULT now()`.

The migration inserts exactly `(TRUE, 'LEGACY', 0)`. `runtime_generation` is a
dedicated future runtime-ownership epoch. It is not the immutable
`owner_generation` that binds a paper-account opening and its manifests, and
this slice neither copies nor derives one from the other.

The zero-argument `np.enforce_legacy_paper_runtime_fence()` trigger function is
`SECURITY DEFINER`, volatile, non-strict, and configured with
`search_path=pg_catalog`. Each statement takes a `FOR SHARE` lock on the control
row before it decides whether legacy DML is allowed. Missing control state fails
closed with SQLSTATE `55000` and `paper runtime control is unavailable`;
ill-formed state fails with `paper runtime control is invalid`. `LEGACY` and
`SHADOW` return without blocking the statement. `PAUSED` and `ACTIVE` fail with
SQLSTATE `55000` and `legacy paper writes are fenced in <mode> mode`.

The migration installs these seven exact guards:

- `legacy_paper_runtime_fence_account_balances` on `np.account_balances`;
- `legacy_paper_runtime_fence_liquidations` on `np.liquidations`;
- `legacy_paper_runtime_fence_margin_history` on `np.margin_history`;
- `legacy_paper_runtime_fence_model_predictions` on `np.model_predictions`;
- `legacy_paper_runtime_fence_open_positions` on `np.open_positions`;
- `legacy_paper_runtime_fence_trades` on `np.trades`; and
- `legacy_paper_runtime_fence_trading_session_resets` on
  `np.trading_session_resets`.

Every guard is `BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE`, statement-level,
and `ENABLE ALWAYS`. The migration therefore covers every DML operation used by
the legacy persistence helpers and does not depend on row cardinality. Because
the seed is `LEGACY`, applying `0004` alone preserves current runtime behavior;
it does not silently pause or activate a writer. `SHADOW` also permits legacy
DML and does not itself execute a second path.

The M9b.13b assessment advances with the migration. Its authority inventory now
contains the original sixteen business tables plus
`np.paper_runtime_control`. The seven legacy tables are the only relations
allowed to report user triggers; a separate exact catalog gate then verifies
the four control columns and defaults, all four named validated constraints,
the function signature, language, volatility, security-definer flag, safe
configuration, owner, and full PostgreSQL function source. It verifies exactly
seven user triggers across the namespace, each with its canonical relation,
name, `ENABLE ALWAYS` state, complete statement-level DML mask, and exact
`np.enforce_legacy_paper_runtime_fence` target. An extra or altered trigger,
function, relation, column, default, constraint, or behavior overlay remains
`MIGRATION_DRIFT`.

Readiness next reads the raw singleton `(control_key, mode,
runtime_generation)`. An absent or repeated row, non-`TRUE` key, unknown mode,
non-integer generation, negative generation, or out-of-range generation is
early `MIGRATION_DRIFT` and prevents account, position, order, manifest, and
legacy-watermark reads. A structurally valid `SHADOW`, `PAUSED`, or `ACTIVE`
row is not schema drift: the adapter records
`RUNTIME_CONTROL_NOT_LEGACY` for `np.paper_runtime_control` and continues the
complete global evidence collection. Only valid `LEGACY` state can produce a
finding-free `PREPARED_FOR_FENCE` result in M9b.14a. That result remains
`snapshot_authoritative == False` and grants no transition authority.

This slice intentionally adds no public activation, pause, shadow, generation,
or rollback API. It changes no database roles or grants, does not revoke schema
`CREATE` or object ownership from the current runtime identity, and adds no
startup, health, CLI, executor, atomic-owner, or compatibility-projection
wiring. A superuser or object owner can still alter the fence. Consequently,
`ACTIVE` remains prohibited until all of the following are implemented and
proved:

- separate migration/admin ownership and a non-superuser runtime role without
  schema/object ownership, DDL, or schema `CREATE` privileges;
- one locked transition boundary that waits out legacy statements, re-runs
  authoritative evidence in `fence -> account -> position` order, advances
  `runtime_generation`, and binds that generation to every durable-owner write;
- startup and restart validation that fails closed on mode, generation,
  migration, role, replay, or ownership mismatch;
- reconciliation/quarantine for unsupported history plus a bounded replay or
  snapshot plan, operational timeouts, and measured soak evidence;
- side-effect-free `SHADOW` parity and an explicit compatibility-projection
  policy; and
- removal or proved fencing of every legacy writer, together with a tested
  `PAUSED` and rollback procedure that never creates two authoritative owners.

The verification gate for this slice includes migration package/upgrade and
catalog checks, direct PostgreSQL DML tests for all four operations on all seven
relations in all four modes, malformed/missing-control fail-closed cases,
transaction and generation visibility, exact readiness-catalog drift tests,
both supported Python interpreters, the full PostgreSQL suite, the complete
non-PostgreSQL suite, formatting/static checks, compilation, and
`git diff --check`.

Literal verification commands for this slice:

```bash
.venv/bin/python -m pytest -q \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_migration_runner.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_migration_runner.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_control_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  /usr/local/bin/python3.10 -m pytest -q \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_control_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q tests/postgres
.venv/bin/black --target-version py310 --check \
  trading/application/paper_account_readiness.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_migration_runner.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_control_postgres.py
.venv/bin/isort --check-only \
  trading/application/paper_account_readiness.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_migration_runner.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_control_postgres.py
.venv/bin/flake8 \
  trading/application/paper_account_readiness.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_migration_runner.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_control_postgres.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/paper_account_readiness.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_migration_runner.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_control_postgres.py
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
git diff --check
```

The literal M9b.14a focused contract/repository/runner command passed 228 tests
under each Python interpreter. The focused PostgreSQL 15 migration/readiness/
fence command passed 181 tests under each interpreter; the complete PostgreSQL
15 suite passed 282 tests under `.venv/bin/python`. The full non-PostgreSQL
Honolulu run passed 2,317 tests, with 50 skipped, 285 deselected, and 7 subtests.
Black, isort, flake8, Python 3.10 compilation, and `git diff --check` were green.

### M9b.14b1 dormant runtime-generation provenance

M9b.14b1 adds forward migration
`0005_paper_runtime_generation.sql`. Its immutable SHA-256 is
`ac995eae0477697dc5517cc377d9af6f2411a53c0fd342e4773964c74d2a3358`.
The migration deliberately leaves its new
`np.paper_runtime_generations` table empty and does not change the M9b.14a
`LEGACY/0` seed. Each future epoch row contains:

- positive primary-key `runtime_generation`;
- one globally unique, trimmed, non-empty `activation_id`, reserved as the
  retry identity for resolving a commit-unknown activation;
- the exact `execution_scope`, `account_key`, positive immutable
  `owner_generation`, opening version, and opening-payload SHA-256;
- finite `activated_at timestamptz NOT NULL DEFAULT clock_timestamp()`; and
- an exact foreign key from the complete opening identity to
  `np.paper_account_streams`, plus the composite uniqueness required as the
  manifest provenance target.

`runtime_generation` is an activation epoch and remains distinct from
`owner_generation`: the latter records how an account was provisioned and does
not rotate. The dormant M9b.14b3 transition appends epoch `N` only when entering
`ACTIVE N`. `PAUSED` retains `N`; a subsequent entry into `ACTIVE` appends
`N+1`. The unique `activation_id` lets a caller resolve whether an activation
transaction committed without creating another epoch or binding the wrong
account.

The epoch table is append-only at the database boundary. The migration creates
the exact zero-argument function
`np.reject_paper_runtime_generation_mutation()`, which is `SECURITY DEFINER`,
uses PL/pgSQL, and fixes `search_path=pg_catalog`. Its exact
`paper_runtime_generations_append_only` trigger is statement-level, `ENABLE
ALWAYS`, and fires before `UPDATE`, `DELETE`, or `TRUNCATE`; statements that
reach it fail with SQLSTATE `55000` and
`paper runtime generations are append-only`. A referenced epoch is also
protected by the manifest foreign key. PostgreSQL may reject a plain
`TRUNCATE` with `0A000` during foreign-key graph validation before the trigger;
`TRUNCATE ... CASCADE` reaches the guard and returns `55000`. Both outcomes are
zero-mutation and are covered explicitly rather than overclaiming one error
path.

The migration appends nullable `runtime_generation bigint` to
`np.paper_account_batch_manifests` and replaces its version check with exactly
two admissible forms:

- version 1 and `runtime_generation IS NULL`; or
- version 2 and `runtime_generation IS NOT NULL AND runtime_generation > 0`.

The explicit `IS NOT NULL` is required because a PostgreSQL `CHECK` otherwise
accepts an unknown result for version 2 plus `NULL`. A version-2 row also has a
composite foreign key from its generation and full opening identity to the
same epoch. It therefore cannot stamp another account's activation merely by
guessing a valid generation number.

The journal codec and replay remain backward compatible without adopting old
history. A manifest with no runtime generation emits the byte-identical,
hash-identical version-1 canonical JSON. A positive generation selects version
2, includes `runtime_generation` in the canonical payload and payload hash, and
cross-checks it against the indexed column while decoding. Replay now exposes
`runtime_generation: int | None`; it performs no rewrite or implicit V1-to-V2
upgrade.

Readiness advances both its catalog and raw-data authority. Its durable
inventory now expects the original sixteen business relations, M9b.14a control,
and the generation registry. The seven legacy-fence triggers plus the one
append-only epoch trigger are the complete allowed user-trigger set. Exact
catalog checks cover every new column, default, validated constraint, foreign
key, function property/source/owner/configuration, trigger target/event
mask/state, and the manifest column at ordinal 22. Extra or altered
generation-bearing schema remains early `MIGRATION_DRIFT`.

After catalog validation, raw generation evidence must satisfy all of these
invariants:

- control generation `N` has exactly the immutable epoch rows `1..N`, with no
  gap, duplicate activation identity, or context/opening mismatch;
- every raw manifest is version 2, carries a non-boolean integer generation in
  `1..N`, and has the full provenance recorded by that exact epoch;
- `LEGACY/0` has no epoch rows and no manifests; `ACTIVE/0` is always invalid;
  and `SHADOW/0` or `PAUSED/0` may be structurally valid but remain blocked by
  their existing non-legacy control finding.

Failure adds stable `RUNTIME_GENERATION_MISMATCH` for
`np.paper_runtime_generations`. It is an ordinary `BLOCKED` readiness finding,
not a claim that automated reconciliation is available. Existing V1 histories
remain replayable, but their presence blocks activation and is never treated as
durable-owner provenance. With an empty registry and manifest inventory, the
new generation evidence does not change an otherwise exact `LEGACY/0`
`PREPARED_FOR_FENCE` result. The snapshot remains non-authoritative and cannot
transition the runtime.

This slice is strictly dormant. It adds no generation-aware owner write, epoch
insert API, mode transition, same-cursor readiness lock, startup/runtime/health
composition, shadow executor, role or grant change, credential rotation,
compatibility projection, reconciliation mutation, policy digest, cut-over, or
rollback authority. No current production path can create version-2 manifests
or enter `ACTIVE`.

The delivery order is now recorded as follows:

1. M9b.14b2 made the atomic owner generation-aware while keeping that owner
   dormant; every new durable batch uses V2 and the supplied epoch's exact
   account provenance.
2. M9b.14b3 adds the dormant locked transition contract documented below,
   including the same-cursor locked readiness re-check and commit-unknown
   recovery by `activation_id`; `ACTIVE` remains unreachable in production.
3. M9b.14c remains the migration/bootstrap entrypoint, migration/admin
   ownership separation, non-superuser runtime design, exact privilege
   boundary, schema `CREATE`/DDL removal, and affected-secret rotation.
4. M9b.14d remains fail-closed startup/composition and side-effect-free shadow
   operation. Explicit reconciliation/quarantine, bounded replay or snapshots,
   compatibility policy, stale-writer removal, tested pause/rollback, soak
   evidence, and an operator-approved cut-over still gate `ACTIVE`.

Literal verification commands for M9b.14b1 are recorded below. Their focused
shared, both-interpreter, PostgreSQL, full-suite, and static-check results were
captured from the final shared tree.

```bash
/usr/bin/shasum -a 256 \
  trading/persistence/sql_migrations/0005_paper_runtime_generation.sql
.venv/bin/python -m pytest -q \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_runtime_generation_postgres.py \
  tests/postgres/test_paper_account_repository_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  /usr/local/bin/python3.10 -m pytest -q \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_runtime_generation_postgres.py \
  tests/postgres/test_paper_account_repository_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q tests/postgres
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
.venv/bin/black --target-version py310 --check \
  trading/application/paper_account_readiness.py \
  trading/persistence/paper_account_journal.py \
  trading/persistence/paper_account_journal_codec.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_generation_postgres.py
.venv/bin/isort --check-only \
  trading/application/paper_account_readiness.py \
  trading/persistence/paper_account_journal.py \
  trading/persistence/paper_account_journal_codec.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_generation_postgres.py
.venv/bin/flake8 \
  trading/application/paper_account_readiness.py \
  trading/persistence/paper_account_journal.py \
  trading/persistence/paper_account_journal_codec.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_generation_postgres.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/paper_account_readiness.py \
  trading/persistence/paper_account_journal.py \
  trading/persistence/paper_account_journal_codec.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_migration_runner.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_migration_runner_postgres.py \
  tests/postgres/test_paper_account_readiness_postgres.py \
  tests/postgres/test_paper_runtime_generation_postgres.py
git diff --check
```

The exact focused shared command passed 465 tests under each Python interpreter.
The focused PostgreSQL 15 command passed 63 tests under each interpreter; the
complete PostgreSQL 15 suite passed 304 tests under `.venv/bin/python`. The full
non-PostgreSQL Honolulu run passed 2,377 tests, with 50 skipped, 307 deselected,
and 7 subtests. Black, isort, flake8, Python 3.10 compilation, and
`git diff --check` were green.

### M9b.14b2 dormant generation-aware atomic owner

M9b.14b2 consumes the M9b.14b1 provenance format only inside the still-unwired
`PostgresAtomicPaperAccountOwner`. The application contract adds required
positive-bigint `runtime_generation` to `PaperAccountSubmissionContext`; the
owner constructor independently requires the positive generation it is pinned
to for its lifetime. A future context generation is rejected before encoding,
connection creation, planner invocation, or DML by the exported frozen
`PaperAccountSubmissionRuntimeUnavailable(context)`. The exception preserves
the client identity and full generation-bearing context and deliberately has
`requires_reconciliation == False`, because no owner write could have occurred.

Within a fresh transaction, the exact lock and validation order is:

1. `SET TRANSACTION ISOLATION LEVEL READ COMMITTED`;
2. select the singleton `np.paper_runtime_control` `FOR SHARE` and require
   `ACTIVE` with generation equal to the constructor pin;
3. select that exact `np.paper_runtime_generations` row `FOR SHARE`;
4. lock and strictly replay the requested account, then match the pinned epoch
   to its scope, key, immutable `owner_generation`, opening version, and opening
   payload SHA-256;
5. require every existing account manifest to be V2 with a generation no later
   than the pin, resolve the target manifest and its exact context generation;
   and
6. only then lock or create the position stream and continue the prior atomic
   replay, rejection, or commit path.

The shared control lock prevents a concurrent generation transition from
committing until the owner transaction finishes. Missing, malformed,
non-`ACTIVE`, stale, or provenance-incompatible runtime state is typed runtime
unavailability. V1 history, future-stamped account history, or a target
manifest whose generation differs from the submission context is instead
`PaperAccountSubmissionReconciliationRequired`; the owner never upgrades or
adopts those facts.

A genuinely new batch requires context generation equal to the constructor
pin. Its manifest is V2 and includes the generation in both the indexed column
and canonical hashed payload. All journal, settlement, posting, balance,
reservation, and stream facts retain the prior one-transaction invariant. A
derived account rejection still produces zero durable batch mutations.

Rollover permits one narrower case. An owner pinned to generation `N+1` may
resolve an exact existing V2 manifest whose retained context is generation `N`.
The complete account and position replay must agree, the result keeps that
generation-`N` context, the planner does not run, and no DML occurs. This makes
a generation-`N` `PaperAccountSubmissionCommitUnknown` resolvable after a
rollover. It does not authorize a new old-generation batch: an absent target
with context `N` returns runtime-unavailable before planner or DML. Conversely,
an owner pinned to `N` is unavailable after control advances to `N+1`.

This is still a dormant proof boundary. No production module constructs the
owner or its context, and the slice adds no transition API, epoch insertion,
mode mutation, readiness lock, startup/runtime/health wiring, shadow executor,
role or grant change, secret rotation, legacy compatibility projection,
reconciliation mutation, or cut-over authority. M9b.14b3, documented below,
supplies the dormant locked same-cursor transition; M9b.14c remains bootstrap
and database-role separation; M9b.14d remains fail-closed composition and
shadow operation.

Literal focused verification commands for M9b.14b2:

```bash
.venv/bin/python -m pytest -q \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q \
  tests/postgres/test_atomic_paper_account_owner_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  /usr/local/bin/python3.10 -m pytest -q \
  tests/postgres/test_atomic_paper_account_owner_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q tests/postgres
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q --disable-warnings \
  tests/ -m 'not perf and not postgres'
.venv/bin/black --target-version py310 --check \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/persistence/atomic_paper_account_owner.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/postgres/test_atomic_paper_account_owner_postgres.py
.venv/bin/isort --check-only \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/persistence/atomic_paper_account_owner.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/postgres/test_atomic_paper_account_owner_postgres.py
.venv/bin/flake8 \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/persistence/atomic_paper_account_owner.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/postgres/test_atomic_paper_account_owner_postgres.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/durable_submission.py \
  trading/application/__init__.py \
  trading/persistence/atomic_paper_account_owner.py \
  tests/test_paper_account_submission.py \
  tests/test_atomic_paper_account_owner.py \
  tests/postgres/test_atomic_paper_account_owner_postgres.py
git diff --check
```

The focused contract and owner command passed 53 tests under each Python
interpreter. The focused PostgreSQL 15 owner command passed 24 tests under each
interpreter, including proof that its control `FOR SHARE` lock blocks a
concurrent control update until commit. The complete PostgreSQL 15 suite passed
315 tests. The non-PostgreSQL Honolulu run passed 2,396 tests, with 50 skipped,
318 deselected, and 7 subtests. Black, isort, flake8, Python 3.10 compilation,
and `git diff --check` were green.

### M9b.14b3 dormant locked activation boundary

M9b.14b3 adds the pure application contract in
`trading/application/paper_runtime_activation.py` and the deliberately
unexported persistence adapter in
`trading/persistence/paper_runtime_activation.py`. The application facade
exports:

- `PaperRuntimeActivationSource` with only `LEGACY` and `PAUSED`;
- `PaperRuntimeActivationContext(readiness, activation_id, source,
  expected_runtime_generation)`, whose target is exactly the next PostgreSQL
  bigint generation;
- `PaperRuntimeActivationDisposition.ACTIVATED | REPLAYED`, the receipt,
  blocked result, result union, and positional-only port; and
- frozen, copy/pickle-safe `PaperRuntimeActivationBusy`,
  `PaperRuntimeActivationConflict`, and
  `PaperRuntimeActivationCommitUnknown`, each retaining the exact context and
  activation ID. Busy is retryable without reconciliation; Conflict and
  CommitUnknown require reconciliation by that identity.

The context enforces the transition shape before connection I/O. A `LEGACY`
source requires expected generation `0`; a `PAUSED` source requires a positive
expected generation; `target_runtime_generation` is exactly expected plus one
and must fit PostgreSQL `bigint`. The activation ID is trimmed, non-empty,
free of NUL and isolated surrogate characters, and no longer than 255
characters. The typed exceptions keep their business fields frozen while
permitting only the traceback/cause/context/suppression/notes attributes Python
needs to propagate exceptions correctly.

The adapter starts one fresh `REPEATABLE READ` transaction, applies
`SET LOCAL lock_timeout = '1s'`, and then issues one canonical table lock:
`LOCK TABLE ... IN SHARE MODE NOWAIT`. Every relation is prefixed by `ONLY`.
The exact nineteen targets are `np.schema_migrations` and the eighteen
relations in the readiness durable-business inventory: the seven legacy
tables; `np.order_events`; `np.orders`; `np.paper_account_balances`;
`np.paper_account_batch_manifests`; `np.paper_account_postings`;
`np.paper_account_settlements`; `np.paper_account_streams`;
`np.paper_margin_reservations`; `np.paper_runtime_control`;
`np.paper_runtime_generations`; and `np.position_streams`. The list is
canonical and lexically ordered in SQL. This boundary fences direct legacy,
journal, manifest, migration-ledger, control, and epoch writers before the
transaction collects evidence. PostgreSQL 15 also proves that the same
transaction can promote its own SHARE locks to insert an epoch and update the
control row.

After the table fence, the transaction validates exact migration and authority
catalogs, selects the singleton control `FOR UPDATE NOWAIT`, and revalidates
the catalog under that lock. Missing or malformed control state becomes a
locked readiness assessment and stable `MIGRATION_DRIFT` when that assessment
can be collected; inability to establish the initial table fence remains a
storage error. It never trusts an activation-ID row before catalog authority
and current control/generation coherence are proven.

For a new transition, control must exactly match the context's
`LEGACY/0` or `PAUSED/N` expectation. The refactored readiness collector runs
on that same cursor, requires the requested mode, locks/replays the account,
then locks/replays positions in sorted identity order. Its public read-only
entrypoint remains behavior-compatible. Any non-prepared assessment returns
`PaperRuntimeActivationBlocked` and explicitly rolls back. The assessment's
`snapshot_authoritative` property remains false: the locked evidence justified
refusal inside the transaction, but the returned object is stale and is not an
activation capability.

Prepared evidence allows exactly three mutation steps before one commit:

1. insert target generation `N+1`, the unique activation ID, and exact opening
   provenance into `np.paper_runtime_generations`;
2. compare-and-swap the singleton from the exact source/generation to
   `ACTIVE/N+1`; and
3. execute `SET CONSTRAINTS ALL IMMEDIATE` before committing once.

A `55P03` lock failure or timeout and `40P01` deadlock become
`PaperRuntimeActivationBusy` before commit. A stale source/generation,
activation-ID collision or mismatch, failed CAS, or unique collision becomes
`PaperRuntimeActivationConflict`. Both paths roll back completely. Other
known-precommit storage failures remain
`PaperRuntimeActivationStorageError`. Only an exception from the final commit
becomes `PaperRuntimeActivationCommitUnknown`; the retained ID and context are
the exact reconciliation input.

An existing exact activation ID is a read-only replay path, including after
later valid control progression. The target epoch must match the context and
full account-opening provenance; current control must be `PAUSED` or `ACTIVE`
at that generation or later; and the complete current epoch prefix and raw V2
manifest provenance must be exact. A stray row under `LEGACY/0`, a gap,
corrupt manifest, wrong account, or reused ID is never replayed. `REPLAYED` and
`Blocked` explicitly roll back and never commit, so neither can produce
CommitUnknown. Only a successfully committed activation or the exact replay of
its immutable row produces a receipt.

This slice remains dormant and unwired. No production composition constructs
the adapter; the persistence facade does not export it; and there is no
startup, health, executor, CLI, shadow, pause, rollback, or cut-over consumer.
It changes no migration, database role, grant, object owner, secret, or legacy
projection. M9b.14c must add the migration/bootstrap entrypoint, assign a
dedicated least-privilege capability that can lock/read all nineteen authority
relations, insert epochs, and update control, separate migration ownership from
runtime, remove schema `CREATE`/DDL from runtime, and rotate the affected
credential. PostgreSQL 15 has no standalone table `LOCK` grant: `SELECT` alone
fails with `42501` for this `SHARE` lock, while a non-`SELECT` table privilege
permits it. M9b.14c must therefore choose a narrowly callable
owner/`SECURITY DEFINER` boundary or another audited design that does not grant
general DML over all nineteen relations. M9b.14d must add fail-closed
startup/composition and side-effect-free shadow operation. Bounded replay or
snapshots, reconciliation/quarantine, compatibility policy, stale-writer
removal, tested pause/rollback, soak evidence, and explicit operator approval
still block an `ACTIVE` cut-over.

Literal focused verification commands for M9b.14b3:

```bash
.venv/bin/python -m pytest -q \
  tests/test_paper_runtime_activation.py \
  tests/test_paper_runtime_activation_adapter.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_runtime_activation.py \
  tests/test_paper_runtime_activation_adapter.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py
.venv/bin/python -m pytest -q \
  tests/test_paper_runtime_activation.py \
  tests/test_paper_runtime_activation_adapter.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_atomic_paper_account_owner.py \
  tests/test_migration_runner.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_paper_runtime_activation.py \
  tests/test_paper_runtime_activation_adapter.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/test_paper_account_journal.py \
  tests/test_paper_account_journal_codec.py \
  tests/test_atomic_paper_account_owner.py \
  tests/test_migration_runner.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q \
  tests/postgres/test_paper_runtime_activation_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  /usr/local/bin/python3.10 -m pytest -q \
  tests/postgres/test_paper_runtime_activation_postgres.py
ELVIS_TEST_POSTGRES_ADMIN_DSN=postgresql://postgres:review@127.0.0.1:55440/elvis_review \
  ELVIS_TEST_POSTGRES_REQUIRED=1 \
  .venv/bin/python -m pytest -q tests/postgres
TZ=Pacific/Honolulu .venv/bin/python -m pytest -q -m 'not postgres'
.venv/bin/black --target-version py310 --check \
  trading/application/paper_runtime_activation.py \
  trading/application/__init__.py \
  trading/persistence/paper_runtime_activation.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_runtime_activation.py \
  tests/test_paper_runtime_activation_adapter.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_paper_runtime_activation_postgres.py
.venv/bin/isort --check-only \
  trading/application/paper_runtime_activation.py \
  trading/application/__init__.py \
  trading/persistence/paper_runtime_activation.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_runtime_activation.py \
  tests/test_paper_runtime_activation_adapter.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_paper_runtime_activation_postgres.py
.venv/bin/flake8 \
  trading/application/paper_runtime_activation.py \
  trading/application/__init__.py \
  trading/persistence/paper_runtime_activation.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_runtime_activation.py \
  tests/test_paper_runtime_activation_adapter.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_paper_runtime_activation_postgres.py \
  --max-line-length=88
/usr/local/bin/python3.10 -m compileall -q \
  trading/application/paper_runtime_activation.py \
  trading/application/__init__.py \
  trading/persistence/paper_runtime_activation.py \
  trading/persistence/paper_account_readiness.py \
  tests/test_paper_runtime_activation.py \
  tests/test_paper_runtime_activation_adapter.py \
  tests/test_paper_account_readiness.py \
  tests/test_paper_account_readiness_repository.py \
  tests/postgres/test_paper_runtime_activation_postgres.py
git diff --check
```

The focused application/adapter/readiness command passed 261 tests under each
Python interpreter. The adjacent journal/owner/migration command passed 540
tests under each interpreter. The focused PostgreSQL 15 activation command
passed 24 tests under each interpreter, and the complete PostgreSQL 15 suite
passed 339 tests. The full non-PostgreSQL Honolulu run passed 2,449 tests, with
50 skipped, 339 deselected, and 7 subtests. That full gate also validates the
exact repository-consumer allowlist update in
`tests/test_order_position_journal.py`: the dormant transition adapter is now
an intentional consumer of the order/position journal internals, not a runtime
composition consumer. Black, isort, flake8, Python 3.10 compilation, and
`git diff --check` were green.

### M9b.14c1 dormant activation database capabilities

M9b.14c1 adds forward migration
`0006_paper_runtime_activation_capabilities.sql`. Its immutable SHA-256 is
`e01c02d1e64b8b136e80dcf2fe365dc85df72d4e1cfa58a8a13b14e4b3f6449d`.
Applying it creates no epoch, changes no control mode, assigns no role or
credential, and exposes no production entrypoint. It creates only two narrowly
callable `SECURITY DEFINER` functions, fixes both search paths to `pg_catalog`,
and executes `REVOKE ALL ... FROM PUBLIC` for both signatures.

`np.acquire_paper_runtime_activation_fence()` returns `void`. It acquires the
M9b.14b3 authority fence on the exact nineteen lexically ordered `ONLY`
relations in `SHARE MODE NOWAIT`: the migration ledger, seven legacy tables,
all journal/account relations, runtime control, and runtime generations. It
then selects the singleton control row `FOR UPDATE NOWAIT`, every account stream
in `account_key` order `FOR UPDATE NOWAIT`, and every position stream in
`position_key` order `FOR UPDATE NOWAIT`. This preserves the global
control-before-account-before-position order. The table fence blocks new
durable DML; the ordered row drain refuses activation when an older account or
position writer already owns a row lock. The locks persist in the caller's
transaction after the function returns.

`PostgresPaperRuntimeActivation` calls this capability from a `READ COMMITTED`
transaction and no longer executes table or replay-row locks directly. This
isolation level is load-bearing. A function invocation establishes its outer
statement snapshot before PL/pgSQL begins locking; retaining `REPEATABLE READ`
would let later readiness queries reuse a snapshot from before the fence was
established. Transaction and local lock-timeout settings do not establish a
data snapshot, so the capability call remains the first data-bearing statement.
The next `READ COMMITTED` statement sees every commit completed before the
fence, while retained table and row locks keep subsequent readiness and replay
reads stable. Account and position replay can therefore use plain reads.

The second capability is
`np.activate_paper_runtime_generation(expected_mode text,
expected_generation bigint, target_generation bigint,
requested_activation_id text, requested_execution_scope text,
requested_account_key text, requested_owner_generation bigint,
requested_opening_payload_sha256 text)`, returning
`TABLE(mode text, runtime_generation bigint)`. It reacquires the complete fence
and admits only the exact shapes `LEGACY/0 -> ACTIVE/1` and
`PAUSED/N -> ACTIVE/N+1` for positive `N`. It validates the bigint successor
and canonical activation/opening argument shapes before inserting the epoch
under the existing opening foreign key and compare-and-swapping control.
Invalid argument shapes raise SQLSTATE `22023`; relational provenance remains
constraint-enforced. A CAS that changes no singleton row raises the stable
custom SQLSTATE `PT001`, aborting the statement so its preceding epoch insert
cannot be committed as an orphan. The adapter maps `PT001` and unique
collisions to `PaperRuntimeActivationConflict`; its existing Busy, Blocked,
exact replay, rollback, and commit-unknown contracts remain unchanged.

The readiness authority gate advances with migration `0006`. It requires both
functions with their exact identity arguments, result shapes, PL/pgSQL source,
volatility, strictness, set-return behavior, `SECURITY DEFINER` state, safe
search path, and a single common owner. Each function ACL must contain exactly
one non-grantable `EXECUTE` entry from that owner to itself. `PUBLIC EXECUTE` or
any third-party grant is catalog drift. The gate also checks that the owner has
effective schema usage; one of PostgreSQL's table-lock-enabling `UPDATE`,
`DELETE`, or `TRUNCATE` privileges on every one of the nineteen relations;
`SELECT` plus `UPDATE` for control, account-stream, and position-stream row
locks; generation `INSERT`; and control `UPDATE`. It does not require that
owner to equal owners from migrations `0001` through `0005`, so exact upgrades
applied by distinct historical owners remain valid when the capability owner
has the required effective privileges.

The mutation function is an offline trusted capability, not a general runtime
API. Its owner can invoke the exact CAS directly and thereby bypass the Python
readiness replay. M9b.14c1 intentionally creates no third-party grant and
remains dormant and unwired. At a successful offline M9b.14c2 `COMPLETE`, both
function ownerships belong to one isolated activation authority and retain the
owner-only ACL in the reconciled catalog. Removing the currently composed
runtime DDL path, deployment, fail-closed composition, and operator-owned
credential rotation remain later gates. Until those gates exist—and the
remaining
reconciliation, bounded-replay, stale-writer, rollback, shadow-soak, and
operator-approval gates pass—entering `ACTIVE` remains an explicit **NO-GO**.

The focused unit, application-contract, adapter, readiness, migration-runner,
and adjacent repository command passed 345 tests under each Python
interpreter. The dedicated PostgreSQL 15 capability suite passed 13 tests under
each interpreter. The complete PostgreSQL 15 suite passed 352 tests under
`.venv/bin/python`. The full non-PostgreSQL Honolulu run passed 2,453 tests,
with 50 skipped, 352 deselected, 293 warnings, and 7 subtests. Black targeting
Python 3.10, isort, flake8 with an 88-character limit, Python 3.10 compilation,
and `git diff --check` were green on the final slice.

### M9b.14c2-c3a dormant role/catalog bootstrap and pre-role admission

M9b.14c2 adds the operator-driven
`trading.persistence.postgres_bootstrap` boundary. Its context, receipts, and
typed errors are secret-free; connection factories may close over credentials
but are excluded from value repr and error graphs. The context fixes one
database, an independent authenticated superuser admin, seven pairwise-distinct
managed roles, and an optional explicit existing-volume adoption authority.
The roles are a `NOLOGIN` schema owner and six separately authenticated login
identities: migrator, legacy runtime, atomic runtime, offline activation,
readiness, and trainer. All are least-privilege, `NOINHERIT`, marker-bound to
the target database, and free of role/database settings. Migrator is the only
member of schema owner and must use explicit `SET ROLE`.

The bootstrap is resumable and fails closed. A fresh first pass creates the
seven roles with null passwords and no login capability, commits that exact
staging state, and returns `CREDENTIALS_REQUIRED`; it neither creates the
schema nor deploys credentials. After an external operator provisions six
credentials, a later pass proves each fresh connection's authenticated backend
identity plus a non-null, non-expired catalog password state before using
migrator to apply packaged migrations `0001` through `0006`. SCRAM and HBA
enforcement remain c3 deployment responsibilities. Existing-volume adoption
requires the exact
checksummed ledger and one declared migration authority owning the complete
historical catalog. Partial history, unledgered legacy relations, mixed owners,
unexpected schemas, public routines, large objects, or surplus grants are
drift and are not repaired.

Catalog admission now precedes every cluster-global managed-role mutation under
the same advisory lock. It accepts only a closed empty fresh database, the
exact prepared fresh-resume state, an exact checksummed historical adoption,
or the exact terminal catalog. A missing or partial ledger, hostile `np`
schema, mixed owner, surplus authority, or unreadable catalog fails with the
existing typed error taxonomy before `CREATE ROLE`, `COMMENT ON ROLE`, or role
membership grants can execute. Already staged exact `NOLOGIN` roles are not
silently deleted if a later operator presents an inadmissible volume.
The admission inventory proves the exact built-in PL/pgSQL extension, all
built-in language rows, referenced PL/pgSQL and access-method handler routines,
and the PL/pgSQL dependency graph. These authority-bearing objects must belong
to the independently authenticated admin. It closes the `public`, prepared
`np`, and user-created `pg_catalog` roots and rejects database-scoped event
triggers, foreign-data wrappers/servers/mappings, publications/subscriptions,
user casts/transforms, default ACLs, relevant settings/parameter ACLs, security
labels, and large objects. Therefore a hook or standalone catalog object cannot
be hidden behind an empty relation inventory and reach the migration phase.
An existing volume whose former shared superuser owns the PL/pgSQL baseline is
rejected rather than silently repaired; remediation belongs to a separately
reviewed offline rehearsal on a clone or a fresh admin-owned target.
The advisory lock coordinates only bootstrap processes that honor it. This
ordering is valid only inside a c3 operator-enforced exclusive DDL and
role-administration window; a concurrent superuser can otherwise mutate the
catalog between evidence collection and role creation, so running without that
quiescence remains a **NO-GO**.

Old shared-runtime retirement is a separate durable barrier. With explicit
demotion intent, memberships must already be absent; one pass proves the
adoption candidate, removes login, password, inheritance, and every
cluster-level privilege, then returns `DEMOTION_REQUIRED`. The catalog cut-over
occurs only on a later pass after old sessions have drained and the role remains
exactly inert. Without
explicit demotion intent, the same status is returned after read-only
preflight. The terminal `COMPLETE` receipt is emitted only after database,
schema, object, function, role, membership, migration, shape, ownership, and
ACL evidence all match the final allowlist.

Final ownership separates administration from execution: the independent
admin owns the database, schema owner owns `np` and its relations/sequences,
and offline activation owns and alone executes the two c1 capabilities. Legacy
runtime, atomic runtime, readiness, and trainer receive only their exact table,
sequence, column, and function matrices. Each roles, migrations, demotion, and
catalog commit has a dedicated phase-specific readback. A failed commit is
accepted only when that readback proves that phase's durable target; otherwise
the phase is reported as commit-unknown. After the role and credential probes,
an exact terminal readback returns repeated `COMPLETE` without entering the
migration or catalog write paths.

This slice is intentionally dormant: the new module has no CLI, Compose or
Ansible hook, environment parsing, secret writer, session terminator, startup
consumer, activation call, or new runtime DDL path. The existing composed
runtime DDL path remains present and is an explicit later blocker. M9b.14c3
must provide the offline
orchestration, SCRAM credential provisioning and rotation, restrictive HBA and
network policy, real existing-volume rehearsal, an exclusive DDL/admin window,
and removal of DDL/migration authority from runtime services. M9b.14d must
compose the dedicated runtime
roles behind fail-closed startup and health checks. Bounded replay,
reconciliation/quarantine, side-effect-free shadow comparison, stale-writer
removal, tested pause/rollback, soak evidence, and explicit operator approval
remain cut-over blockers. No credential has been deployed and no production
volume has been migrated; `ACTIVE` remains an explicit **NO-GO**.

Reproducible M9b.14c2-c3a verification commands use an operator-supplied,
disposable PostgreSQL 15 admin DSN:

```bash
.venv/bin/python -m pytest -q tests/test_postgres_bootstrap.py
/usr/local/bin/python3.10 -m pytest -q tests/test_postgres_bootstrap.py
.venv/bin/python -m pytest -q \
  tests/test_postgres_bootstrap.py \
  tests/test_paper_account_readiness_repository.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/test_postgres_bootstrap.py \
  tests/test_paper_account_readiness_repository.py
export ELVIS_TEST_POSTGRES_ADMIN_DSN='<disposable PostgreSQL 15 admin DSN>'
export ELVIS_TEST_POSTGRES_REQUIRED=1
.venv/bin/python -m pytest -q \
  tests/postgres/test_postgres_bootstrap_postgres.py
/usr/local/bin/python3.10 -m pytest -q \
  tests/postgres/test_postgres_bootstrap_postgres.py
.venv/bin/python -m pytest -q tests/postgres
.venv/bin/python -m pytest -q --ignore=tests/postgres
git diff --check
```

The focused bootstrap suite passed 86 tests under each Python 3.14 and Python
3.10 interpreter. The adjacent bootstrap/readiness-catalog command passed 200
tests under each interpreter. The dedicated PostgreSQL 15 bootstrap suite
passed 110 tests under each interpreter. The complete PostgreSQL 15 suite
passed 463 tests. The full non-PostgreSQL run passed 2,561 tests, with 50
skipped, 280 warnings, and 7 subtests. Black targeting Python 3.10, isort,
flake8's fatal-error selectors, Python 3.10 compilation, relative Markdown-link
validation, and `git diff --check` were green on the final c3a tree.

### M9b.14c3b dormant offline PostgreSQL bootstrap CLI

M9b.14c3b exposes the c2-c3a library as one explicit operator command:

```text
python -m scripts.postgres_bootstrap \
  --config <bootstrap-v1.json> \
  --apply \
  --confirm-exclusive-ddl-role-window \
  [--confirm-old-runtime-demotion]
```

The CLI accepts an exact version-1 JSON object containing only the expected
database, independent admin role, exact seven-role manifest, service-name map,
and optional adoption manifest. `services.admin` is required,
`services.schema_owner` is always `null`, and each of the six login-role
services is either a libpq service identifier or `null`. Connection data and
secrets are forbidden from the JSON; libpq resolves the identifiers from the
operator-controlled `PGSERVICEFILE` and `PGPASSFILE` outside Git.

The two confirmation flags keep mutation behind explicit operator intent. The
exclusive-window confirmation does not turn the advisory lock into a
superuser fence. The old-runtime confirmation is required in addition to an
adoption manifest with `demote_old_shared_runtime: true`; it cannot silently
request demotion. The command has no plan mode and never retries itself.

Stdout contains one compact, secret-free JSON object. Terminal and resumable
receipts retain exactly `status`, `migration_versions`,
`verified_role_probes`, `pending_role_credentials`, and
`old_shared_runtime_demoted`. Their exits are `0` for `COMPLETE`, `10` for
`CREDENTIALS_REQUIRED`, and `11` for `DEMOTION_REQUIRED`. Typed errors expose
only `status: ERROR` plus `INPUT`, `STORAGE`, `DRIFT`, `MIGRATION`,
`COMMIT_UNKNOWN`, or `INTERNAL`, mapped to exits `2`, `20`, `21`, `22`, `23`,
and `70`. Only commit-unknown adds its `ROLES`, `MIGRATIONS`, `CATALOG`, or
`DEMOTION` phase. Exception and connection text is not serialized.

The [operator runbook](../V2_POSTGRES_BOOTSTRAP.md) records the fresh,
adoption, and demotion prerequisites; external secret boundary; two Mermaid
flows; status handling; and commit-unknown recovery. A repeated invocation is
always a new operator decision after evidence review. `COMPLETE` is not a
deployment receipt and cannot request activation.

This slice adds no credential generation or rotation, Compose/Ansible wiring,
HBA or network policy, session termination, startup hook, runtime credential
composition, runtime-DDL removal, health gate, or activation call. The V2
application target remains Python 3.14; Python 3.10 remains a temporary
compatibility floor and the isolated TensorFlow/ML trainer runtime. The
remaining deployment, replay, reconciliation, rollback, soak, and approval
gates still block `ACTIVE`.

The CLI contract suite passed 29 tests under Python 3.10 and 29 under Python
3.14. The adjacent bootstrap-library plus CLI unit suites passed 116 tests on
each interpreter. Two disposable PostgreSQL 15 CLI scenarios passed on each
interpreter: staged roles and exit `10`, external credential provisioning,
terminal exit `0`, repeat idempotence, and injected `CREATEDB` drift returning
exit `21` without repair. Post-test inspection found no residual test roles or
databases. Black under both interpreters, isort, fatal Flake8 selectors,
Python compilation, relative-link validation, both Mermaid renders, and
`git diff --check` were green. The broader Python 3.14 gates passed 465
PostgreSQL tests in 149.73 seconds and 2,591 non-PostgreSQL tests with 50
expected skips in 45.05 seconds. Pull-request CI remains the acceptance gate;
these results do not prove deployment or cut-over.

### M9b.14c3c1 isolated fresh PostgreSQL rehearsal

M9b.14c3c1 adds `deploy/v2/compose.bootstrap.yml`, a standalone rehearsal
project containing only pinned PostgreSQL 15 and the one-shot bootstrap image.
PostgreSQL has no host-published port, uses an internal fixed subnet and an
exact SCRAM-only HBA allowlist, and stores data only in a labelled rehearsal
volume guarded by an exact marker. Operator secrets and populated libpq files
remain outside Git and are mounted read-only.

The [rehearsal runbook](../V2_POSTGRES_REHEARSAL.md) records the trust boundary,
fresh state flow, existing-volume decision, rollback, and three Mermaid flows.
The stage manifest contains no login services and must return exit `10`; after
external parameterized credential provisioning, the complete manifest must
return exit `0` twice. The composition contains no bot, trainer, activation,
runtime startup hook, or active ELVIS volume.

The current shared-owner volume is explicitly outside c3c1. It requires a
stopped physical clone and a separate ownership-remediation or fresh-target
data-migration decision. M9b.14c3c2 selects the fresh-target branch through a
read-only preflight; it does not mount or copy the volume. The active root
Compose, Ansible, Apple scripts, runtime DDL, startup health, and `ACTIVE`
authority remain unchanged.

The frozen c3c1 contract suite passed 6 tests under Python 3.10 and 6 under
Python 3.14. The opt-in Docker/PostgreSQL 15 rehearsal passed 2 scenarios: the
nominal `10 -> 0 -> 0` operator flow and the non-mutating refusal of a non-empty
unmarked volume. It additionally proved six SCRAM verifiers, the configured
SCRAM encryption mode, an error-free HBA rule catalog, each role's own
credential, rejection of six crossed credentials and the non-allowlisted
database, an exact mode-0600 marker across restart, bounded expected refusal
logs, secret redaction, and zero residual Compose resources. Compose rendering,
shell syntax, Black, isort, fatal Flake8 selectors, compilation, relative-link
validation, the three Mermaid source/render sets, and `git diff --check` were
green. The dedicated GitHub Actions rehearsal remains the pull-request
acceptance gate; these results do not prove deployment or cut-over.

### M9b.14c3c2 stopped-clone/fresh-target preflight

M9b.14c3c2 chooses a fresh-target migration path rather than repairing the
legacy shared-superuser ownership boundary in place. It adds the dormant
`trading.application.fresh_target_cutover` contract, the read-only
`trading.persistence.postgres_cutover_preflight` adapter, and the one-shot
`scripts.postgres_cutover_preflight` CLI. The source must be a verified physical
clone captured after all source writers stop; the target must be a separate,
freshly bootstrapped terminal V2 database. The active source volume is never an
input.

The CLI requires `--config`, `--inspect`,
`--confirm-stopped-source-clone`, and
`--confirm-exclusive-database-window`. Both confirmations are operator
assertions. The command cannot establish external quiescence or prevent a
non-cooperating database administrator from changing evidence. It has no apply,
repair, copy, reconcile, bootstrap, session-termination, or activation mode.

Admission requires different PostgreSQL cluster system identifiers and an
exact V1 `0001` `np` import surface: seven tables, seven owned sequences, ten
canonical indexes, and the declared shared owner. It admits no migration
ledger, V2 table, routine, type, user trigger, surplus ACL, default ACL, or
other `np` object. Unrelated
source schemas are outside this inventory; the later importer remains
allowlisted to these seven schema-qualified relations. Admission also requires
no other source session or open position, semantically valid source
rows, the exact terminal target catalog, the target's `LEGACY` generation-zero
control row, and an empty target import boundary. Source relation evidence
contains the name, row count, primary-key minimum and maximum, and SHA-256
accumulated over a stable ordered stream of typed rows. System identifiers are
serialized as decimal strings. The adapter keeps memory bounded across fetch
batches and runs only read-only inspection SQL.

Target emptiness is a row-level boundary. It deliberately does not certify the
current runtime values of the seven legacy serial sequences; an insert/delete
cycle can advance them without leaving a row. The future importer must
validate and normalize every target sequence and include that state in its
post-import parity receipt.

Exit `0` returns `READY_FOR_FRESH_TARGET`; exit `21` returns `BLOCKED` with only
the exact applicable codes `SOURCE_IDENTITY`, `SOURCE_ACTIVE_SESSIONS`,
`SOURCE_SCHEMA`, `SOURCE_OPEN_POSITIONS`, `SOURCE_DATA_QUALITY`, `SAME_CLUSTER`,
`TARGET_NOT_COMPLETE`, `TARGET_MODE`, and `TARGET_NOT_EMPTY`. Exits `2`, `20`,
and `70` return compact input, storage, and internal errors. Receipts exclude
service and role identifiers, connection data, SQL, exception messages, and
arbitrary error text.

Every receipt says `stale_on_return: true` and
`snapshot_authoritative: false`. Success neither reserves the pair nor proves
a later import. The
[fresh-target runbook](../V2_FRESH_TARGET_CUTOVER.md) records the exact command,
configuration, trust boundary, no-copy decision, phased rollback, next bounded
importer slice, and two Mermaid source/SVG/PNG/Excalidraw graph sets.

The frozen local snapshot passed 29 focused contract/startup-guard tests under
both Python 3.10 and 3.14. The dedicated two-cluster PostgreSQL 15 suite passed
3 scenarios. The complete PostgreSQL suite passed 470 tests; the complete
non-PostgreSQL suite passed 2,641 tests with 57 expected skips and 7 subtests.
Black, isort, fatal Flake8, compilation, link, diagram-artifact, diff, and
Docker-cleanup checks were green.

This evidence covers exact ready and blocked outcomes, stable hashing across
fetch boundaries, read-only SQL, redaction, and zero source/target mutation.
Pull-request CI remains the remote acceptance gate. This slice does not deploy
or copy data, and `ACTIVE` remains a **NO-GO**.

### M9b.14c3c3a bounded legacy snapshot import

M9b.14c3c3a adds the dormant pure contract in
`trading.application.legacy_snapshot_import`, the offline PostgreSQL adapter in
`trading.persistence.postgres_legacy_snapshot_import`, and the one-shot
`scripts.postgres_legacy_snapshot_import` CLI. The application boundary is
exactly `LegacySnapshotImportContext(cutover_context, batch_size=512)`, the
`IMPORTED | REPLAYED` disposition, per-relation and aggregate receipts, and the
positional-only `import_snapshot(context, preflight_receipt)` port. Every
receipt remains stale on return, snapshot-non-authoritative, target-exact, and
incapable of authorizing runtime activation.

The CLI accepts the exact secret-free c3c2 `READY_FOR_FRESH_TARGET` JSON only
as a strict document with a 65,536-byte maximum. Operator hygiene keeps config
and receipt files owner-controlled and non-world-writable, but ownership and
exact-mode policy are not CLI guarantees; the CLI does reject symlinks and
non-regular paths. It binds every source fingerprint, canonical hash, and pair
system identifier as stale expected evidence, then revalidates source and
target under its own transactions. This permits exact prior-copy classification
as `REPLAYED`, which a fresh c3c2 empty-target preflight could not express. The
strict version-1 intent contains only the source service and expected identity,
the target admin and migrator services, and the same fresh-target bootstrap
context used by c3c2. Top-level `batch_size` is the only configurable cap and
must be 1 through 512. Connection details and credentials remain in external
libpq files.

The data allowlist is the exact seven V1 relations:
`np.account_balances`, `np.liquidations`, `np.margin_history`,
`np.model_predictions`, `np.open_positions`, `np.trades`, and
`np.trading_session_resets`. `np.open_positions` must remain empty. The other
rows are copied 1:1 with explicit primary keys and intentional gaps through
bounded batches inside one target row transaction. The application batch size
is 1 through 512. Compiled, non-configurable limits admit at most 100,000 rows,
65,536 canonical bytes per row, and 512 MiB of canonical source bytes. The
source uses a repeatable-read, read-only snapshot, rehashes the stable typed row
stream before insertion, and never spools business rows to disk.

The adapter uses three pairwise-distinct connection factories: stopped source,
target admin, and target migrator. It reinspects the target database and
migrator identity before `SET ROLE` to the schema owner, takes the fixed target
locks, and revalidates the terminal catalog, `LEGACY/0` mode, and exact
empty-or-prior-copy state before mutation. Partial, surplus, or foreign data is
conflict. It never uses DDL, `GRANT`, `REVOKE`, `DELETE`, `TRUNCATE`, upsert,
trigger disabling, role administration, migration-ledger mutation, or a V2
journal/ledger synthesis path.

All raw rows commit atomically. A known pre-commit failure rolls back. If commit
acknowledgement is lost, exact target readback alone classifies the result:
exact committed rows continue to post-commit recovery, an empty target remains
`COMMIT_UNKNOWN`, and a partial or changed target is `CONFLICT`. An exact prior
copy returns `REPLAYED` after completing the same recovery checks instead of
inserting again.

Serial-sequence normalization is deliberately after the row commit because
PostgreSQL `setval` state is not transactional. Each target next value is the
safe maximum of the observed source next value and one past the imported
primary-key maximum, with one as the empty-table floor. Every normalized
sequence and every row fingerprint is reread before `IMPORTED` or `REPLAYED`
can be returned.

The exact command is:

```bash
python -m scripts.postgres_legacy_snapshot_import \
  --config <legacy-snapshot-import-v1.json> \
  --preflight-receipt <cutover-preflight-ready.json> \
  --import-snapshot \
  --confirm-stopped-source-clone \
  --confirm-exclusive-database-window \
  --confirm-disposable-target
```

Exit `0` returns `IMPORTED` or `REPLAYED`. Exits `2`, `20`, `22`, `23`, `24`,
and `70` return only `INPUT`, `STORAGE`, `BUSY`, `CONFLICT`, `COMMIT_UNKNOWN`,
and `INTERNAL`, respectively. There is no automatic retry. The complete data,
receipt, recovery, and rollback contract is in the
[legacy snapshot import runbook](../V2_LEGACY_SNAPSHOT_IMPORT.md), with two
Mermaid/SVG/PNG/editable-Excalidraw graph sets.

This slice deliberately does not synthesize `np.orders`, `np.order_events`,
position streams, paper-account streams or ledgers, runtime epochs, or opening
provenance from V1 rows. The legacy schema lacks the immutable causal facts
needed to do so safely. Replay and semantic reconciliation of imported history,
runtime DDL removal, dedicated production composition, fail-closed health,
side-effect-free shadow comparison, stale-writer removal, rollback rehearsal,
soak, and explicit operator approval remain later gates. `ACTIVE` remains a
**NO-GO**.

The frozen c3c3a contract, adjacent cut-over, migration-runner, and deployment
guard command passed 111 tests under both Python 3.10 and Python 3.14. The
dedicated three-cluster-capable PostgreSQL 15 importer file passed 9 scenarios,
including exact import and replay, cross-cluster miswiring, commit-unknown
recovery, target drift before row copy and before sequence normalization,
primary-key exhaustion, and a write committed exactly between target identity
inspection and lock acquisition. The complete PostgreSQL suite, with the
bootstrap, fresh-rehearsal, cut-over-preflight, and snapshot-import opt-ins,
passed 479 tests. The final-byte non-PostgreSQL suite passed 2,667 tests with
50 expected skips and 7 subtests. It emitted non-gating warnings whose count
varied across identical-byte runs, so no false deterministic total is claimed.

Black, isort, fatal-error Flake8, Python 3.10 and Python 3.14 compilation,
JSON/YAML parsing, 100 relative Markdown links, exact Mermaid fence/source
parity, both SVG/PNG/Excalidraw render triplets, visual PNG inspection, and
`git diff --check` were green. The official PostgreSQL 15 disposable fixture
and all nested V2 rehearsal, preflight, and importer resources left zero
containers, volumes, or networks after the run. Byte-for-byte file hashes,
worktree status, and `HEAD` were unchanged across both global suites.

### M9b.14c3c3b read-only legacy snapshot reconciliation

M9b.14c3c3b adds a dormant, read-only review after the bounded c3c3a import.
It hashes the exact parsed configuration and import-receipt documents, checks
the receipt's combined seven-relation hash for internal consistency, then
sequentially revalidates the same target's cluster identity, terminal catalog,
`LEGACY/0` control, raw relation fingerprints, and sequence parity through
distinct readiness and admin identities. It opens no source connection and
performs no DML, DDL, sequence change, `SET ROLE`, account opening,
provisioning, or activation.

The review preserves two candidate interpretations rather than manufacturing
history. The imported candidate retains the complete canonical
`np.account_balances` tuple, including every zero or non-zero additional asset.
`OPERATOR_EQUITY_HYPOTHESIS` uses an explicit starting-collateral assumption and
the imported target only. It reconstructs exact float4 PnL and fee inputs,
folds each sequence deterministically in primary-key order with binary64
addition, and applies `max(0, starting hypothesis + folded PnL)`. The resulting
USDT-specific tuple is `BNB=0`, `BTC=0`, and `USDT=hypothesised equity`.
Trade and liquidation fees remain separate and are never deducted.

This is not PostgreSQL `SUM`, source replay, or evidence of the active runtime's
starting capital, ordering, algorithm, or state. The latest-reset window
includes equal timestamps; without a reset it includes every row. A canonical
naive microsecond timestamp and each exact binary64 result are retained. Equal
candidate documents merely omit `CANDIDATE_MISMATCH`; they cannot establish
runtime provenance.

The only dispositions are `DECISION_REQUIRED` and `BLOCKED`.
`DECISION_REQUIRED` always includes `RUNTIME_PROVENANCE_UNPROVEN`, whether the
canonical balance documents and SHA-256 values agree or differ. `BLOCKED`
exposes neither partial opening nor partial numeric evidence. Every receipt
hard-codes stale true and snapshot authority, coherent snapshot, source
provenance, target-observation authentication, enforced database window,
account opening, account provisioning, and runtime activation false. The
adapter derives the three hypothesis folds from target reads, but the typed
receipt does not itself authenticate them.

The pure contract is in
`trading.application.legacy_snapshot_reconciliation`: the explicit
`LegacySnapshotReconciliationContext`, canonical config and receipt SHA-256
bindings, ordered candidates, pure opening/hypothesis derivation helpers,
bounded findings, evidence, two-value disposition, and permanently
non-authoritative receipt. The only port operation is
`LegacySnapshotReconciliationPort.reconcile(context, import_receipt)`.

The target session check is a point-in-time query. Terminal, admin, and
readiness evidence spans separate repeatable-read transactions, so c3c3b never
claims one coherent observation snapshot or an enforced window. Canonical
document hashes and the combined source relation-evidence hash bind the supplied
documents and target readback but do not authenticate their author or the
declared source. The receipt exposes the declared import disposition and source
system identifier under that limitation.

The one-shot command is:

```bash
python -m scripts.postgres_legacy_snapshot_reconciliation \
  --config <legacy-snapshot-reconciliation-v1.json> \
  --import-receipt <legacy-snapshot-import.json> \
  --assess \
  --confirm-reviewed-database-window \
  --confirm-disposable-target
```

Its closed version-1 JSON contains the declared c3c3a source and target
bootstrap intent, distinct admin/readiness libpq services, explicit execution
scope, account, generation, collateral and margin quantum, plus
`hypothesis_starting_collateral_decimal`. The committed example uses `100`.
The reviewed-window flag is an operator assertion, not an exclusive lock or
fence. Exit `10` is `DECISION_REQUIRED` and `21` is `BLOCKED`; `2`, `20`, `23`,
and `70` are typed `INPUT`, `STORAGE`, `CONFLICT`, and `INTERNAL` errors. The
listed values are the complete exit contract; equality never becomes an
ordinary-success result.

On the accepted c3c3b snapshot, the focused application, CLI, and
migration-runner selection passed 74 tests under Python 3.10 and 74 under
Python 3.14. The dedicated opt-in PostgreSQL 15 reconciliation file passed all
19 scenarios in 99.48 seconds. Black, isort, Flake8 with the repository's
88-character limit, Python 3.10/3.14 compilation, and `git diff --check` were
green. On the final frozen slice, the complete PostgreSQL 15 gate with all five
V2 opt-ins passed 500 tests, including deterministic retry canaries for
transient Docker port-publication and host-connection delays. The complete
non-PostgreSQL gate passed
2,706 tests, skipped 50, and passed 7 subtests. Both gates preserved the exact
935-file inventory byte for byte, and the disposable PostgreSQL resources left
zero containers, volumes, or networks. The complete operator contract and two
Mermaid/SVG/PNG/editable-Excalidraw graph sets are in the [legacy snapshot
reconciliation runbook](../V2_LEGACY_SNAPSHOT_RECONCILIATION.md).

This slice structures the historical ambiguity but deliberately does not
resolve it. A separate reviewed operation must authenticate source/runtime
provenance, choose and encode the durable opening, provision the V2 account,
and prove replay before later shadow, rollback, soak, and cut-over gates.
Python 3.14 remains the V2 application runtime target; Python 3.10 remains a
temporary compatibility and isolated trainer requirement. `ACTIVE` remains a
**NO-GO**.

## Cut-over policy

The paper-runtime control has four modes:

1. **legacy** — current implementation is authoritative;
2. **shadow** — both implementations evaluate the same frozen input, only the
   legacy path may act, and differences are recorded; and
3. **paused** — legacy writes are fenced and no new owner may act while a
   transition, reconciliation, or rollback is unresolved; and
4. **active** — the new durable owner alone may act and the database rejects
   writes to every legacy paper table.

Shadow mode must never call a second executor or mutate cooldown, positions,
portfolio, model feedback, or persistence. A comparison is useful only if it is
side-effect free. Rollback from `ACTIVE` must first enter `PAUSED`, wait out and
reconcile the active owner, and then select one authority deliberately; it must
never create a window in which legacy and durable writers can both commit.

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
