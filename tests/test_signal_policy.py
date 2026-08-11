import ast
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from trading.application.signal_policy import (
    SignalPolicyPipeline,
    SignalPolicyResult,
)
from trading.domain.signals import Signal, SignalAction


def make_signal(**overrides: object) -> Signal:
    values = {
        "decision_id": "decision-1",
        "symbol": "BTCUSDT",
        "action": SignalAction.BUY,
        "confidence": 0.8,
        "reference_price": 123456.75,
        "observed_at": datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
        "strategy_id": "ensemble",
        "reasons": ("strategy.consensus",),
    }
    values.update(overrides)
    return Signal(**values)


class StubPolicy:
    def __init__(
        self,
        policy_id: str,
        callback: Callable[[Signal], object],
    ) -> None:
        self.policy_id = policy_id
        self.callback = callback
        self.calls: list[Signal] = []

    def evaluate(self, signal: Signal, /) -> SignalPolicyResult:
        self.calls.append(signal)
        return self.callback(signal)  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("veto", "confidence", "reasons"),
    [
        (False, None, ()),
        (False, None, ("observed",)),
        (False, 0.0, ("lowered",)),
        (False, 1.0, ("raised",)),
        (True, None, ("blocked",)),
    ],
)
def test_policy_result_accepts_only_unambiguous_states(
    veto: bool,
    confidence: float | None,
    reasons: tuple[str, ...],
) -> None:
    result = SignalPolicyResult(
        veto=veto,
        confidence=confidence,
        reasons=reasons,
    )

    assert result.veto is veto
    assert result.confidence == confidence
    assert result.reasons == reasons
    assert isinstance(hash(result), int)


@pytest.mark.parametrize("veto", [0, 1, "false", None, object()])
def test_policy_result_rejects_untyped_veto(veto: object) -> None:
    with pytest.raises(TypeError):
        SignalPolicyResult(veto=veto)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("veto", "confidence", "reasons"),
    [
        (False, float("nan"), ("adjust",)),
        (False, float("inf"), ("adjust",)),
        (False, -0.1, ("adjust",)),
        (False, 1.1, ("adjust",)),
        (False, True, ("adjust",)),
        (False, 0.5, ()),
        (True, 0.0, ("veto",)),
        (True, None, ()),
    ],
)
def test_policy_result_rejects_invalid_payloads(
    veto: bool,
    confidence: object,
    reasons: tuple[str, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SignalPolicyResult(
            veto=veto,
            confidence=confidence,  # type: ignore[arg-type]
            reasons=reasons,
        )


@pytest.mark.parametrize("reasons", [["mutable"], ("",), (" padded ",), (1,)])
def test_policy_result_rejects_invalid_reasons(reasons: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SignalPolicyResult(
            veto=True,
            reasons=reasons,  # type: ignore[arg-type]
        )


def test_policy_result_is_frozen_and_cannot_express_a_side_change() -> None:
    result = SignalPolicyResult()

    with pytest.raises(FrozenInstanceError):
        result.veto = True  # type: ignore[misc]

    field_names = {field.name for field in fields(SignalPolicyResult)}
    assert "action" not in field_names
    assert "signal" not in field_names


def test_pipeline_rejects_mutable_or_invalid_policy_collections() -> None:
    valid = StubPolicy("valid", lambda _signal: SignalPolicyResult())

    with pytest.raises(TypeError):
        SignalPolicyPipeline([valid])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SignalPolicyPipeline((object(),))  # type: ignore[arg-type]


@pytest.mark.parametrize("policy_id", ["", " padded ", "contains:colon", 1, None])
def test_pipeline_rejects_invalid_policy_ids(policy_id: object) -> None:
    policy = StubPolicy(  # type: ignore[arg-type]
        policy_id,
        lambda _signal: SignalPolicyResult(),
    )

    with pytest.raises((TypeError, ValueError)):
        SignalPolicyPipeline((policy,))


def test_pipeline_rejects_duplicate_policy_ids() -> None:
    first = StubPolicy("rsi", lambda _signal: SignalPolicyResult())
    duplicate = StubPolicy("rsi", lambda _signal: SignalPolicyResult())

    with pytest.raises(ValueError, match="unique"):
        SignalPolicyPipeline((first, duplicate))


def test_pipeline_snapshots_policy_ids_at_composition_time() -> None:
    policy = StubPolicy(
        "stable-id",
        lambda _signal: SignalPolicyResult(reasons=("observed",)),
    )
    pipeline = SignalPolicyPipeline((policy,))

    policy.policy_id = "mutated-id"
    result = pipeline.evaluate(make_signal())

    assert pipeline.policy_ids == ("stable-id",)
    assert result.reasons[-1] == "policy:stable-id:observed"


def test_pass_policies_run_once_in_fixed_order() -> None:
    order: list[str] = []
    first = StubPolicy(
        "first",
        lambda _signal: (
            order.append("first") or SignalPolicyResult(reasons=("pass",))
        ),
    )
    second = StubPolicy(
        "second",
        lambda _signal: (
            order.append("second") or SignalPolicyResult(reasons=("pass",))
        ),
    )

    result = SignalPolicyPipeline((first, second)).evaluate(make_signal())

    assert order == ["first", "second"]
    assert len(first.calls) == len(second.calls) == 1
    assert result.reasons == (
        "strategy.consensus",
        "policy:first:pass",
        "policy:second:pass",
    )


def test_noop_pass_preserves_signal_identity() -> None:
    signal = make_signal()
    policy = StubPolicy("pass", lambda _signal: SignalPolicyResult())

    result = SignalPolicyPipeline((policy,)).evaluate(signal)

    assert result is signal


def test_adjusted_confidence_is_visible_to_the_next_policy() -> None:
    observed: list[Signal] = []
    adjust = StubPolicy(
        "liquidity",
        lambda _signal: SignalPolicyResult(
            confidence=0.55,
            reasons=("reduced",),
        ),
    )
    observe = StubPolicy(
        "observe",
        lambda signal: (
            observed.append(signal) or SignalPolicyResult(reasons=("pass",))
        ),
    )

    result = SignalPolicyPipeline((adjust, observe)).evaluate(make_signal())

    assert observed[0].confidence == 0.55
    assert observed[0].reasons[-1] == "policy:liquidity:reduced"
    assert result.confidence == 0.55
    assert result.action is SignalAction.BUY


def test_veto_makes_the_signal_non_actionable_and_short_circuits() -> None:
    veto = StubPolicy(
        "stale-market",
        lambda _signal: SignalPolicyResult(veto=True, reasons=("stale",)),
    )
    forbidden = StubPolicy("must-not-run", lambda _signal: SignalPolicyResult())
    original = make_signal(action=SignalAction.SELL)

    result = SignalPolicyPipeline((veto, forbidden)).evaluate(original)

    assert result.action is SignalAction.HOLD
    assert result.confidence == 0.0
    assert result.reasons == ("strategy.consensus", "policy:stale-market:stale")
    assert len(veto.calls) == 1
    assert forbidden.calls == []


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (TimeoutError("secret endpoint"), "policy:external-check:timeout"),
        (RuntimeError("secret database details"), "policy:external-check:exception"),
    ],
)
def test_policy_failures_veto_without_leaking_exception_details(
    failure: Exception,
    reason: str,
) -> None:
    def raise_failure(_signal: Signal) -> SignalPolicyResult:
        raise failure

    policy = StubPolicy("external-check", raise_failure)

    result = SignalPolicyPipeline((policy,)).evaluate(make_signal())

    assert result.action is SignalAction.HOLD
    assert result.confidence == 0.0
    assert result.reasons[-1] == reason
    assert str(failure) not in repr(result)


def test_policy_failure_preserves_prior_reasons_and_stops_the_pipeline() -> None:
    adjust = StubPolicy(
        "adjust",
        lambda _signal: SignalPolicyResult(
            confidence=0.6,
            reasons=("reduced",),
        ),
    )

    def fail(_signal: Signal) -> SignalPolicyResult:
        raise RuntimeError("private details")

    failing = StubPolicy("failing", fail)
    forbidden = StubPolicy("forbidden", lambda _signal: SignalPolicyResult())

    result = SignalPolicyPipeline((adjust, failing, forbidden)).evaluate(make_signal())

    assert result.reasons == (
        "strategy.consensus",
        "policy:adjust:reduced",
        "policy:failing:exception",
    )
    assert failing.calls[0].confidence == 0.6
    assert forbidden.calls == []


@pytest.mark.parametrize("malformed", [None, True, {}, "PASS", object()])
def test_malformed_policy_results_fail_closed(malformed: object) -> None:
    policy = StubPolicy("malformed", lambda _signal: malformed)

    result = SignalPolicyPipeline((policy,)).evaluate(make_signal())

    assert result.action is SignalAction.HOLD
    assert result.confidence == 0.0
    assert result.reasons[-1] == "policy:malformed:invalid-result"


def test_result_that_bypassed_dataclass_validation_fails_closed() -> None:
    malformed = object.__new__(SignalPolicyResult)
    object.__setattr__(malformed, "veto", False)
    object.__setattr__(malformed, "confidence", float("nan"))
    object.__setattr__(malformed, "reasons", ("adjust",))
    policy = StubPolicy("malformed", lambda _signal: malformed)

    result = SignalPolicyPipeline((policy,)).evaluate(make_signal())

    assert result.action is SignalAction.HOLD
    assert result.confidence == 0.0
    assert result.reasons[-1] == "policy:malformed:invalid-result"


def test_hold_short_circuits_without_invoking_policies() -> None:
    policy = StubPolicy("forbidden", lambda _signal: SignalPolicyResult())
    signal = make_signal(action=SignalAction.HOLD, confidence=0.4)

    result = SignalPolicyPipeline((policy,)).evaluate(signal)

    assert result is signal
    assert policy.calls == []


def test_empty_pipeline_returns_the_same_signal() -> None:
    signal = make_signal()

    assert SignalPolicyPipeline().evaluate(signal) is signal


def test_pipeline_rejects_an_untyped_signal_without_policy_calls() -> None:
    policy = StubPolicy("forbidden", lambda _signal: SignalPolicyResult())

    with pytest.raises(TypeError):
        SignalPolicyPipeline((policy,)).evaluate(  # type: ignore[arg-type]
            {"action": "BUY"}
        )

    assert policy.calls == []


def test_pipeline_preserves_signal_identity_fields() -> None:
    signal = make_signal(action=SignalAction.SELL)
    policy = StubPolicy(
        "confidence",
        lambda _signal: SignalPolicyResult(
            confidence=0.4,
            reasons=("cut",),
        ),
    )

    result = SignalPolicyPipeline((policy,)).evaluate(signal)

    assert result.decision_id == signal.decision_id
    assert result.symbol == signal.symbol
    assert result.action is signal.action
    assert result.reference_price == signal.reference_price
    assert result.observed_at is signal.observed_at
    assert result.strategy_id == signal.strategy_id


def test_keyboard_interrupt_is_not_swallowed() -> None:
    def interrupt(_signal: Signal) -> SignalPolicyResult:
        raise KeyboardInterrupt

    policy = StubPolicy("interrupt", interrupt)

    with pytest.raises(KeyboardInterrupt):
        SignalPolicyPipeline((policy,)).evaluate(make_signal())


def test_signal_policy_module_has_no_runtime_or_io_imports() -> None:
    source_path = Path("trading/application/signal_policy.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    allowed_roots = {"dataclasses", "typing", "trading"}
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert imported_roots <= allowed_roots
