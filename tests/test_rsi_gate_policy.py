import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading.application.rsi_gate_policy import RsiGatePolicy
from trading.application.signal_policy import SignalPolicyPipeline
from trading.domain.signals import Signal, SignalAction
from trading.signals.filters import rsi_gate as legacy_rsi_gate


def make_signal(action: SignalAction, confidence: float = 0.8) -> Signal:
    return Signal(
        decision_id="decision-rsi-1",
        symbol="BTCUSDT",
        action=action,
        confidence=confidence,
        reference_price=123456.75,
        observed_at=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
        strategy_id="ensemble",
        reasons=("strategy.consensus",),
    )


@pytest.mark.parametrize(
    ("action", "rsi"),
    [
        (SignalAction.BUY, 0.0),
        (SignalAction.BUY, 29.9),
        (SignalAction.BUY, 30.0),
        (SignalAction.BUY, 50.0),
        (SignalAction.BUY, 70.0),
        (SignalAction.BUY, 70.1),
        (SignalAction.BUY, 100.0),
        (SignalAction.SELL, 0.0),
        (SignalAction.SELL, 29.9),
        (SignalAction.SELL, 30.0),
        (SignalAction.SELL, 50.0),
        (SignalAction.SELL, 70.0),
        (SignalAction.SELL, 70.1),
        (SignalAction.SELL, 100.0),
    ],
)
def test_policy_matches_legacy_for_every_valid_boundary(
    action: SignalAction,
    rsi: float,
) -> None:
    signal = make_signal(action)
    legacy_action, _ = legacy_rsi_gate(action.value, rsi)

    candidate = SignalPolicyPipeline((RsiGatePolicy(rsi),)).evaluate(signal)

    assert candidate.action.value == legacy_action
    assert candidate.confidence == (0.0 if legacy_action == "HOLD" else 0.8)


@pytest.mark.parametrize(
    ("action", "rsi", "reason"),
    [
        (SignalAction.BUY, 70.0001, "policy:rsi-gate:overbought"),
        (SignalAction.SELL, 29.9999, "policy:rsi-gate:oversold"),
    ],
)
def test_policy_vetoes_only_the_stretched_side(
    action: SignalAction,
    rsi: float,
    reason: str,
) -> None:
    candidate = SignalPolicyPipeline((RsiGatePolicy(rsi),)).evaluate(
        make_signal(action)
    )

    assert candidate.action is SignalAction.HOLD
    assert candidate.confidence == 0.0
    assert candidate.reasons[-1] == reason


@pytest.mark.parametrize(
    ("rsi", "reason"),
    [
        (None, "unavailable"),
        (float("nan"), "unavailable"),
        (float("inf"), "unavailable"),
        (float("-inf"), "unavailable"),
        (True, "unavailable"),
        ("75", "unavailable"),
        (-0.1, "out-of-range"),
        (100.1, "out-of-range"),
    ],
)
def test_missing_nonfinite_or_out_of_range_rsi_fails_closed(
    rsi: object,
    reason: str,
) -> None:
    candidate = SignalPolicyPipeline((RsiGatePolicy(rsi),)).evaluate(
        make_signal(SignalAction.BUY)
    )

    assert candidate.action is SignalAction.HOLD
    assert candidate.confidence == 0.0
    assert candidate.reasons[-1] == f"policy:rsi-gate:{reason}"


def test_custom_thresholds_use_strict_comparisons() -> None:
    pipeline = SignalPolicyPipeline(
        (RsiGatePolicy(60.0, overbought=60.0, oversold=40.0),)
    )

    assert pipeline.evaluate(make_signal(SignalAction.BUY)).action is SignalAction.BUY
    sell = SignalPolicyPipeline(
        (RsiGatePolicy(39.9, overbought=60.0, oversold=40.0),)
    ).evaluate(make_signal(SignalAction.SELL))
    assert sell.action is SignalAction.HOLD


@pytest.mark.parametrize(
    ("overbought", "oversold"),
    [
        (float("nan"), 30.0),
        (70.0, float("inf")),
        (True, 30.0),
        (70.0, False),
        (-0.1, 30.0),
        (100.1, 30.0),
        (70.0, -0.1),
        (70.0, 100.1),
        (30.0, 30.0),
        (20.0, 30.0),
    ],
)
def test_policy_rejects_invalid_threshold_configuration(
    overbought: object,
    oversold: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RsiGatePolicy(  # type: ignore[arg-type]
            50.0,
            overbought=overbought,
            oversold=oversold,
        )


def test_policy_is_immutable_and_has_a_stable_id() -> None:
    policy = RsiGatePolicy(50)

    assert policy.policy_id == "rsi-gate"
    assert policy.rsi == 50.0
    with pytest.raises(FrozenInstanceError):
        policy.rsi = 80.0  # type: ignore[misc]


def test_policy_module_has_only_standard_library_and_domain_imports() -> None:
    source_path = Path("trading/application/rsi_gate_policy.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    allowed_roots = {"dataclasses", "trading"}
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
