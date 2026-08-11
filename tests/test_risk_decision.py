from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trading.domain import RiskDecision
from trading.domain.orders import OrderIntent, OrderSide, OrderType

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def make_intent(**overrides: object) -> OrderIntent:
    values = {
        "client_order_id": "elvis-order-1",
        "decision_id": "decision-1",
        "symbol": "BTCUSDT",
        "side": OrderSide.BUY,
        "quantity": Decimal("0.001"),
        "order_type": OrderType.MARKET,
        "reference_price": Decimal("123456.75"),
        "leverage": 3,
        "created_at": NOW,
    }
    values.update(overrides)
    return OrderIntent(**values)


def make_decision(**overrides: object) -> RiskDecision:
    values = {
        "decision_id": "decision-1",
        "approved": True,
        "reasons": (),
        "order_intent": make_intent(),
    }
    values.update(overrides)
    return RiskDecision(**values)


def test_approved_risk_decision_carries_one_correlated_intent() -> None:
    decision = make_decision(reasons=("within exposure limit",))

    assert decision.approved is True
    assert decision.order_intent == make_intent()


def test_rejected_risk_decision_is_non_actionable() -> None:
    decision = make_decision(
        approved=False,
        reasons=("cooldown.active",),
        order_intent=None,
    )

    assert decision.approved is False
    assert decision.order_intent is None


def test_risk_decision_is_frozen_and_hashable() -> None:
    decision = make_decision()

    assert isinstance(hash(decision), int)
    with pytest.raises(FrozenInstanceError):
        decision.approved = False  # type: ignore[misc]


@pytest.mark.parametrize("decision_id", ["", "   ", " padded "])
def test_risk_decision_rejects_invalid_identifier(decision_id: str) -> None:
    with pytest.raises(ValueError):
        make_decision(decision_id=decision_id)


@pytest.mark.parametrize("decision_id", [None, 1])
def test_risk_decision_requires_a_string_identifier(decision_id: object) -> None:
    with pytest.raises(TypeError):
        make_decision(decision_id=decision_id)


@pytest.mark.parametrize("approved", [0, 1, "true", None])
def test_risk_decision_requires_a_strict_boolean(approved: object) -> None:
    with pytest.raises(TypeError):
        make_decision(approved=approved)


@pytest.mark.parametrize(
    "reasons",
    [["mutable"], ("",), (" padded ",), (1,), (None,)],
)
def test_risk_decision_rejects_invalid_reasons(reasons: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_decision(reasons=reasons)


def test_approved_risk_decision_requires_an_intent() -> None:
    with pytest.raises(ValueError):
        make_decision(order_intent=None)


def test_approved_risk_decision_rejects_wrong_intent_type() -> None:
    with pytest.raises(TypeError):
        make_decision(order_intent="order")


def test_approved_risk_decision_requires_matching_decision_id() -> None:
    with pytest.raises(ValueError):
        make_decision(order_intent=make_intent(decision_id="other-decision"))


def test_rejected_risk_decision_requires_a_reason() -> None:
    with pytest.raises(ValueError):
        make_decision(
            approved=False,
            reasons=(),
            order_intent=None,
        )


def test_rejected_risk_decision_forbids_an_intent() -> None:
    with pytest.raises(ValueError):
        make_decision(
            approved=False,
            reasons=("risk.limit",),
            order_intent=make_intent(),
        )
