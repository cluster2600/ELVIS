from datetime import timezone
from decimal import Decimal
from inspect import getsource
from unittest.mock import Mock, patch

import pytest

import main as main_module
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)


def make_intent() -> OrderIntent:
    return main_module._legacy_order_intent(
        symbol="BTCUSDT",
        signal="BUY",
        confidence=0.8,
        current_price=123_456.75,
        position_size=0.001,
        leverage=3,
        strategy_id="EnsembleStrategy",
    )


def make_report(status: SubmissionStatus) -> SubmissionReport:
    if status is SubmissionStatus.SUBMITTED:
        return SubmissionReport(
            client_order_id="ELV-0123456789abcdef0123456789abcdef",
            status=status,
            retry_safety=RetrySafety.UNSAFE,
            venue_order_id="MOCK_BTCUSDT_1",
            venue_status="FILLED",
        )
    return SubmissionReport(
        client_order_id="ELV-0123456789abcdef0123456789abcdef",
        status=status,
        retry_safety=(
            RetrySafety.UNSAFE
            if status is SubmissionStatus.AMBIGUOUS
            else RetrySafety.SAFE
        ),
        reason="not acknowledged",
        venue_status=(
            "REJECTED" if status is SubmissionStatus.VENUE_REJECTED else None
        ),
    )


def test_live_mode_is_rejected_before_bootstrap() -> None:
    with patch.object(main_module, "bootstrap_application") as bootstrap:
        with pytest.raises(RuntimeError, match="paper trading"):
            main_module.main(mode="live", log_level="INFO")

    bootstrap.assert_not_called()


def test_legacy_order_intent_is_exact_and_correlated() -> None:
    intent = make_intent()

    assert intent.client_order_id == f"ELV-{intent.decision_id}"
    assert len(intent.client_order_id) == 36
    assert intent.side is OrderSide.BUY
    assert intent.quantity == Decimal("0.001")
    assert intent.reference_price == Decimal("123456.75")
    assert intent.created_at.utcoffset() == timezone.utc.utcoffset(intent.created_at)


def test_hold_cannot_create_a_legacy_order_intent() -> None:
    with pytest.raises(ValueError):
        main_module._legacy_order_intent(
            symbol="BTCUSDT",
            signal="HOLD",
            confidence=0.8,
            current_price=123_456.75,
            position_size=0.001,
            leverage=3,
            strategy_id="EnsembleStrategy",
        )


def test_acknowledged_submission_records_votes_and_cooldown_once() -> None:
    intent = make_intent()
    report = SubmissionReport(
        client_order_id=intent.client_order_id,
        status=SubmissionStatus.SUBMITTED,
        retry_safety=RetrySafety.UNSAFE,
        venue_order_id="MOCK_BTCUSDT_1",
        venue_status="FILLED",
    )
    strategy = object()
    cooldown = Mock()
    logger = Mock()

    with patch.object(main_module, "_record_model_votes") as record_votes:
        acknowledged = main_module._record_acknowledged_legacy_order(
            report,
            intent,
            confidence=0.8,
            strategy=strategy,
            cooldown_manager=cooldown,
            logger=logger,
        )

    assert acknowledged is True
    record_votes.assert_called_once_with(strategy, "BTCUSDT", "BUY", logger)
    cooldown.record_trade.assert_called_once_with("BTCUSDT", "BUY", 0.001, 0.8)


@pytest.mark.parametrize(
    "status",
    [
        SubmissionStatus.NOT_SENT,
        SubmissionStatus.VENUE_REJECTED,
        SubmissionStatus.AMBIGUOUS,
    ],
)
def test_unacknowledged_submission_records_nothing(
    status: SubmissionStatus,
) -> None:
    intent = make_intent()
    report = make_report(status)
    report = SubmissionReport(
        client_order_id=intent.client_order_id,
        status=report.status,
        retry_safety=report.retry_safety,
        reason=report.reason,
        venue_status=report.venue_status,
    )
    cooldown = Mock()

    with patch.object(main_module, "_record_model_votes") as record_votes:
        acknowledged = main_module._record_acknowledged_legacy_order(
            report,
            intent,
            confidence=0.8,
            strategy=object(),
            cooldown_manager=cooldown,
            logger=Mock(),
        )

    assert acknowledged is False
    record_votes.assert_not_called()
    cooldown.record_trade.assert_not_called()


def test_main_primary_path_has_one_typed_submission_call() -> None:
    source_text = getsource(main_module.main)

    assert "ELVIS_LEGACY_EXECUTION" not in source_text
    assert "executor.execute_buy" not in source_text
    assert "executor.execute_sell" not in source_text
    assert "executor.place_order" not in source_text
    assert source_text.count("order_service.submit(intent)") == 1
