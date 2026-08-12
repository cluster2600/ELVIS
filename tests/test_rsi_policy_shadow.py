import ast
import inspect
from unittest.mock import Mock, patch

import pytest

import main as main_module
from trading.domain.signals import SignalAction
from trading.signals.filters import rsi_gate as legacy_rsi_gate


def observe(
    *,
    action: str = "BUY",
    confidence: float = 0.8,
    rsi: object = 50.0,
    legacy_rsi_reason: str | None = None,
    logger: Mock | None = None,
) -> tuple[object, Mock]:
    selected_logger = logger or Mock()
    result = main_module._observe_rsi_policy_shadow(
        signal_symbol="BTCUSDT",
        legacy_action=action,
        legacy_confidence=confidence,
        reference_price=123456.75,
        strategy_id="EnsembleStrategy",
        rsi=rsi,
        legacy_rsi_reason=legacy_rsi_reason,
        logger=selected_logger,
    )
    return result, selected_logger


def logged_extra(logger: Mock) -> dict[str, object]:
    method = logger.info if logger.info.called else logger.warning
    return method.call_args.kwargs["extra"]


@pytest.mark.parametrize("action", ["BUY", "SELL"])
@pytest.mark.parametrize("missing_rsi", [None, float("nan"), main_module.pd.NA])
def test_missing_rsi_normalization_preserves_legacy_gate_outcome(
    action: str,
    missing_rsi: object,
) -> None:
    old_imputed_rsi = 50.0
    normalized_rsi = None if not main_module.pd.notna(missing_rsi) else missing_rsi

    assert legacy_rsi_gate(action, normalized_rsi) == legacy_rsi_gate(
        action,
        old_imputed_rsi,
    )


@pytest.mark.parametrize(
    ("action", "rsi"),
    [
        ("BUY", 30.0),
        ("BUY", 70.0),
        ("BUY", 70.1),
        ("SELL", 29.9),
        ("SELL", 30.0),
        ("SELL", 80.0),
    ],
)
def test_shadow_matches_legacy_on_valid_rsi_boundaries(
    action: str,
    rsi: float,
) -> None:
    legacy_action, legacy_reason = legacy_rsi_gate(action, rsi)

    result, logger = observe(
        action=action,
        rsi=rsi,
        legacy_rsi_reason=legacy_reason,
    )
    extra = logged_extra(logger)

    assert result is None
    assert extra["legacy_action"] == legacy_action
    assert extra["candidate_action"] == legacy_action
    assert extra["legacy_confidence"] == (0.0 if legacy_reason else 0.8)
    assert extra["candidate_confidence"] == (0.0 if legacy_reason else 0.8)
    assert extra["matched"] is True
    logger.info.assert_called_once()
    logger.warning.assert_not_called()


@pytest.mark.parametrize(
    ("action", "rsi"),
    [
        ("BUY", None),
        ("BUY", float("nan")),
        ("BUY", -0.1),
        ("SELL", 100.1),
    ],
)
def test_shadow_records_expected_fail_closed_divergence(
    action: str,
    rsi: object,
) -> None:
    _, legacy_reason = legacy_rsi_gate(action, rsi)  # type: ignore[arg-type]
    result, logger = observe(
        action=action,
        rsi=rsi,
        legacy_rsi_reason=legacy_reason,
    )
    extra = logged_extra(logger)

    assert result is None
    assert extra["legacy_action"] == action
    assert extra["legacy_confidence"] == 0.8
    assert extra["candidate_action"] == "HOLD"
    assert extra["candidate_confidence"] == 0.0
    assert extra["matched"] is False
    assert extra["candidate_reasons"]
    logger.warning.assert_called_once()
    logger.info.assert_not_called()


@pytest.mark.parametrize(
    ("action", "rsi"),
    [
        ("BUY", float("inf")),
        ("SELL", float("-inf")),
        ("BUY", 100.1),
        ("SELL", -0.1),
    ],
)
def test_shadow_can_match_when_both_paths_veto_invalid_rsi(
    action: str,
    rsi: float,
) -> None:
    _, legacy_reason = legacy_rsi_gate(action, rsi)

    _, logger = observe(
        action=action,
        rsi=rsi,
        legacy_rsi_reason=legacy_reason,
    )
    extra = logged_extra(logger)

    assert legacy_reason is not None
    assert extra["legacy_action"] == "HOLD"
    assert extra["candidate_action"] == "HOLD"
    assert extra["matched"] is True
    logger.info.assert_called_once()


def test_shadow_log_has_correlated_structured_fields_without_market_payload() -> None:
    _, logger = observe(
        action="SELL",
        confidence=0.7,
        rsi=25.0,
        legacy_rsi_reason="rsi_gate: SELL blocked",
    )
    extra = logged_extra(logger)

    assert set(extra) == {
        "event_type",
        "migration_slice",
        "migration_mode",
        "stage",
        "policy_id",
        "shadow_evaluation_id",
        "signal_symbol",
        "strategy_id",
        "rsi",
        "legacy_action",
        "legacy_confidence",
        "candidate_action",
        "candidate_confidence",
        "legacy_reason",
        "candidate_reasons",
        "action_match",
        "confidence_match",
        "matched",
    }
    assert extra["event_type"] == "signal_policy_shadow"
    assert extra["migration_slice"] == "M6b2"
    assert extra["migration_mode"] == "shadow"
    assert extra["stage"] == "roadmap_filters.rsi"
    assert extra["policy_id"] == "rsi-gate"
    assert extra["signal_symbol"] == "BTCUSDT"
    assert extra["strategy_id"] == "EnsembleStrategy"
    assert extra["rsi"] == 25.0
    assert isinstance(extra["shadow_evaluation_id"], str)
    assert len(extra["shadow_evaluation_id"]) == 32
    assert "symbol" not in extra
    assert "mode" not in extra
    assert "market_data" not in extra


def test_candidate_failure_is_sanitized_and_never_propagates() -> None:
    logger = Mock()

    with patch.object(
        main_module.SignalPolicyPipeline,
        "evaluate",
        side_effect=RuntimeError("private candidate details"),
    ):
        result, _ = observe(logger=logger)

    assert result is None
    logger.warning.assert_called_once()
    call = logger.warning.call_args
    assert call.args[1] == "RuntimeError"
    assert "private candidate details" not in repr(call)
    assert call.kwargs["extra"]["event_type"] == "signal_policy_shadow_error"


def test_logger_failures_are_swallowed() -> None:
    logger = Mock()
    logger.info.side_effect = RuntimeError("private logger details")
    logger.warning.side_effect = RuntimeError("private fallback details")

    result, _ = observe(logger=logger)

    assert result is None
    logger.info.assert_called_once()
    logger.warning.assert_called_once()


def test_shadow_observer_has_no_stateful_dependency_or_return_value() -> None:
    source = inspect.getsource(main_module._observe_rsi_policy_shadow)
    tree = ast.parse(source)
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    referenced_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    forbidden = {
        "executor",
        "order_service",
        "submit",
        "cooldown_manager",
        "record_trade",
        "_record_model_votes",
        "paper_trade_db",
        "container",
    }
    forbidden_attributes = {
        "submit",
        "execute_buy",
        "execute_sell",
        "place_order",
        "record_trade",
        "record_entry",
        "score_closed_trades",
        "execute",
        "commit",
    }

    assert referenced_names.isdisjoint(forbidden)
    assert referenced_attributes.isdisjoint(forbidden_attributes)
    assert inspect.signature(
        main_module._observe_rsi_policy_shadow
    ).return_annotation in (
        None,
        type(None),
        "None",
    )


def test_main_wires_one_non_authoritative_shadow_call_after_legacy_filter() -> None:
    source = inspect.getsource(main_module.main)
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    shadow_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_observe_rsi_policy_shadow"
    ]
    legacy_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_signal_filters"
    ]

    assert len(shadow_calls) == 1
    assert len(legacy_calls) == 1
    shadow_call = shadow_calls[0]
    assert isinstance(parents[shadow_call], ast.Expr)
    assert shadow_call.lineno > legacy_calls[0].lineno

    keywords = {keyword.arg: keyword.value for keyword in shadow_call.keywords}
    assert isinstance(keywords["signal_symbol"], ast.Name)
    assert keywords["signal_symbol"].id == "symbol"
    assert isinstance(keywords["rsi"], ast.Name)
    assert keywords["rsi"].id == "_filter_rsi"

    legacy_keywords = {
        keyword.arg: keyword.value for keyword in legacy_calls[0].keywords
    }
    assert isinstance(legacy_keywords["rsi"], ast.Name)
    assert legacy_keywords["rsi"].id == "_filter_rsi"

    mode_checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and isinstance(node.left.func, ast.Attribute)
        and isinstance(node.left.func.value, ast.Name)
        and node.left.func.value.id == "os"
        and node.left.func.attr == "getenv"
        and len(node.left.args) == 2
        and all(isinstance(argument, ast.Constant) for argument in node.left.args)
        and [argument.value for argument in node.left.args]
        == ["ELVIS_RSI_POLICY_MODE", "legacy"]
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == "shadow"
    ]
    assert len(mode_checks) == 1
    assert isinstance(parents[mode_checks[0]], ast.If)

    ancestors = []
    current = parents[shadow_call]
    while current in parents:
        ancestors.append(current)
        current = parents[current]
    ancestors.append(current)
    assert parents[mode_checks[0]] in ancestors


def test_shadow_cannot_add_an_execution_path() -> None:
    source = inspect.getsource(main_module.main)

    assert source.count("order_service.submit(intent)") == 1
    assert "executor.execute_buy" not in source
    assert "executor.execute_sell" not in source
    assert "executor.place_order" not in source
    assert SignalAction.HOLD.value == "HOLD"
