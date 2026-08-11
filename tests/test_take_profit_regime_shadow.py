"""Side-effect-free shadow wiring for the take-profit regime migration."""

import ast
import inspect
from unittest.mock import Mock

import pytest

import main as main_module


def observe(
    *,
    quality_regime: object = "optimal",
    candidate_regime: object = "RANGING",
    logger: Mock | None = None,
) -> tuple[object, Mock]:
    selected_logger = logger or Mock()
    result = main_module._observe_take_profit_regime_shadow(
        signal_symbol="BTCUSDT",
        quality_regime=quality_regime,
        candidate_regime=candidate_regime,
        logger=selected_logger,
    )
    return result, selected_logger


def logged_extra(logger: Mock) -> dict[str, object]:
    method = logger.info if logger.info.called else logger.warning
    return method.call_args.kwargs["extra"]


@pytest.mark.parametrize(
    "quality_regime", ["optimal", "favorable", "neutral", "unfavorable", None]
)
def test_shadow_resolves_legacy_quality_labels_to_ranging(
    quality_regime: object,
) -> None:
    result, logger = observe(quality_regime=quality_regime)
    extra = logged_extra(logger)

    assert result is None
    assert extra["legacy_effective_regime"] == "RANGING"
    assert extra["candidate_regime"] == "RANGING"
    assert extra["matched"] is True
    logger.info.assert_called_once()
    logger.warning.assert_not_called()


@pytest.mark.parametrize("candidate_regime", ["TRENDING", "CHOPPY"])
def test_shadow_records_candidate_behavior_change(candidate_regime: str) -> None:
    _, logger = observe(candidate_regime=candidate_regime)
    extra = logged_extra(logger)

    assert extra["legacy_effective_regime"] == "RANGING"
    assert extra["candidate_regime"] == candidate_regime
    assert extra["matched"] is False
    logger.warning.assert_called_once()
    logger.info.assert_not_called()


@pytest.mark.parametrize("candidate_regime", [None, "unknown", True])
def test_shadow_marks_invalid_candidate_unavailable(candidate_regime: object) -> None:
    _, logger = observe(candidate_regime=candidate_regime)
    extra = logged_extra(logger)

    assert extra["candidate_regime"] is None
    assert extra["matched"] is False
    assert extra["candidate_available"] is False
    logger.warning.assert_called_once()


def test_shadow_log_has_bounded_structured_fields() -> None:
    _, logger = observe(quality_regime="optimal", candidate_regime="TRENDING")
    extra = logged_extra(logger)

    assert set(extra) == {
        "event_type",
        "migration_slice",
        "migration_mode",
        "stage",
        "shadow_evaluation_id",
        "signal_symbol",
        "quality_regime",
        "legacy_effective_regime",
        "candidate_regime",
        "candidate_available",
        "matched",
    }
    assert extra["event_type"] == "take_profit_regime_shadow"
    assert extra["migration_slice"] == "M7g"
    assert extra["migration_mode"] == "shadow"
    assert extra["stage"] == "pretrade.take_profit_regime"
    assert extra["signal_symbol"] == "BTCUSDT"
    assert isinstance(extra["shadow_evaluation_id"], str)
    assert len(extra["shadow_evaluation_id"]) == 32
    assert "symbol" not in extra
    assert "mode" not in extra
    assert "market_data" not in extra


def test_logger_failures_are_swallowed() -> None:
    logger = Mock()
    logger.warning.side_effect = RuntimeError("private logger details")

    result, _ = observe(candidate_regime="TRENDING", logger=logger)

    assert result is None
    logger.warning.assert_called_once()


def test_shadow_helper_has_no_stateful_dependency_or_return_value() -> None:
    source = inspect.getsource(main_module._observe_take_profit_regime_shadow)
    tree = ast.parse(source)
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    referenced_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert referenced_names.isdisjoint(
        {
            "executor",
            "order_service",
            "cooldown_manager",
            "paper_trade_db",
            "container",
            "main",
        }
    )
    assert referenced_attributes.isdisjoint(
        {
            "submit",
            "execute_buy",
            "execute_sell",
            "place_order",
            "record_trade",
            "record_entry",
            "commit",
        }
    )
    assert inspect.signature(
        main_module._observe_take_profit_regime_shadow
    ).return_annotation in (None, type(None), "None")


def test_main_wires_one_non_authoritative_shadow_call() -> None:
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
        and node.func.id == "_observe_take_profit_regime_shadow"
    ]
    submit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit"
    ]

    assert len(shadow_calls) == 1
    assert len(submit_calls) == 1
    shadow_call = shadow_calls[0]
    assert isinstance(parents[shadow_call], ast.Expr)
    assert shadow_call.lineno < submit_calls[0].lineno

    keywords = {keyword.arg: keyword.value for keyword in shadow_call.keywords}
    assert isinstance(keywords["signal_symbol"], ast.Name)
    assert keywords["signal_symbol"].id == "symbol"
    assert isinstance(keywords["candidate_regime"], ast.Name)
    assert keywords["candidate_regime"].id == "take_profit_regime"

    mode_checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and isinstance(node.left.func, ast.Attribute)
        and isinstance(node.left.func.value, ast.Name)
        and node.left.func.value.id == "os"
        and node.left.func.attr == "getenv"
        and [
            argument.value
            for argument in node.left.args
            if isinstance(argument, ast.Constant)
        ]
        == ["ELVIS_TP_REGIME_MODE", "legacy"]
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == "shadow"
    ]
    assert len(mode_checks) == 1
    mode_if = parents[mode_checks[0]]
    assert isinstance(mode_if, ast.If)
    assert len(mode_if.body) == 1
    assert mode_if.body[0] is parents[shadow_call]
    assert isinstance(mode_if.body[0], ast.Expr)

    ancestors = []
    current = parents[shadow_call]
    while current in parents:
        ancestors.append(current)
        current = parents[current]
    assert mode_if in ancestors

    resets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "take_profit_regime"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]
    candidate_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "take_profit_regime"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "get"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "take_profit_regime"
    ]
    analysis_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "analyze_signal_quality"
    ]
    cached_candidate_targets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Subscript))
        and isinstance(node.ctx, ast.Store)
        and "take_profit_regime" in ast.unparse(node)
    ]

    assert len(resets) == 1
    assert len(candidate_assignments) == 1
    assert len(analysis_calls) == 1
    assert resets[0].lineno < analysis_calls[0].lineno
    assert analysis_calls[0].lineno < candidate_assignments[0].lineno
    assert candidate_assignments[0].lineno < shadow_call.lineno
    assert cached_candidate_targets == []

    def enclosing_symbol_loops(node: ast.AST) -> set[ast.For]:
        loops = set()
        current_node = node
        while current_node in parents:
            current_node = parents[current_node]
            if (
                isinstance(current_node, ast.For)
                and isinstance(current_node.target, ast.Name)
                and current_node.target.id == "symbol"
                and isinstance(current_node.iter, ast.Name)
                and current_node.iter.id == "symbols_to_trade"
            ):
                loops.add(current_node)
        return loops

    common_symbol_loops = (
        enclosing_symbol_loops(resets[0])
        & enclosing_symbol_loops(candidate_assignments[0])
        & enclosing_symbol_loops(shadow_call)
    )
    assert len(common_symbol_loops) == 1


def test_shadow_does_not_change_legacy_cache_or_execution_count() -> None:
    source = inspect.getsource(main_module.main)
    tree = ast.parse(source)
    cache_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "main"
            and target.value.attr == "_last_regime"
            and isinstance(target.slice, ast.Name)
            and target.slice.id == "symbol"
            for target in node.targets
        )
    ]

    assert source.count("main._last_regime[symbol]") == 1
    assert len(cache_assignments) == 1
    cached_value = cache_assignments[0].value
    assert isinstance(cached_value, ast.Subscript)
    assert isinstance(cached_value.slice, ast.Constant)
    assert cached_value.slice.value == "class"
    regime_value = cached_value.value
    assert isinstance(regime_value, ast.Subscript)
    assert isinstance(regime_value.value, ast.Name)
    assert regime_value.value.id == "regime_result"
    assert isinstance(regime_value.slice, ast.Constant)
    assert regime_value.slice.value == "regime"
    assert source.count("order_service.submit(intent)") == 1
    assert "executor.execute_buy" not in source
    assert "executor.execute_sell" not in source
    assert "executor.place_order" not in source
