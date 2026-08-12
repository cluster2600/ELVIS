"""Fail-closed fee-gate cut-over for purpose-specific TP regimes."""

import ast
import inspect

import pytest

import main as main_module


class RegimeString(str):
    """A string subtype which must not cross the exact active boundary."""


@pytest.mark.parametrize("regime", ["TRENDING", "RANGING", "CHOPPY"])
def test_active_mode_selects_only_current_produced_regimes(regime: str) -> None:
    selected = main_module._validated_active_fee_profile(regime)

    assert selected == regime


@pytest.mark.parametrize(
    "candidate_regime",
    [
        None,
        "",
        "REVERSAL",
        "optimal",
        "unknown",
        RegimeString("TRENDING"),
        True,
        1,
    ],
)
def test_active_mode_rejects_missing_or_unproduced_regime(
    candidate_regime: object,
) -> None:
    selected = main_module._validated_active_fee_profile(candidate_regime)

    assert selected is None


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _contains_call(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and (
            isinstance(candidate.func, ast.Name)
            and candidate.func.id == name
            or isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == name
        )
        for candidate in ast.walk(node)
    )


def test_main_active_fee_gate_uses_fresh_candidate_and_fails_closed() -> None:
    source = inspect.getsource(main_module.main)
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    active_mode_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and _is_name(node.test.left, "_fee_regime_mode")
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "active"
    ]
    assert len(active_mode_guards) == 1
    active_mode_guard = active_mode_guards[0]

    mode_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(_is_name(target, "_fee_regime_mode") for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "os"
        and node.value.func.attr == "getenv"
    ]
    assert len(mode_assignments) == 1
    mode_arguments = mode_assignments[0].value.args
    assert [
        argument.value
        for argument in mode_arguments
        if isinstance(argument, ast.Constant)
    ] == ["ELVIS_TP_REGIME_MODE", "legacy"]

    validation_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validated_active_fee_profile"
    ]
    assert len(validation_calls) == 1
    assert len(validation_calls[0].args) == 1
    assert _is_name(validation_calls[0].args[0], "take_profit_regime")
    assert any(
        validation_calls[0] in ast.walk(statement)
        for statement in active_mode_guard.body
    )

    fail_closed_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and _is_name(node.test.left, "_fee_regime")
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Is)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None
        and any(
            isinstance(statement, ast.Assign)
            and any(_is_name(target, "signal") for target in statement.targets)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value == "HOLD"
            for statement in node.body
        )
    ]
    assert len(fail_closed_guards) == 1
    guard = fail_closed_guards[0]
    hold_assignments = [
        node
        for node in guard.body
        if isinstance(node, ast.Assign)
        and any(_is_name(target, "signal") for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value == "HOLD"
    ]
    assert len(hold_assignments) == 1
    assert not any(
        _contains_call(statement, "dynamic_take_profit") for statement in guard.body
    )
    assert any(
        _contains_call(statement, "dynamic_take_profit") for statement in guard.orelse
    )
    assert any(
        _contains_call(statement, "is_trade_viable") for statement in guard.orelse
    )

    def enclosing_try(node: ast.AST) -> ast.Try | None:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.Try):
                return current
        return None

    fee_try = enclosing_try(active_mode_guard)
    assert fee_try is not None
    assert enclosing_try(validation_calls[0]) is fee_try
    assert enclosing_try(guard) is fee_try
    assert mode_assignments[0].lineno < active_mode_guard.lineno < guard.lineno


def test_active_candidate_stays_none_when_high_winrate_analysis_does_not_run() -> None:
    source = inspect.getsource(main_module.main)
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    active_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(_is_name(target, "take_profit_regime") for target in node.targets)
    ]
    resets = [
        node
        for node in active_assignments
        if isinstance(node.value, ast.Constant) and node.value.value is None
    ]
    candidate_extractions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            _is_name(target, "candidate_take_profit_regime") for target in node.targets
        )
        if isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "get"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "take_profit_regime"
    ]
    publications = [
        node
        for node in active_assignments
        if isinstance(node.value, ast.Name)
        and node.value.id == "candidate_take_profit_regime"
    ]
    assert len(resets) == 1
    assert len(candidate_extractions) == 1
    assert len(publications) == 1

    producer_ancestors = []
    current = candidate_extractions[0]
    while current in parents:
        current = parents[current]
        producer_ancestors.append(current)
    high_winrate_guards = [
        node
        for node in producer_ancestors
        if isinstance(node, ast.If)
        and any(
            isinstance(candidate, ast.Constant)
            and candidate.value == "ELVIS_WINRATE_FILTER"
            for candidate in ast.walk(node.test)
        )
    ]
    assert len(high_winrate_guards) == 1
    guard_source = ast.unparse(high_winrate_guards[0].test)
    assert "signal in ['BUY', 'SELL']" in guard_source
    assert "confidence >= 0.6" in guard_source
    assert resets[0].lineno < high_winrate_guards[0].lineno

    publication_ancestors = []
    current = publications[0]
    while current in parents:
        current = parents[current]
        publication_ancestors.append(current)
    assert high_winrate_guards[0] in publication_ancestors

    analysis_tries = [node for node in producer_ancestors if isinstance(node, ast.Try)]
    assert analysis_tries
    analysis_try = analysis_tries[0]
    assert parents[publications[0]] is analysis_try

    regime_calls = [
        node
        for node in ast.walk(analysis_try)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "detect_current_regime"
    ]
    assert len(regime_calls) == 1
    assert candidate_extractions[0].lineno < regime_calls[0].lineno
    assert regime_calls[0].lineno < publications[0].lineno

    filtering_guards = [
        node
        for node in analysis_try.body
        if isinstance(node, ast.If)
        and "filter_result['trade_approved']" in ast.unparse(node.test)
    ]
    assert len(filtering_guards) == 1
    assert analysis_try.body.index(filtering_guards[0]) < analysis_try.body.index(
        publications[0]
    )


def test_active_fee_gate_does_not_read_stale_cache_and_exits_remain_legacy() -> None:
    source = inspect.getsource(main_module.main)
    tree = ast.parse(source)
    mode_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and _is_name(node.test.left, "_fee_regime_mode")
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "active"
    ]
    assert len(mode_guards) == 1
    active_guard = mode_guards[0]
    assert not any(
        isinstance(node, ast.Constant) and node.value == "_last_regime"
        for statement in active_guard.body
        for node in ast.walk(statement)
    )
    assert any(
        isinstance(node, ast.Constant) and node.value == "_last_regime"
        for statement in active_guard.orelse
        for node in ast.walk(statement)
    )

    exit_cache_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(_is_name(target, "_tp_regime") for target in node.targets)
        and any(
            isinstance(candidate, ast.Constant) and candidate.value == "_last_regime"
            for candidate in ast.walk(node.value)
        )
    ]
    assert len(exit_cache_reads) == 1
    assert source.count("order_service.submit(intent)") == 1
