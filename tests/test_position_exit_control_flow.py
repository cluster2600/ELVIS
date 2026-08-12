"""Regression tests for one-shot legacy position exits."""

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import main as main_module
from utils import paper_trade_db


def _position_loop() -> ast.For:
    root = Path(__file__).parents[1]
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        calls = {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        if {"_stop_loss_threshold", "_close_position_by_id"} <= calls:
            candidates.append(node)
    assert len(candidates) == 1
    return candidates[0]


def _is_name(node: ast.AST, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _sql_calls(cursor: MagicMock) -> list[str]:
    return [
        " ".join(str(call.args[0]).split()) for call in cursor.execute.call_args_list
    ]


def test_close_position_reports_whether_the_row_was_deleted(monkeypatch) -> None:
    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.rowcount = 1
    cursor.fetchone.return_value = (
        "BTCUSDT",
        "BUY",
        50000.0,
        0.01,
        3.0,
    )
    monkeypatch.setattr(paper_trade_db, "get_conn", lambda: connection)
    legacy_record_trade = MagicMock()
    monkeypatch.setattr(paper_trade_db, "record_trade", legacy_record_trade)

    assert paper_trade_db.close_position(7, "51000", "10", "0.2") is True

    sql = _sql_calls(cursor)
    assert any(statement.endswith("WHERE id = %s FOR UPDATE") for statement in sql)
    assert any(statement.startswith("INSERT INTO trades") for statement in sql)
    cursor.execute.assert_any_call(
        "DELETE FROM open_positions WHERE id = %s",
        (7,),
    )
    legacy_record_trade.assert_not_called()
    connection.commit.assert_called_once_with()
    connection.rollback.assert_not_called()


def test_close_position_fails_when_the_row_is_missing(monkeypatch) -> None:
    connection = MagicMock()
    connection.cursor.return_value.fetchone.return_value = None
    monkeypatch.setattr(paper_trade_db, "get_conn", lambda: connection)

    assert paper_trade_db.close_position(999, 51000.0, 0.0) is False
    connection.commit.assert_not_called()


def test_close_position_fails_when_database_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(paper_trade_db, "get_conn", lambda: None)

    assert paper_trade_db.close_position(7, 51000.0, 10.0) is False


def test_close_position_rolls_back_and_fails_on_database_error(monkeypatch) -> None:
    connection = MagicMock()
    connection.cursor.return_value.execute.side_effect = RuntimeError("db failure")
    monkeypatch.setattr(paper_trade_db, "get_conn", lambda: connection)

    assert paper_trade_db.close_position(7, 51000.0, 10.0) is False
    connection.rollback.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_close_position_rolls_back_when_exact_delete_loses_the_row(monkeypatch) -> None:
    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.fetchone.return_value = (
        "BTCUSDT",
        "BUY",
        50000.0,
        0.01,
        3.0,
    )
    cursor.rowcount = 0
    monkeypatch.setattr(paper_trade_db, "get_conn", lambda: connection)

    assert paper_trade_db.close_position(7, 51000.0, 10.0) is False
    connection.commit.assert_not_called()
    connection.rollback.assert_called_once_with()


@pytest.mark.parametrize(
    ("exit_price", "pnl", "fee"),
    [
        (float("nan"), 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (51000.0, float("inf"), 0.0),
        (51000.0, 1.0, -0.1),
    ],
)
def test_close_position_rejects_invalid_values_before_sql(
    monkeypatch,
    exit_price,
    pnl,
    fee,
) -> None:
    connection = MagicMock()
    monkeypatch.setattr(paper_trade_db, "get_conn", lambda: connection)

    assert paper_trade_db.close_position(7, exit_price, pnl, fee) is False
    connection.cursor.assert_not_called()
    connection.commit.assert_not_called()
    connection.rollback.assert_called_once_with()


@pytest.mark.parametrize("outcome", [False, True])
def test_close_helper_propagates_the_database_outcome(monkeypatch, outcome) -> None:
    position = (7, "BTCUSDT", "BUY", 50000.0, 0.01, 3.0, None)
    close = MagicMock(return_value=outcome)
    monkeypatch.setattr(paper_trade_db, "close_position", close)

    assert (
        main_module._close_position_by_id(
            position,
            51000.0,
            10.0,
            logging.getLogger("position-exit-test"),
        )
        is outcome
    )
    close.assert_called_once()
    assert close.call_args.args[:3] == (7, 51000.0, 10.0)
    assert close.call_args.args[3] == pytest.approx(0.204)


def test_every_successful_exit_short_circuits_other_exit_checks() -> None:
    loop = _position_loop()
    named_success_guards = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If) and _is_name(node.test, "success")
    ]
    direct_close_guards = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "_close_position_by_id"
    ]
    guards = named_success_guards + direct_close_guards

    assert len(named_success_guards) == 2
    assert len(direct_close_guards) == 1
    assert all(
        any(isinstance(statement, ast.Continue) for statement in guard.body)
        for guard in guards
    )


def test_position_exit_loop_has_no_undefined_legacy_close_names() -> None:
    referenced = {
        node.id for node in ast.walk(_position_loop()) if isinstance(node, ast.Name)
    }

    assert referenced.isdisjoint({"close_signal", "close_size"})


def test_balanced_strategy_checks_every_close_outcome() -> None:
    root = Path(__file__).parents[1]
    tree = ast.parse(
        (root / "trading/strategies/balanced_starter.py").read_text(encoding="utf-8")
    )
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "close_position"
    ]

    assert len(calls) == 6
    assert all(
        isinstance(parents[call], ast.If) and parents[call].test is call
        for call in calls
    )
