#!/usr/bin/env python3
"""
Schema tests for utils.paper_trade_db.

Regression guard: reset_trading_session() INSERTs into
np.trading_session_resets and several dashboards SELECT from it, but the
table was never created by init_db()/init_db_with_balances(). On a fresh
database that raised UndefinedTable. These tests assert the CREATE TABLE
IF NOT EXISTS is issued by both init paths and the INSERT by the reset
path.

Runs without a live database: get_conn is patched to a MagicMock and we
inspect the SQL text passed to cursor.execute().
"""

from unittest.mock import MagicMock, patch

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from utils import paper_trade_db  # noqa: E402


def _executed_sql(cursor):
    """Return every SQL string passed to cursor.execute(), whitespace-squashed."""
    statements = []
    for call in cursor.execute.call_args_list:
        sql = call.args[0] if call.args else call.kwargs.get("query", "")
        statements.append(" ".join(str(sql).split()))
    return statements


def _fake_conn():
    """A MagicMock connection whose cursor works as both a callable and a
    context manager (paper_trade_db uses both styles)."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


def test_init_db_creates_session_resets_table():
    conn, cursor = _fake_conn()
    with patch.object(paper_trade_db, "get_conn", return_value=conn):
        paper_trade_db.init_db()

    sql = _executed_sql(cursor)
    assert any(
        "CREATE TABLE IF NOT EXISTS np.trading_session_resets" in s for s in sql
    ), sql
    conn.commit.assert_called()


def test_init_db_preserves_existing_open_positions():
    conn, cursor = _fake_conn()
    with patch.object(paper_trade_db, "get_conn", return_value=conn):
        paper_trade_db.init_db()

    sql = _executed_sql(cursor)
    assert any("CREATE TABLE IF NOT EXISTS np.open_positions" in s for s in sql), sql
    assert not any("DROP TABLE" in s and "open_positions" in s for s in sql), sql


def test_init_db_with_balances_creates_session_resets_table():
    conn, cursor = _fake_conn()
    with patch.object(paper_trade_db, "get_conn", return_value=conn):
        paper_trade_db.init_db_with_balances()

    sql = _executed_sql(cursor)
    assert any(
        "CREATE TABLE IF NOT EXISTS np.trading_session_resets" in s for s in sql
    ), sql
    conn.commit.assert_called()


def test_reset_trading_session_inserts_into_table():
    conn, cursor = _fake_conn()
    # COUNT(*) fetchone used for the log line
    cursor.fetchone.return_value = (1,)
    with patch.object(paper_trade_db, "get_conn", return_value=conn):
        ok = paper_trade_db.reset_trading_session()

    assert ok is True
    sql = _executed_sql(cursor)
    assert any("INSERT INTO trading_session_resets" in s for s in sql), sql
    conn.commit.assert_called()
