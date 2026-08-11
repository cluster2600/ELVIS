"""Transactional semantics for closing one legacy paper position."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import psycopg2

from utils import paper_trade_db


def _insert_position(dsn: str, symbol: str) -> int:
    connection = psycopg2.connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO np.open_positions
                    (symbol, side, entry_price, quantity, leverage)
                VALUES (%s, 'BUY', 50000, 0.01, 3)
                RETURNING id
                """,
                (symbol,),
            )
            position_id = cursor.fetchone()[0]
        connection.commit()
        return position_id
    finally:
        connection.close()


def _paper_connection(dsn: str):
    connection = psycopg2.connect(dsn)
    with connection.cursor() as cursor:
        cursor.execute("SET search_path TO np, public")
    connection.commit()
    return connection


def test_close_position_commits_one_trade_and_one_delete(
    migrated_postgres_dsn,
    monkeypatch,
) -> None:
    symbol = "TEST_CLOSE_ONCE"
    position_id = _insert_position(migrated_postgres_dsn, symbol)
    monkeypatch.setattr(
        paper_trade_db,
        "get_conn",
        lambda: _paper_connection(migrated_postgres_dsn),
    )

    assert paper_trade_db.close_position(position_id, 51000.0, 10.0, 0.2) is True
    assert paper_trade_db.close_position(position_id, 51000.0, 10.0, 0.2) is False

    connection = psycopg2.connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM np.open_positions WHERE id = %s",
                (position_id,),
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                "SELECT COUNT(*) FROM np.trades WHERE symbol = %s",
                (symbol,),
            )
            assert cursor.fetchone()[0] == 1
    finally:
        connection.close()


def test_competing_closers_cannot_record_the_same_position_twice(
    migrated_postgres_dsn,
    monkeypatch,
) -> None:
    symbol = "TEST_CONCURRENT_CLOSE"
    position_id = _insert_position(migrated_postgres_dsn, symbol)
    ready = Barrier(2)

    def connect_together():
        connection = _paper_connection(migrated_postgres_dsn)
        ready.wait(timeout=5)
        return connection

    monkeypatch.setattr(paper_trade_db, "get_conn", connect_together)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda _: paper_trade_db.close_position(
                    position_id,
                    51000.0,
                    10.0,
                    0.2,
                ),
                range(2),
            )
        )

    assert sorted(outcomes) == [False, True]
    connection = psycopg2.connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM np.trades WHERE symbol = %s",
                (symbol,),
            )
            assert cursor.fetchone()[0] == 1
    finally:
        connection.close()


def test_delete_failure_rolls_back_the_trade_insert(
    migrated_postgres_dsn,
    monkeypatch,
) -> None:
    symbol = "TEST_CLOSE_ROLLBACK"
    position_id = _insert_position(migrated_postgres_dsn, symbol)
    connection = psycopg2.connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE np.position_close_guard (
                    position_id INTEGER REFERENCES np.open_positions(id)
                )
                """)
            cursor.execute(
                "INSERT INTO np.position_close_guard (position_id) VALUES (%s)",
                (position_id,),
            )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(
        paper_trade_db,
        "get_conn",
        lambda: _paper_connection(migrated_postgres_dsn),
    )

    assert paper_trade_db.close_position(position_id, 51000.0, 10.0, 0.2) is False

    connection = psycopg2.connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM np.open_positions WHERE id = %s",
                (position_id,),
            )
            assert cursor.fetchone()[0] == 1
            cursor.execute(
                "SELECT COUNT(*) FROM np.trades WHERE symbol = %s",
                (symbol,),
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()
