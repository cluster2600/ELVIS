"""SQL semantics required by adaptive model feedback."""

import datetime as dt

import psycopg2

from utils import paper_trade_db


def test_zero_pnl_close_is_returned_before_a_later_profitable_close(
    migrated_postgres_dsn,
    monkeypatch,
):
    symbol = "TEST_ZERO_PNL"
    observed_at = dt.datetime(2026, 8, 11, 12, 0, 0)
    connection = psycopg2.connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO np.trades (
                    timestamp, symbol, side, price, quantity, pnl, fee
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    (
                        observed_at - dt.timedelta(seconds=5),
                        symbol,
                        "BUY",
                        100.0,
                        1.0,
                        0.0,
                        0.0,
                    ),
                    (
                        observed_at + dt.timedelta(seconds=5),
                        symbol,
                        "SELL",
                        100.0,
                        1.0,
                        0.0,
                        0.0,
                    ),
                    (
                        observed_at + dt.timedelta(seconds=60),
                        symbol,
                        "SELL",
                        105.0,
                        1.0,
                        5.0,
                        0.0,
                    ),
                ),
            )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(
        paper_trade_db,
        "get_conn",
        lambda: psycopg2.connect(migrated_postgres_dsn),
    )

    closing = paper_trade_db.get_first_closing_trade_after(symbol, observed_at)

    assert closing is not None
    closed_at, pnl = closing
    assert closed_at == observed_at + dt.timedelta(seconds=5)
    assert pnl == 0.0
