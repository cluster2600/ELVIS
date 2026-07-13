"""DB recording validation (loss-forensics fix): a test once called
add_open_position with a missing side argument, writing side='105000.0'
rows that later 'closed' for a fictional -$63,914 — poisoning Kelly and
every win-rate statistic. Invalid sides are now rejected at the door."""

from unittest.mock import patch

from utils import paper_trade_db as db


def test_record_trade_rejects_price_as_side():
    with patch.object(db, "get_conn") as gc:
        db.record_trade("BTCUSDT", 105000.0, 64000.0, 1.0)
    gc.assert_not_called()  # rejected before touching the DB


def test_add_open_position_rejects_invalid_side():
    with patch.object(db, "get_conn") as gc:
        db.add_open_position("BTCUSDT", 105000.0, 0.001, 1.0)
    gc.assert_not_called()


def test_valid_sides_pass_through():
    with patch.object(db, "get_conn", return_value=None) as gc:
        db.record_trade("BTCUSDT", "buy", 64000.0, 1.0)  # case-normalized
    gc.assert_called_once()
