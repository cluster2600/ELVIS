"""Tests for the Prometheus ``GET /metrics`` endpoint of the trade-history API.

These tests exercise the Flask test client only. The paper-trade database is
mocked (``utils.paper_trade_db.get_conn`` returns ``None``; ``get_all_trades``
is patched) so no PostgreSQL / Redis / Vault service is required.
"""

from unittest.mock import patch

import pytest

# The endpoint depends on prometheus_client / flask; skip cleanly if unavailable.
pytest.importorskip("flask")
pytest.importorskip("prometheus_client")


@pytest.fixture
def client():
    """Flask test client for the trade-history API with the DB mocked out."""
    import utils.paper_trade_db as db

    # get_conn() -> None makes get_trade_count()/get_open_positions() return
    # 0 / [] without touching a live database.
    with patch.object(db, "get_conn", return_value=None):
        from trading.utils.trade_history_api import app

        app.config.update(TESTING=True)
        with app.test_client() as test_client:
            yield test_client


def test_metrics_returns_prometheus_text(client):
    """/metrics returns 200, text/plain content type, and known metric names."""
    resp = client.get("/metrics")

    assert resp.status_code == 200

    content_type = resp.headers.get("Content-Type", "")
    assert content_type.startswith("text/plain")

    body = resp.get_data(as_text=True)
    # A real gauge we populate on every scrape from utils.paper_trade_db.
    assert "elvis_total_trades" in body
    assert "elvis_open_positions_count" in body
    assert "elvis_portfolio_value" in body


def test_metrics_reflects_paper_equity_from_db(client):
    """Paper equity gauge = base equity (2000) + summed realized P&L of trades."""
    import trading.utils.trade_history_api as api

    # Two trades with pnl in column index 6 (id, ts, symbol, side, price, qty, pnl, fee).
    # get_all_trades is imported into the api module namespace, so patch it there.
    fake_trades = [
        (1, "t", "BTCUSDT", "SELL", 50000.0, 0.1, 12.5, 0.1),
        (2, "t", "BTCUSDT", "SELL", 51000.0, 0.1, -2.5, 0.1),
    ]
    with patch.object(api, "get_all_trades", return_value=fake_trades):
        resp = client.get("/metrics")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # 2000 base + (12.5 - 2.5) = 2010.0
    assert "elvis_portfolio_value 2010.0" in body


def test_metrics_exempt_from_api_key(client, monkeypatch):
    """The scrape endpoint stays reachable even when API-key auth is enforced."""
    import trading.utils.trade_history_api as api

    # Simulate a server that requires X-API-Key on every other route.
    monkeypatch.setattr(api, "_API_KEY", "secret-token")

    # No X-API-Key header supplied, yet /metrics must still return 200.
    assert client.get("/metrics").status_code == 200
    # A normal data route is still protected.
    assert client.get("/trades/count").status_code == 401
