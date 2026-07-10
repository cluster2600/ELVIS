"""
Tests for the multi-exchange REST API endpoints in ``trading/api/app.py``.

These verify that the endpoints return the *real* ExchangeManager output
(mocked here) instead of the old random/hardcoded mock data, and that when
no exchange manager is registered in the DI container the endpoints return a
typed structure with ``available: False`` rather than fabricated numbers.

The tests run in a minimal environment: no live exchange, database, Redis or
Vault is required. The DI container's ``exchange_manager`` is mocked.
"""

import os
from datetime import datetime
from unittest.mock import MagicMock

import pytest

# The Flask app refuses to import without a signing key. Set one before import.
os.environ.setdefault("API_SECRET_KEY", "test-secret-key-for-multi-exchange-api")

# flask / flask_cors / flask_limiter / jwt are hard requirements of the module.
pytest.importorskip("flask")
pytest.importorskip("jwt")

import importlib  # noqa: E402

import jwt  # noqa: E402

# ``trading.api.__init__`` re-exports the Flask object as ``app``, which would
# shadow the submodule on a plain ``from trading.api import app``. Import the
# module explicitly so we can monkeypatch its ``_get_exchange_manager`` helper.
api_module = importlib.import_module("trading.api.app")


@pytest.fixture
def client():
    api_module.app.config["TESTING"] = True
    with api_module.app.test_client() as c:
        yield c


@pytest.fixture
def auth_headers():
    token = jwt.encode(
        {"user": "tester"},
        api_module.app.config["SECRET_KEY"],
        algorithm="HS256",
    )
    if isinstance(token, bytes):  # PyJWT < 2 returned bytes
        token = token.decode("utf-8")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_manager(monkeypatch):
    """Patch the endpoint helper to return a mocked ExchangeManager."""
    manager = MagicMock()
    monkeypatch.setattr(api_module, "_get_exchange_manager", lambda: manager)
    return manager


@pytest.fixture
def no_manager(monkeypatch):
    """Patch the endpoint helper to simulate an unregistered manager."""
    monkeypatch.setattr(api_module, "_get_exchange_manager", lambda: None)


# ---------------------------------------------------------------------------
# /api/exchanges
# ---------------------------------------------------------------------------
def test_get_exchanges_returns_real_data(client, auth_headers, mock_manager):
    mock_manager.get_exchange_info.return_value = {
        "binance": {"health": {"status": "healthy"}, "config": {}},
        "kraken": {"health": {"status": "unhealthy"}, "config": {}},
    }

    resp = client.get("/api/exchanges", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.get_json()

    mock_manager.get_exchange_info.assert_called_once()
    assert data["available"] is True
    assert data["total_exchanges"] == 2
    assert data["healthy_exchanges"] == 1
    assert set(data["exchanges"].keys()) == {"binance", "kraken"}


def test_get_exchanges_unavailable(client, auth_headers, no_manager):
    resp = client.get("/api/exchanges", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["available"] is False
    assert data["exchanges"] == {}
    assert data["total_exchanges"] == 0
    assert data["healthy_exchanges"] == 0
    assert "not available" in data["detail"].lower()


# ---------------------------------------------------------------------------
# /api/exchanges/prices/<symbol>
# ---------------------------------------------------------------------------
def test_get_prices_returns_real_data(client, auth_headers, mock_manager):
    mock_manager.get_prices_all_exchanges.return_value = {
        "binance": 50000.0,
        "kraken": 50250.0,
    }

    resp = client.get("/api/exchanges/prices/BTCUSDT", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.get_json()

    mock_manager.get_prices_all_exchanges.assert_called_once_with("BTCUSDT")
    assert data["available"] is True
    assert data["prices"] == {"binance": 50000.0, "kraken": 50250.0}
    assert data["min_price"] == 50000.0
    assert data["max_price"] == 50250.0
    assert data["spread"] == 250.0
    assert data["spread_percentage"] == pytest.approx(0.5, abs=1e-3)


def test_get_prices_unavailable(client, auth_headers, no_manager):
    resp = client.get("/api/exchanges/prices/BTCUSDT", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["available"] is False
    assert data["prices"] == {}
    assert data["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# /api/arbitrage/opportunities
# ---------------------------------------------------------------------------
def test_get_arbitrage_returns_real_data(client, auth_headers, mock_manager):
    opps = [
        {
            "symbol": "BTCUSDT",
            "buy_exchange": "binance",
            "sell_exchange": "kraken",
            "buy_price": 50000.0,
            "sell_price": 50250.0,
            "profit_pct": 0.005,
            "profit_abs": 250.0,
            "timestamp": datetime.now().isoformat(),
        }
    ]
    mock_manager.detect_arbitrage_opportunities.return_value = opps

    resp = client.get(
        "/api/arbitrage/opportunities?symbol=BTCUSDT", headers=auth_headers
    )
    assert resp.status_code == 200
    data = resp.get_json()

    mock_manager.detect_arbitrage_opportunities.assert_called_once_with("BTCUSDT")
    assert data["available"] is True
    assert data["count"] == 1
    assert data["opportunities"] == opps


def test_get_arbitrage_unavailable(client, auth_headers, no_manager):
    resp = client.get("/api/arbitrage/opportunities", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["available"] is False
    assert data["opportunities"] == []
    assert data["count"] == 0


# ---------------------------------------------------------------------------
# /api/portfolio/consolidated
# ---------------------------------------------------------------------------
def test_get_consolidated_portfolio_returns_real_data(
    client, auth_headers, mock_manager
):
    balances = {
        "BTC": {
            "total_free": 0.15,
            "total_locked": 0.01,
            "total_balance": 0.16,
            "exchanges": {
                "binance": {"free": 0.1, "locked": 0.0, "total": 0.1},
                "kraken": {"free": 0.05, "locked": 0.01, "total": 0.06},
            },
        },
        "USDT": {
            "total_free": 1000.0,
            "total_locked": 0.0,
            "total_balance": 1000.0,
            "exchanges": {
                "binance": {"free": 1000.0, "locked": 0.0, "total": 1000.0},
            },
        },
    }
    mock_manager.get_consolidated_balance.return_value = balances

    resp = client.get("/api/portfolio/consolidated", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.get_json()

    mock_manager.get_consolidated_balance.assert_called_once()
    assert data["available"] is True
    assert data["balances"] == balances
    # Distinct exchanges across all currencies: binance + kraken.
    assert data["exchange_count"] == 2


def test_get_consolidated_portfolio_unavailable(client, auth_headers, no_manager):
    resp = client.get("/api/portfolio/consolidated", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["available"] is False
    assert data["balances"] == {}
    assert data["exchange_count"] == 0


# ---------------------------------------------------------------------------
# /api/exchanges/health
# ---------------------------------------------------------------------------
def test_get_exchanges_health_returns_real_data(client, auth_headers, mock_manager):
    now = datetime.now()
    mock_manager.check_all_exchanges_health.return_value = {
        "binance": {"status": "healthy", "last_check": now, "error_count": 0},
        "kraken": {"status": "unhealthy", "last_check": now, "error_count": 3},
    }

    resp = client.get("/api/exchanges/health", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.get_json()

    mock_manager.check_all_exchanges_health.assert_called_once()
    assert data["available"] is True
    assert data["summary"]["total_exchanges"] == 2
    assert data["summary"]["healthy_count"] == 1
    # datetime last_check must have been serialised to an ISO string.
    assert data["health"]["binance"]["last_check"] == now.isoformat()


def test_get_exchanges_health_unavailable(client, auth_headers, no_manager):
    resp = client.get("/api/exchanges/health", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["available"] is False
    assert data["health"] == {}
    assert data["summary"]["total_exchanges"] == 0
    assert data["summary"]["healthy_count"] == 0


# ---------------------------------------------------------------------------
# auth guard still applies
# ---------------------------------------------------------------------------
def test_endpoints_require_auth(client):
    resp = client.get("/api/exchanges")
    assert resp.status_code == 401
