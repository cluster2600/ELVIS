"""API latency benchmark for the compatibility monitoring API.

A lightweight regression tripwire — not a load generator and not flaky: it
measures p50/p95 wall-clock over N calls to the trade-history API's read
endpoints using the Flask test client (no live server, DB mocked) and asserts
a generous p95 ceiling. Marked `perf`; run with `pytest -m perf`.
"""

import os
import statistics
import time
from unittest.mock import patch

import pytest

os.environ.setdefault("API_SECRET_KEY", "test-secret-key-for-perf")

pytest.importorskip("flask")

import importlib  # noqa: E402

api = importlib.import_module("trading.utils.trade_history_api")

N = 50
P95_CEILING_S = 0.5  # generous; trips only on a real regression


@pytest.fixture
def client():
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        yield c


def _latencies(client, method):
    lat = []
    for _ in range(N):
        t0 = time.perf_counter()
        method(client)
        lat.append(time.perf_counter() - t0)
    return lat


def _percentile(values, pct):
    s = sorted(values)
    idx = min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1))))
    return s[idx]


@pytest.mark.perf
def test_health_latency(client):
    lat = _latencies(client, lambda c: c.get("/health"))
    p50, p95 = _percentile(lat, 50), _percentile(lat, 95)
    print(f"\n/health  p50={p50*1000:.2f}ms  p95={p95*1000:.2f}ms  (N={N})")
    assert p95 < P95_CEILING_S


@pytest.mark.perf
def test_trades_read_latency(client):
    # Mock the DB layer so no Postgres is needed.
    with patch.object(api, "get_all_trades", return_value=[]):
        lat = _latencies(client, lambda c: c.get("/trades"))
    p50, p95 = _percentile(lat, 50), _percentile(lat, 95)
    print(f"\n/trades  p50={p50*1000:.2f}ms  p95={p95*1000:.2f}ms  (N={N})")
    assert p95 < P95_CEILING_S
