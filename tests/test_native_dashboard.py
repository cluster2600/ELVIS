"""Non-curses logic of scripts/native_console_dashboard.py.

Covers the API-error handling that used to crash the panes, the TTL cache
that keeps the 1s render loop off the network, the real system-status
checks, and the dance-frame invariants.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from scripts.native_console_dashboard import NativeConsoleDashboard


@pytest.fixture
def dash(monkeypatch):
    # Pin the env key so the OpenBao fallback isn't exercised implicitly;
    # TestDashboardApiKey covers the fallback explicitly.
    monkeypatch.setenv("API_KEY", "fixture-key")
    return NativeConsoleDashboard()


def _resp(ok=True, payload=None):
    r = MagicMock()
    r.ok = ok
    r.json.return_value = payload
    return r


class TestGetApiData:
    def test_dict_payload_passes(self, dash):
        with patch("requests.get", return_value=_resp(payload={"status": "healthy"})):
            assert dash.get_api_data("/health") == {"status": "healthy"}

    def test_error_dict_becomes_none(self, dash):
        with patch("requests.get", return_value=_resp(payload={"error": "no auth"})):
            assert dash.get_api_data("/trades") is None

    def test_non_ok_becomes_none(self, dash):
        with patch("requests.get", return_value=_resp(ok=False, payload={})):
            assert dash.get_api_data("/trades") is None

    def test_network_failure_becomes_none(self, dash):
        with patch("requests.get", side_effect=OSError("down")):
            assert dash.get_api_data("/trades") is None

    def test_api_key_header_sent(self, dash, monkeypatch):
        monkeypatch.setenv("API_KEY", "sekrit")
        with patch("requests.get", return_value=_resp(payload=[])) as mock_get:
            dash.get_api_data("/trades")
        assert mock_get.call_args[1]["headers"] == {"X-API-Key": "sekrit"}


class TestDashboardApiKey:
    def test_env_key_takes_priority(self, dash, monkeypatch):
        monkeypatch.setenv("API_KEY", "from-env")
        assert dash._dashboard_api_key() == "from-env"

    def test_openbao_fallback_when_env_absent(self, dash, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        sm = MagicMock()
        sm.get_secret.return_value = "from-vault"
        with patch(
            "utils.secrets_manager.get_enhanced_secrets_manager", return_value=sm
        ):
            assert dash._dashboard_api_key() == "from-vault"
        sm.get_secret.assert_called_once_with(
            "DASHBOARD_API_KEY", warn_if_missing=False
        )

    def test_resolution_cached_per_process(self, dash, monkeypatch):
        monkeypatch.setenv("API_KEY", "first")
        assert dash._dashboard_api_key() == "first"
        monkeypatch.setenv("API_KEY", "second")
        assert dash._dashboard_api_key() == "first"  # cached


class TestSharedResolver:
    """The env->OpenBao resolution is a single shared helper (review #46)."""

    def test_vault_key_map_tuple_pinned(self):
        # A typo in the path/field tuple would be invisible to mocked tests
        from utils.secrets_manager import _VAULT_KEY_MAP

        assert _VAULT_KEY_MAP["DASHBOARD_API_KEY"] == ("dashboard", "api_key")

    def test_env_priority(self, monkeypatch):
        from utils.secrets_manager import resolve_dashboard_api_key

        monkeypatch.setenv("API_KEY", "from-env")
        assert resolve_dashboard_api_key() == "from-env"

    def test_vault_fallback(self, monkeypatch):
        import utils.secrets_manager as smod

        monkeypatch.delenv("API_KEY", raising=False)
        sm = MagicMock()
        sm.get_secret.return_value = "from-vault"
        monkeypatch.setattr(smod, "get_enhanced_secrets_manager", lambda: sm)
        assert smod.resolve_dashboard_api_key() == "from-vault"
        sm.get_secret.assert_called_once_with(
            "DASHBOARD_API_KEY", warn_if_missing=False
        )

    def test_failure_logged_not_swallowed(self, monkeypatch, caplog):
        import logging as _logging

        import utils.secrets_manager as smod

        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setattr(
            smod,
            "get_enhanced_secrets_manager",
            lambda: (_ for _ in ()).throw(RuntimeError("vault down")),
        )
        with caplog.at_level(_logging.WARNING, logger="utils.secrets_manager"):
            assert smod.resolve_dashboard_api_key() is None
        assert any("OpenBao resolution failed" in r.message for r in caplog.records)

    def test_server_module_uses_shared_resolver(self):
        # Symmetry guard: the server must not re-grow its own copy
        import trading.utils.trade_history_api as api
        from utils.secrets_manager import resolve_dashboard_api_key

        assert api.resolve_dashboard_api_key is resolve_dashboard_api_key


class TestNetCache:
    def test_second_call_within_ttl_hits_cache(self, dash):
        with patch("requests.get", return_value=_resp(payload={"price": "1"})) as g:
            a = dash._get_json_cached("https://x/y", {"s": "BTC"}, ttl=60)
            b = dash._get_json_cached("https://x/y", {"s": "BTC"}, ttl=60)
        assert a == b == {"price": "1"}
        assert g.call_count == 1

    def test_distinct_params_are_distinct_entries(self, dash):
        with patch("requests.get", return_value=_resp(payload={})) as g:
            dash._get_json_cached("https://x/y", {"s": "BTC"}, ttl=60)
            dash._get_json_cached("https://x/y", {"s": "BNB"}, ttl=60)
        assert g.call_count == 2

    def test_failure_cached_as_none(self, dash):
        with patch("requests.get", side_effect=OSError("down")) as g:
            assert dash._get_json_cached("https://x/y", ttl=60) is None
            assert dash._get_json_cached("https://x/y", ttl=60) is None
        assert g.call_count == 1

    def test_ticker_price_parses_and_defaults(self, dash):
        with patch("requests.get", return_value=_resp(payload={"price": "64000.5"})):
            assert dash._ticker_price("BTCUSDT", 1.0) == 64000.5
        dash._net_cache = {}
        with patch("requests.get", side_effect=OSError("down")):
            assert dash._ticker_price("BTCUSDT", 42.0) == 42.0


class TestSystemStatuses:
    def test_statuses_cached_within_ttl(self, dash):
        with patch("socket.create_connection", side_effect=OSError("no")) as sc:
            with patch("utils.paper_trade_db.get_conn", return_value=None):
                s1 = dash._system_statuses(ttl=60)
                s2 = dash._system_statuses(ttl=60)
        assert s1 == s2 == {"redis": False, "postgres": False}
        assert sc.call_count == 1

    def test_statuses_reflect_reachability(self, dash):
        conn = MagicMock()
        with patch("socket.create_connection", return_value=MagicMock()):
            with patch("utils.paper_trade_db.get_conn", return_value=conn):
                s = dash._system_statuses(ttl=0)
        assert s == {"redis": True, "postgres": True}
        conn.close.assert_called_once()


class TestElvisFrames:
    def test_four_frames_same_height(self, dash):
        heights = {len(f) for f in dash.ELVIS_FRAMES}
        assert len(dash.ELVIS_FRAMES) == 4
        assert heights == {7}  # header rows 1-7, boxes start at row 8

    def test_frames_fit_left_gutter(self, dash):
        # art draws at x=6; logo starts around col 42 on a 120-col terminal
        assert all(len(line) <= 20 for f in dash.ELVIS_FRAMES for line in f)

    def test_captions_match_frames(self, dash):
        assert len(dash.ELVIS_CAPTIONS) == len(dash.ELVIS_FRAMES)
