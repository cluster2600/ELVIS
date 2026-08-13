"""Tests for JWT role-based access control (RBAC) in trading/api/app.py.

Documented in SECURITY.md: the login token carries a ``role`` claim; mutating
endpoints require the ``admin`` role; tokens without a role claim degrade to
read-only ``viewer``. Runs with no live services (Flask test client only).
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("API_SECRET_KEY", "test-secret-key-for-rbac-32-bytes-minimum")

pytest.importorskip("flask")
pytest.importorskip("jwt")

import importlib  # noqa: E402

import jwt  # noqa: E402

api_module = importlib.import_module("trading.api.app")
api_runner = importlib.import_module("trading.scripts.run_api")


@pytest.fixture
def client():
    original_state = dict(api_module.bot_state)
    api_module.bot_state.update(
        running=False,
        mode="paper",
        start_time=None,
        strategy=None,
    )
    api_module.app.config["TESTING"] = True
    with api_module.app.test_client() as c:
        yield c
    api_module.bot_state.clear()
    api_module.bot_state.update(original_state)


def _token(role=None, user="tester"):
    payload = {"user": user, "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
    if role is not None:
        payload["role"] = role
    return jwt.encode(payload, api_module.app.config["SECRET_KEY"], algorithm="HS256")


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_login_issues_role_claim(client, monkeypatch):
    monkeypatch.setenv("API_USERNAME", "op")
    monkeypatch.setenv("API_PASSWORD", "pw")
    r = client.post("/api/auth/login", json={"username": "op", "password": "pw"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["role"] == "admin"  # default role
    decoded = jwt.decode(
        body["token"], api_module.app.config["SECRET_KEY"], algorithms=["HS256"]
    )
    assert decoded["role"] == "admin"


def test_viewer_can_read_but_not_mutate(client):
    viewer = _token(role="viewer")
    # read endpoint: any valid token works
    r = client.get("/api/bot/status", headers=_auth(viewer))
    assert r.status_code == 200
    # mutating endpoint: viewer is rejected with 403
    r = client.post("/api/bot/start", headers=_auth(viewer), json={})
    assert r.status_code == 403
    assert "admin" in r.get_json()["error"]


def test_legacy_token_without_role_is_viewer(client):
    legacy = _token(role=None)  # pre-RBAC token shape
    assert client.get("/api/bot/status", headers=_auth(legacy)).status_code == 200
    assert (
        client.post("/api/bot/stop", headers=_auth(legacy), json={}).status_code == 403
    )


def test_admin_can_mutate(client):
    admin = _token(role="admin")
    r = client.post("/api/bot/start", headers=_auth(admin), json={"mode": "paper"})
    assert r.status_code == 200
    r = client.post("/api/bot/stop", headers=_auth(admin), json={})
    assert r.status_code == 200


def test_admin_cannot_claim_live_mode(client):
    admin = _token(role="admin")

    response = client.post(
        "/api/bot/start",
        headers=_auth(admin),
        json={"mode": "live"},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Only paper mode is supported"}
    assert api_module.bot_state["running"] is False
    assert api_module.bot_state["mode"] == "paper"
    assert api_module.bot_state["start_time"] is None


def test_api_health_and_schema_report_the_python314_preview(client):
    assert client.get("/health").get_json()["version"] == "2.0.0a1"

    schema = client.get("/api/swagger.json").get_json()
    assert schema["info"]["version"] == "2.0.0a1"
    assert schema["components"]["schemas"]["BotStatus"]["properties"]["mode"][
        "enum"
    ] == ["paper"]
    assert schema["components"]["schemas"]["StartBotRequest"]["properties"]["mode"][
        "enum"
    ] == ["paper"]


def test_no_token_still_401(client):
    assert client.post("/api/bot/start", json={}).status_code == 401


def test_control_api_runner_is_local_and_environment_driven_by_default(monkeypatch):
    for name in ("API_HOST", "API_PORT", "API_WORKERS", "API_DEBUG"):
        monkeypatch.delenv(name, raising=False)

    defaults = api_runner._parser().parse_args([])
    assert defaults.host == "127.0.0.1"
    assert defaults.port == 5000
    assert defaults.workers == 1
    assert defaults.debug is False

    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "5100")
    monkeypatch.setenv("API_WORKERS", "3")
    monkeypatch.setenv("API_DEBUG", "true")
    configured = api_runner._parser().parse_args([])
    assert configured.host == "0.0.0.0"
    assert configured.port == 5100
    assert configured.workers == 3
    assert configured.debug is True
