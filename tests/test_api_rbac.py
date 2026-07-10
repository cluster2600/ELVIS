"""Tests for JWT role-based access control (RBAC) in trading/api/app.py.

Documented in SECURITY.md: the login token carries a ``role`` claim; mutating
endpoints require the ``admin`` role; tokens without a role claim degrade to
read-only ``viewer``. Runs with no live services (Flask test client only).
"""

import os
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("API_SECRET_KEY", "test-secret-key-for-rbac")

pytest.importorskip("flask")
pytest.importorskip("jwt")

import importlib  # noqa: E402

import jwt  # noqa: E402

api_module = importlib.import_module("trading.api.app")


@pytest.fixture
def client():
    api_module.app.config["TESTING"] = True
    with api_module.app.test_client() as c:
        yield c


def _token(role=None, user="tester"):
    payload = {"user": user, "exp": datetime.utcnow() + timedelta(hours=1)}
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


def test_no_token_still_401(client):
    assert client.post("/api/bot/start", json={}).status_code == 401
