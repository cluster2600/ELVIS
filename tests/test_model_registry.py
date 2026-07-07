"""Tests for the model registry (version control + approval workflow)."""

import os

from core.models.model_registry import ModelRegistry


def _reg(tmp_path):
    return ModelRegistry(path=str(tmp_path / "registry.json"), clock=lambda: 123.0)


def _model(tmp_path, name, content=b"weights-v1"):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_register_creates_pending_version(tmp_path):
    reg = _reg(tmp_path)
    e = reg.register(_model(tmp_path, "m.joblib"), "rf", metrics={"f1": 0.7})
    assert e["version"] == 1 and e["status"] == "pending"
    assert e["metrics"] == {"f1": 0.7} and len(e["sha256"]) == 64
    assert reg.get_production("rf") is None  # pending is not production


def test_approve_promotes_to_production(tmp_path):
    reg = _reg(tmp_path)
    reg.register(_model(tmp_path, "m.joblib"), "rf")
    reg.approve("rf", 1)
    prod = reg.get_production("rf")
    assert prod is not None and prod["version"] == 1 and prod["status"] == "production"


def test_reject_excludes_from_production(tmp_path):
    reg = _reg(tmp_path)
    reg.register(_model(tmp_path, "m.joblib"), "rf")
    reg.reject("rf", 1)
    assert reg.get_production("rf") is None


def test_versions_increment_and_sha_changes(tmp_path):
    reg = _reg(tmp_path)
    reg.register(_model(tmp_path, "a.joblib", b"v1"), "rf")
    e2 = reg.register(_model(tmp_path, "b.joblib", b"v2-different"), "rf")
    versions = reg.list_versions("rf")
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[0]["sha256"] != versions[1]["sha256"]
    assert e2["version"] == 2


def test_latest_approved_wins(tmp_path):
    reg = _reg(tmp_path)
    reg.register(_model(tmp_path, "a.joblib", b"v1"), "rf")
    reg.register(_model(tmp_path, "b.joblib", b"v2"), "rf")
    reg.approve("rf", 1)
    reg.approve("rf", 2)
    assert reg.get_production("rf")["version"] == 2
