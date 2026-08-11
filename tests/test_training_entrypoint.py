"""Contract tests for the vault-free training process wrapper."""

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_no_vault


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config="training/config/model_config.yaml",
        output=str(tmp_path / "models"),
        debug=False,
        limit=100,
        prediction_horizon=5,
        epochs=5,
    )


def test_training_environment_preserves_container_database_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.setenv("POSTGRES_USER", "elvis_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "elvis_password")
    monkeypatch.setenv("POSTGRES_DBNAME", "elvis_trading")
    monkeypatch.delenv("REDIS_HOST", raising=False)

    train_no_vault.setup_training_environment()

    assert train_no_vault.os.environ["POSTGRES_HOST"] == "postgres"
    assert train_no_vault.os.environ["POSTGRES_USER"] == "elvis_user"
    assert train_no_vault.os.environ["POSTGRES_PASSWORD"] == "elvis_password"
    assert train_no_vault.os.environ["POSTGRES_DBNAME"] == "elvis_trading"
    assert train_no_vault.os.environ["REDIS_HOST"] == "localhost"


@pytest.mark.parametrize("child_returncode", [0, 23])
def test_run_training_uses_repository_root_and_propagates_child_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    child_returncode: int,
) -> None:
    invocation = {}
    monkeypatch.setattr(train_no_vault, "setup_training_environment", lambda: None)
    monkeypatch.setattr(train_no_vault, "patch_imports", lambda: None)

    def fake_run(command, *, env, cwd):
        invocation.update(command=command, env=env, cwd=cwd)
        return SimpleNamespace(returncode=child_returncode)

    monkeypatch.setattr(train_no_vault.subprocess, "run", fake_run)

    status = train_no_vault.run_training(_args(tmp_path))

    repository_root = Path(train_no_vault.__file__).resolve().parent.parent
    assert status == child_returncode
    assert Path(invocation["cwd"]) == repository_root
    assert invocation["command"][1] == "training/train_models.py"


def test_main_returns_training_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(train_no_vault, "run_training", lambda _args: 17)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_no_vault.py", "--output", str(tmp_path / "models")],
    )

    assert train_no_vault.main() == 17


def test_shell_entrypoint_help_exits_successfully(tmp_path: Path) -> None:
    repository_root = Path(train_no_vault.__file__).resolve().parent.parent
    isolated_scripts = tmp_path / "repo" / "scripts"
    isolated_scripts.mkdir(parents=True)
    wrapper = isolated_scripts / "run_training.sh"
    shutil.copy2(repository_root / "scripts/run_training.sh", wrapper)

    result = subprocess.run(
        ["bash", str(wrapper), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "Cleanup complete" in result.stdout
    assert "python main.py --mode live" not in wrapper.read_text(encoding="utf-8")
