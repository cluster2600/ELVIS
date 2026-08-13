"""Release contract for the Python 3.14-only ELVIS V2 distribution."""

import ast
import re
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_DIRECTORY_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "htmlcov",
    "node_modules",
}


def _project_metadata() -> dict:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _active_text_files():
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPOSITORY_ROOT)
        if any(part in IGNORED_DIRECTORY_PARTS for part in relative.parts):
            continue
        if relative.parts[:2] == ("docs", "archive"):
            continue
        if (
            path.suffix in ACTIVE_TEXT_SUFFIXES
            or path.name
            in {
                ".python-version",
                "Dockerfile",
                "Makefile",
            }
            or path.name.startswith("Dockerfile.")
        ):
            yield relative, path


def _module_version(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"No __version__ assignment found in {path}")


def test_release_metadata_is_python314_only() -> None:
    metadata = _project_metadata()

    assert metadata["project"]["version"] == "2.0.0a1"
    assert metadata["project"]["requires-python"] == ">=3.14,<3.15"
    assert metadata["project"]["classifiers"] == [
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.14",
    ]
    assert "tenacity" in metadata["project"]["dependencies"]
    assert "TA-Lib==0.7.1" in metadata["project"]["dependencies"]
    assert metadata["tool"]["black"]["target-version"] == ["py314"]
    assert metadata["tool"]["pytest"]["ini_options"]["filterwarnings"] == [
        "error::pytest.PytestReturnNotNoneWarning"
    ]
    assert (REPOSITORY_ROOT / ".python-version").read_text().strip() == "3.14"
    assert _module_version(REPOSITORY_ROOT / "trading/__init__.py") == "2.0.0a1"


def test_active_tree_has_no_retired_python_runtime_references() -> None:
    old_minor = "3" + "." + "10"
    old_black_target = "py" + "3" + "10"
    old_ml_suffix = "ml" + "3" + "10"
    retired = re.compile(
        rf"(?i)(?<![0-9.]){re.escape(old_minor)}(?![0-9.])"
        rf"|{old_black_target}|{old_ml_suffix}"
    )
    violations = []

    for relative, path in _active_text_files():
        if retired.search(relative.as_posix()):
            violations.append(relative.as_posix())
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if retired.search(text):
            violations.append(relative.as_posix())

    assert violations == [], f"Retired Python runtime references: {violations}"


def test_ci_uses_single_python314_jobs() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )

    assert "PYTHON_VERSION: '3.14'" in workflow
    assert "name: Run Tests (3.14)" in workflow
    assert "name: PostgreSQL 15 Integration (3.14)" in workflow
    assert "name: Python 3.14 ML Contracts" in workflow
    assert 'python -m pip install -e ".[test]"' in workflow
    assert '-e ".[ml,test]"' in workflow
    assert "import rich, seaborn, talib, trading" in workflow
    assert "TRIVY_DB_REPOSITORY: ghcr.io/aquasecurity/trivy-db:2" in workflow
    assert "matrix.python-version" not in workflow
    assert "name: Build Compatibility Paper Image" in workflow
    assert "name: Build V2 Operator Image (3.14, amd64/arm64)" in workflow
    assert "file: deploy/v2/operator.Dockerfile" in workflow
    assert "platforms: linux/amd64,linux/arm64" in workflow
    assert "ghcr.io/cluster2600/elvis" not in workflow
    assert "push: ${{" not in workflow


def test_training_scripts_require_python314_without_retired_backend_probe() -> None:
    shell_script = (REPOSITORY_ROOT / "scripts/run_training.sh").read_text(
        encoding="utf-8"
    )
    diagnostic = (REPOSITORY_ROOT / "scripts/debug_training.py").read_text(
        encoding="utf-8"
    )

    assert "command -v python3.14" in shell_script
    assert 'exec python3.14 -m training.train_models "$@"' in shell_script
    assert "python3 " not in shell_script
    assert "tensorflow" not in shell_script.lower()
    assert diagnostic.startswith("#!/usr/bin/env python3.14")
    assert "sys.version_info[:2] != (3, 14)" in diagnostic
    assert "tensorflow" not in diagnostic.lower()


def test_active_python_entrypoints_require_python314() -> None:
    entrypoints = [
        path
        for parent in ("core", "scripts", "tests", "trading", "training")
        for path in (REPOSITORY_ROOT / parent).rglob("*.py")
        if path.read_text(encoding="utf-8").startswith("#!")
    ]

    assert entrypoints
    assert all(
        path.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3.14\n")
        for path in entrypoints
    )
    api_wrapper = (REPOSITORY_ROOT / "scripts/run_api.sh").read_text(encoding="utf-8")
    assert 'exec python3.14 trading/scripts/run_api.py "$@"' in api_wrapper
    assert "pip install" not in api_wrapper
    assert 'python3.14 "$SCRIPT_DIR/run_dashboard.py"' in (
        REPOSITORY_ROOT / "trading/scripts/run_dashboard.sh"
    ).read_text(encoding="utf-8")


def test_compatibility_launcher_exposes_only_paper_mode() -> None:
    main_source = (REPOSITORY_ROOT / "main.py").read_text(encoding="utf-8")
    wrapper = (REPOSITORY_ROOT / "scripts/run_elvis.sh").read_text(encoding="utf-8")

    assert 'choices=["paper"]' in main_source
    assert 'if [[ "${2:-}" != "paper" ]]' in wrapper
    assert "exec python3.14 main.py --mode paper" in wrapper
    assert "pip install" not in wrapper
    assert "LIVE trading" not in wrapper


def test_ml_trainer_uses_the_resolvable_python314_extra() -> None:
    metadata = _project_metadata()
    project_dependencies = "\n".join(metadata["project"]["dependencies"]).lower()
    ml_dependencies = metadata["project"]["optional-dependencies"]["ml"]
    ml_dependencies_text = "\n".join(ml_dependencies).lower()
    compatibility_requirements = (
        (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    )
    trainer_requirements = (
        REPOSITORY_ROOT / "requirements/requirements_ml314.txt"
    ).read_text(encoding="utf-8")
    trainer_dockerfile = (REPOSITORY_ROOT / "docker/Dockerfile.ml314").read_text(
        encoding="utf-8"
    )
    compose = (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "tensorflow" not in project_dependencies
    assert "protobuf" not in project_dependencies
    assert "tensorflow" not in ml_dependencies_text
    assert "protobuf" not in compatibility_requirements
    assert ml_dependencies == [
        "torch==2.13.0",
        "gymnasium==1.3.0",
        "openai==3.0.0",
        "xgboost==3.4.0",
        "lightgbm==4.7.0",
    ]
    assert ".[ml]" in trainer_requirements.splitlines()
    assert trainer_dockerfile.startswith("# ML training container — Python 3.14")
    assert "FROM python:3.14-slim" in trainer_dockerfile
    assert "requirements_ml314.txt" in trainer_dockerfile
    assert "dockerfile: docker/Dockerfile.ml314" in compose
