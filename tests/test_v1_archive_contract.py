"""Safety contract for incremental ELVIS V1 retirement."""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[1]
_ARCHIVE_INDEX = _REPOSITORY / "docs" / "archive" / "v1" / "README.md"
_DIAGRAM = _REPOSITORY / "diagrams" / "v1-retirement-boundary"

_ARCHIVED_DOCUMENTS = {
    _REPOSITORY
    / "docs"
    / "RELEASE_NOTES.md": (
        _REPOSITORY / "docs" / "archive" / "v1" / "RELEASE_NOTES.md"
    ),
    _REPOSITORY
    / "docs"
    / "test_suite_fixes.md": (
        _REPOSITORY / "docs" / "archive" / "v1" / "test_suite_fixes.md"
    ),
    _REPOSITORY
    / "docs"
    / "bot_architecture_mermaid.md": (
        _REPOSITORY / "docs" / "archive" / "v1" / "bot_architecture_mermaid.md"
    ),
}

_ACTIVE_COMPATIBILITY_GUIDES = {
    _REPOSITORY / "docs" / "architecture.md",
    _REPOSITORY / "docs" / "COMPONENTS.md",
    _REPOSITORY / "docs" / "DEPLOYMENT.md",
    _REPOSITORY / "docs" / "ELVIS_SYSTEM_ARCHITECTURE.md",
    _REPOSITORY / "docs" / "PAPER_TRADING_SETUP.md",
    _REPOSITORY / "docs" / "APPLE_CONTAINER_SETUP.md",
    _REPOSITORY / "docs" / "APPLE_NATIVE_CONTAINER_GUIDE.md",
    _REPOSITORY / "docs" / "README_APPLE_CONTAINERS.md",
}

_VERSIONED_V2_MANIFESTS = {
    _REPOSITORY / "deploy" / "v2" / "bootstrap-stage-v1.example.json",
    _REPOSITORY / "deploy" / "v2" / "bootstrap-complete-v1.example.json",
    _REPOSITORY / "deploy" / "v2" / "cutover-preflight-v1.example.json",
    _REPOSITORY / "deploy" / "v2" / "legacy-snapshot-import-v1.example.json",
    (_REPOSITORY / "deploy" / "v2" / "legacy-snapshot-reconciliation-v1.example.json"),
}

_RETIRED_UNUSED_CONFIGS = {
    _REPOSITORY / "trading" / "config" / "data_config.yaml",
    _REPOSITORY / "trading" / "config" / "model_config.yaml",
    _REPOSITORY / "trading" / "config" / "risk_config.yaml",
}

_ACTIVE_YAML_CONFIGS = {
    _REPOSITORY / "trading" / "config" / "validation_config.yaml",
    _REPOSITORY / "training" / "config" / "model_config.yaml",
    _REPOSITORY / "trading_config.yaml",
}

_ACTIVE_CONFIG_FILES = {
    *_ACTIVE_YAML_CONFIGS,
    _REPOSITORY / "config" / "config.py",
}


def test_v1_archive_is_an_explicit_historical_allowlist() -> None:
    index = _ARCHIVE_INDEX.read_text(encoding="utf-8")
    expected_files = {_ARCHIVE_INDEX, *_ARCHIVED_DOCUMENTS.values()}
    assert {path for path in _ARCHIVE_INDEX.parent.iterdir() if path.is_file()} == (
        expected_files
    )

    for previous_path, archived_path in _ARCHIVED_DOCUMENTS.items():
        assert not previous_path.exists()
        assert archived_path.is_file()
        assert (
            "Historical V1 document" in archived_path.read_text(encoding="utf-8")[:800]
        )
        assert previous_path.relative_to(_REPOSITORY).as_posix() in index
        assert archived_path.name in index


def test_operational_compatibility_guides_remain_active_until_cutover() -> None:
    assert all(path.is_file() for path in _ACTIVE_COMPATIBILITY_GUIDES)


def test_versioned_v2_manifests_are_not_classified_as_v1_debris() -> None:
    for manifest in _VERSIONED_V2_MANIFESTS:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        assert document["schema_version"] == 1


def test_dead_plaintext_credential_copier_is_removed() -> None:
    helper = _REPOSITORY / "scripts" / "setup_secure_config.sh"
    assert not helper.exists()
    assert "setup_secure_config.sh" not in (
        _REPOSITORY / "scripts" / "README.md"
    ).read_text(encoding="utf-8")


def test_unused_v1_yaml_is_removed_without_touching_active_configs() -> None:
    import yaml

    assert all(not path.exists() for path in _RETIRED_UNUSED_CONFIGS)
    assert all(path.is_file() for path in _ACTIVE_CONFIG_FILES)
    assert all(
        isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
        for path in _ACTIVE_YAML_CONFIGS
    )

    data_docs = (_REPOSITORY / "docs" / "data_processing.md").read_text(
        encoding="utf-8"
    )
    trading_docs = (_REPOSITORY / "docs" / "trading_system.md").read_text(
        encoding="utf-8"
    )
    components = (_REPOSITORY / "docs" / "COMPONENTS.md").read_text(encoding="utf-8")
    assert "Data-processing configuration lives in" not in data_docs
    assert "Additional per-domain YAML lives under" not in trading_docs
    assert "training/config/model_config.yaml" in components

    validation_entrypoint = (
        _REPOSITORY / "trading" / "scripts" / "validate_strategy.py"
    ).read_text(encoding="utf-8")
    training_entrypoint = (_REPOSITORY / "training" / "train_models.py").read_text(
        encoding="utf-8"
    )
    assert 'default="trading/config/validation_config.yaml"' in validation_entrypoint
    assert 'default="training/config/model_config.yaml"' in training_entrypoint


def test_unused_galaxy_roles_are_not_vendored_or_declared() -> None:
    import yaml

    roles_directory = _REPOSITORY / "ansible" / "roles"
    assert not roles_directory.exists()
    requirements = yaml.safe_load(
        (_REPOSITORY / "ansible" / "requirements.yml").read_text(encoding="utf-8")
    )
    assert "roles" not in requirements
    assert {item["name"] for item in requirements["collections"]} == {
        "community.general",
        "community.docker",
    }
    setup = (_REPOSITORY / "ansible" / "run_setup.sh").read_text(encoding="utf-8")
    assert "ansible-galaxy collection install -r requirements.yml --force" in setup
    ignored = (_REPOSITORY / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "ansible/roles/" in ignored

    playbooks = (
        "playbook.yml",
        "docker_playbook.yml",
        "deploy_containers.yml",
        "test_playbook.yml",
    )
    role_syntax = re.compile(r"^\s*(roles|include_role|import_role)\s*:", re.MULTILINE)
    for name in playbooks:
        source = (_REPOSITORY / "ansible" / name).read_text(encoding="utf-8")
        assert role_syntax.search(source) is None

    retained_ansible = {
        _REPOSITORY / "ansible" / "ansible.cfg",
        _REPOSITORY / "ansible" / "inventory.yml",
        _REPOSITORY / "ansible" / "requirements.yml",
        _REPOSITORY / "ansible" / "run_setup.sh",
        _REPOSITORY / "ansible" / "templates" / "elvis-bot.service.j2",
        *(_REPOSITORY / "ansible" / name for name in playbooks),
    }
    assert all(path.is_file() for path in retained_ansible)
    archive = _ARCHIVE_INDEX.read_text(encoding="utf-8")
    assert "v0.3.0" in archive
    assert "git restore --source=v0.3.0" in archive


def test_v1_retirement_mermaid_source_and_render_set_are_complete() -> None:
    source = _DIAGRAM.with_suffix(".mmd").read_text(encoding="utf-8")
    archive_index = _ARCHIVE_INDEX.read_text(encoding="utf-8")
    fence = re.search(r"```mermaid\n(?P<source>.*?)```", archive_index, re.DOTALL)
    assert fence is not None
    assert fence.group("source").rstrip() == source.rstrip()

    svg = _DIAGRAM.with_suffix(".svg")
    png = _DIAGRAM.with_suffix(".png")
    excalidraw = _DIAGRAM.with_suffix(".excalidraw")
    assert "<svg" in svg.read_text(encoding="utf-8")[:1000]
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    scene = json.loads(excalidraw.read_text(encoding="utf-8"))
    assert scene["type"] == "excalidraw"
    assert scene["elements"]
