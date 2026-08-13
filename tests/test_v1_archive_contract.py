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
