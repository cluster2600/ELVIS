"""Executable guards for the trajectory-B/1B production evidence contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DOCS = ROOT / "docs" / "architecture_migration"
PLAN = MIGRATION_DOCS / "05-v2-production-plan.md"
REGISTER = MIGRATION_DOCS / "06-v2-production-failure-register.md"
MATRIX = MIGRATION_DOCS / "07-v2-production-e2e-matrix.md"
INDEX = MIGRATION_DOCS / "README.md"
ALPHA2_COURSE = ROOT / "videos" / "elvis-v2-alpha2-operator-preview-course"
ROLLBACK_DIAGRAM = ROOT / "diagrams" / "v2-c3c2-cutover-rollback.mmd"
FRESH_CUTOVER = ROOT / "docs" / "V2_FRESH_TARGET_CUTOVER.md"

REQUIRED_GATE_SECTIONS = (
    "Preconditions",
    "Exact scenario",
    "Evidence to retain",
    "Pass / fail",
    "Rollback / containment",
    "Dependencies:",
    "Applicability:",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalised(path: Path) -> str:
    return re.sub(r"\s+", " ", _text(path)).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_documents_freeze_trajectory_b_and_async_scope() -> None:
    plan = _text(PLAN)
    normalised_plan = _normalised(PLAN)
    index = _text(INDEX)

    assert "trajectory B approved" in plan
    assert "selected **1B**" in plan
    assert "asynchronous virtual-venue lifecycle" in plan
    assert (
        "The release cannot reduce execution to an immediate full fill."
        in normalised_plan
    )
    assert "ACTIVE_PAPER_PRODUCTION" in plan
    assert "unresolved A/B/C" not in plan
    assert "unresolved A/B/C" not in index
    assert "G0 through\n   G17" in index


def test_every_g0_to_g17_gate_has_one_complete_b_only_contract() -> None:
    source = _text(MATRIX)
    matches = list(
        re.finditer(
            r"(?ms)^## (G\d+) — .*?(?=^## G\d+ — |^## Final acceptance record)",
            source,
        )
    )

    assert [match.group(1) for match in matches] == [f"G{i}" for i in range(18)]
    for match in matches:
        gate = match.group(0)
        for section in REQUIRED_GATE_SECTIONS:
            assert gate.count(f"**{section}**") == 1, (match.group(1), section)
        assert "**Applicability:** trajectory B only" in gate


def test_failure_register_is_open_and_covers_async_structural_blockers() -> None:
    source = _text(REGISTER)
    expected_ids = tuple(
        [f"P0-{index:03d}" for index in range(1, 16)]
        + [f"P1-{index:03d}" for index in range(1, 15)]
        + [f"P2-{index:03d}" for index in range(1, 4)]
    )
    heading_ids = tuple(re.findall(r"^### ELVIS-V2-(P[012]-\d{3}) — ", source, re.M))
    triage_ids = tuple(re.findall(r"^\| (P[012]-\d{3}) \|", source, re.M))

    assert heading_ids == expected_ids
    assert triage_ids == expected_ids
    assert len(set(heading_ids)) == len(heading_ids)
    assert len(set(triage_ids)) == len(triage_ids)
    assert source.count("- **Status:** OPEN") == 32
    for required in (
        "terminal full-fill only",
        "Open orders reserve no account capacity",
        "No durable causal venue-input/event transport or atomic event projector exists",
        "Recovery omits non-terminal orders",
        "Async queue retention, backpressure and operator SLOs are undefined",
    ):
        assert required in source


def test_authority_and_async_invariants_do_not_overload_counters() -> None:
    plan = _normalised(PLAN)
    matrix = _normalised(MATRIX)

    assert "`LEGACY/0/S0 -> PAUSED/0/S1`" in plan
    assert "`PAUSED/0/S1 -> ACTIVE/1/S2`" in plan
    assert "`runtime_generation` changes only on activation" in plan
    assert "pause does not advance activation generation" in matrix
    assert "separate gapless global venue-input and causal-event sequences" in plan
    assert "future fills are never pre-appended" in plan
    assert "consumes only `next_input`" in matrix
    assert "apply only `last_applied + 1`" in plan
    assert "kill switch blocks new risk" in plan
    assert "continue fenced ingest, settlement" in matrix.lower()


def test_activation_records_are_chronological_durable_and_recoverable() -> None:
    plan = _normalised(PLAN)
    matrix = _normalised(MATRIX)
    register = _normalised(REGISTER)

    assert "paused_backup_restore_receipt_sha256" in plan
    assert (
        "persists the canonical `activation-receipt-v1` bytes in the same "
        "transaction"
    ) in plan
    assert "lost acknowledgement can return identical bytes" in plan
    assert "same-candidate safe-redeploy manifest" in plan
    assert "canonical null PAUSED-backup field" in matrix
    assert "cannot be signed until the retirement receipt exists" in matrix
    assert "No reactivation approval is signed in advance" in matrix
    assert "twin-specific `PAUSED/0/S1` backup" in matrix
    assert "fresh quiescent twin `PAUSED/N/(S+1)` backup" in matrix
    assert (
        "every twin activation binds its applicable quiescent PAUSED "
        "backup/restore receipt"
    ) in matrix
    assert "persist the canonical receipt core in the same transaction" in register


def test_nonce_scope_and_safe_action_precedence_are_unambiguous() -> None:
    plan = _normalised(PLAN)
    matrix = _normalised(MATRIX)
    register = _normalised(REGISTER)

    nonce_scope = "`(trust_domain, signer_key_id, nonce)`"
    for source in (plan, matrix, register):
        assert nonce_scope in source
        assert "target-local" in source
        assert "cross-database" in source
    assert "another trust domain or signer key is a distinct namespace" in matrix
    assert "both targets remain `LEGACY/0/S0`" in register
    assert "sole-writer admission/activation can select only one" in matrix
    assert plan.index("`VENUE_WORKER_NOT_READY`") < plan.index("`RUNTIME_PAUSED`")
    assert "prevent accepted work from reaching finality" in plan


def test_alpha2_preflight_is_historical_evidence_and_never_a_b_seed() -> None:
    without_quote_markers = re.sub(r"(?m)^>\s?", "", _text(FRESH_CUTOVER))
    source = re.sub(r"\s+", " ", without_quote_markers).strip()

    assert "Historical alpha.2 preflight — superseded for production." in source
    for authority in (
        "05-v2-production-plan.md",
        "06-v2-production-failure-register.md",
        "07-v2-production-e2e-matrix.md",
    ):
        assert authority in source
    assert (
        "No V1 source state, V1 clone, or c3c2/c3c3 import output may seed the "
        "trajectory-B production opening or account."
    ) in source
    assert "a stopped V1 clone is pre-retirement read-only evidence only" in source
    assert "V1 is never a writer or rollback authority" in source
    assert "## Planned production path and rollback" not in _text(FRESH_CUTOVER)


def test_alpha2_course_is_historical_and_cannot_render_or_publish() -> None:
    assert ALPHA2_COURSE.is_dir()
    assert not (ROOT / "videos" / "elvis-v2-production-course").exists()

    brief = _normalised(ALPHA2_COURSE / "BRIEF.md")
    meta = json.loads(_text(ALPHA2_COURSE / "meta.json"))

    assert "release_tag: v2.0.0-alpha.2" in brief
    assert "status: preview-documentation-only" in brief
    assert "aucun runtime" in brief
    assert "approuvé explicitement" in brief
    assert meta["id"] == "elvis-v2-alpha2-operator-preview-course"
    assert "historical" in meta["name"].lower()
    for retired_scaffold in (
        "index.html",
        "package.json",
        "package-lock.json",
        "hyperframes.json",
        "node_modules",
    ):
        assert not (ALPHA2_COURSE / retired_scaffold).exists()
    videos_index = _text(ROOT / "videos" / "README.md")
    assert "historical documentation" in videos_index
    assert "future trajectory-B/1B production course" in videos_index


def test_alpha2_capture_manifest_matches_the_safe_protocol() -> None:
    manifest = json.loads(_text(ALPHA2_COURSE / "capture-manifest.template.json"))
    captures = {capture["shot_id"]: capture for capture in manifest["captures"]}
    protocol = _text(ALPHA2_COURSE / "capture-session-plan.md")

    assert manifest["course_id"] == "elvis-v2-alpha2-operator-preview-course"
    assert manifest["final_render_allowed"] is False
    assert manifest["publication_allowed"] is False
    assert manifest["expected_capture_count"] == len(manifest["captures"]) == 14
    assert len(captures) == len(manifest["captures"])
    assert set(captures) == {f"SS{i:02d}" for i in range(1, 15)}
    assert protocol.count("```bash") == 1
    assert "single executable command source" in protocol
    assert len(manifest["required_session_setup_commands"]) == 1
    setup = manifest["required_session_setup_commands"][0]
    assert "BASELINE_CONTAINERS_SHA256" in setup
    assert "LC_ALL=C sort | sha256sum" in setup
    for resource_query in (
        "docker ps -aq --filter",
        "docker network ls -q --filter",
        "docker volume ls -q --filter",
    ):
        assert resource_query in setup
    assert setup.count('label=com.docker.compose.project="$PROJECT"') == 3
    assert setup.index("docker ps -aq --filter") < setup.index(
        "BASELINE_CONTAINERS_SHA256"
    )
    for shot_id in (f"SS{i:02d}" for i in range(2, 15)):
        assert captures[shot_id]["exact_command"]
    assert "--check --strict" in captures["SS04"]["exact_command"]
    assets = captures["SS03"]["exact_command"]
    assert "elvis-v2-operator-v2.0.0-alpha.2-linux-amd64.spdx.json" in assets
    assert "elvis-v2-operator-v2.0.0-alpha.2-linux-arm64.spdx.json" in assets
    assert 'test "$actual" = "$expected"' in assets
    assert "LC_ALL=C sort" in assets
    attestation = captures["SS05"]["exact_command"]
    for strict_argument in (
        "--bundle",
        "--source-digest",
        "--source-ref",
        "--signer-workflow",
        "--deny-self-hosted-runners",
    ):
        assert strict_argument in attestation
    archive_audit = captures["SS06"]["exact_command"]
    assert "tar -tzf" in archive_audit
    assert "tar -tvzf" in archive_audit
    assert 'test "$actual" = "$expected"' in archive_audit
    assert "bootstrap-complete-v1.example.json" in archive_audit
    assert "pg_service.preview.conf.example" in archive_audit
    extraction = captures["SS07"]["exact_command"]
    assert 'test ! -e "$BUNDLE"' in extraction
    assert "--no-same-owner" in extraction
    assert "--no-same-permissions" in extraction
    assert "sha256sum --check --strict SHA256SUMS" in extraction
    assert manifest["image_digest"] in extraction
    assert 'test "$IMAGE" = "$expected_image"' in extraction
    assert 'test "$outer_digest" = "$expected_image"' in extraction
    assert 'test "$inner_digest" = "$expected_image"' in extraction
    assert 'test "$outer_digest" = "$inner_digest"' in extraction
    compose_validation = captures["SS08"]["exact_command"]
    assert "install -m 0600" in compose_validation
    assert "install -d -m 0700" in compose_validation
    assert 'find "$OPERATOR_DIR" -mindepth 1 -print -quit' in compose_validation
    assert "config --services" in compose_validation
    assert "config --images" in compose_validation
    assert 'test "$services" = operator' in compose_validation
    assert 'test "$images" = "$IMAGE"' in compose_validation
    public_pull = captures["SS09"]["exact_command"]
    assert "docker image inspect" in public_pull
    assert "DOCKER_CONFIG_DIR" in public_pull
    assert "mktemp -d" in public_pull
    assert 'DOCKER_CONFIG="$DOCKER_CONFIG_DIR"' in public_pull
    assert "cleanup_docker_config" in public_pull
    assert "DOCKER_AUTH_CONFIG" in public_pull
    assert "REGISTRY_AUTH_FILE" in public_pull
    for shot_id in ("SS08", "SS09", "SS13", "SS14"):
        command = captures[shot_id]["exact_command"]
        assert '--project-name "$PROJECT"' in command
        assert 'ELVIS_V2_OPERATOR_IMAGE="$IMAGE"' in command
        assert 'ELVIS_V2_OPERATOR_DIR="$OPERATOR_DIR"' in command
        assert "ELVIS_V2_OPERATOR_UID=65532" in command
        assert "ELVIS_V2_OPERATOR_GID=65532" in command
        assert "$(cat IMAGE_DIGEST.txt)" not in command
    for shot_id in ("SS10", "SS11", "SS12"):
        command = captures[shot_id]["exact_command"]
        assert "docker run --rm --network none --read-only" in command
        assert "--cap-drop ALL" in command
        assert "no-new-privileges" in command
    hardened_surface = captures["SS13"]["exact_command"]
    assert "config --format json" in hardened_surface
    assert "python3.14" not in hardened_surface
    assert "--entrypoint python" in hardened_surface
    assert "docker run --rm -i --network none --read-only" in hardened_surface
    assert "environment" not in hardened_surface
    assert 'user=="65532:65532"' in hardened_surface
    assert 'EXPECTED_OPERATOR_DIR="$OPERATOR_DIR"' in hardened_surface
    assert (
        'volume.get("source")==os.environ["EXPECTED_OPERATOR_DIR"]' in hardened_surface
    )
    assert 'service.get("read_only") is True' in hardened_surface
    assert 'service.get("cap_drop")==["ALL"]' in hardened_surface
    assert 'service.get("security_opt")==["no-new-privileges:true"]' in hardened_surface
    assert 'service.get("pids_limit")==64' in hardened_surface
    assert 'service.get("tmpfs")==["/tmp:size=16m,mode=1777"]' in hardened_surface
    assert 'assert not service.get("ports")' in hardened_surface
    assert "assert len(volumes)==1" in hardened_surface
    assert "volume_types" in hardened_surface
    assert "volume_targets" in hardened_surface
    assert 'volume.get("type")=="bind"' in hardened_surface
    assert 'volume.get("target")=="/run/operator"' in hardened_surface
    assert 'volume.get("read_only") is True' in hardened_surface
    assert 'volume["source"]' not in hardened_surface
    parser_match = re.search(
        r"""--entrypoint python "\$IMAGE" -c '(.*)'$""",
        hardened_surface,
    )
    assert parser_match is not None
    parser = parser_match.group(1)
    compile(parser, "<ss13-compose-guard>", "exec")
    valid_service = {
        "user": "65532:65532",
        "read_only": True,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "pids_limit": 64,
        "tmpfs": ["/tmp:size=16m,mode=1777"],
        "volumes": [
            {
                "type": "bind",
                "source": "/private/not-printed",
                "target": "/run/operator",
                "read_only": True,
            }
        ],
    }

    def run_parser(service: dict[str, object]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", parser],
            input=json.dumps({"services": {"operator": service}}),
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "EXPECTED_OPERATOR_DIR": "/private/not-printed"},
        )

    valid_result = run_parser(valid_service)
    assert valid_result.returncode == 0
    assert "/private/not-printed" not in valid_result.stdout
    invalid_variants = (
        {**valid_service, "user": "0:0"},
        {**valid_service, "user": "00:00"},
        {**valid_service, "read_only": False},
        {**valid_service, "cap_drop": []},
        {**valid_service, "security_opt": []},
        {**valid_service, "pids_limit": 65},
        {**valid_service, "tmpfs": ["/tmp"]},
        {**valid_service, "ports": ["5050:5050"]},
        {**valid_service, "volumes": []},
        {
            **valid_service,
            "volumes": [
                {
                    "type": "bind",
                    "source": "/private/ambient-override",
                    "target": "/run/operator",
                    "read_only": True,
                }
            ],
        },
        {
            **valid_service,
            "volumes": [
                {
                    "type": "bind",
                    "source": "/private/not-printed",
                    "target": "/run/operator",
                    "read_only": False,
                }
            ],
        },
    )
    assert all(run_parser(service).returncode != 0 for service in invalid_variants)
    cleanup = captures["SS14"]["exact_command"]
    for expected in ("containers=%s", "networks=%s", "volumes=%s"):
        assert expected in cleanup
    assert "unrelated_stack_hash=unchanged" in cleanup


def test_existing_media_records_are_not_misrepresented_as_public_screenshots() -> None:
    screenshot = json.loads(_text(ALPHA2_COURSE / "captures" / "SS01.json"))
    transcript = json.loads(_text(ALPHA2_COURSE / "captures" / "KALI-SESSION-01.json"))

    assert screenshot["shot_id"] == "SS01"
    assert screenshot["publication_allowed"] is False
    assert transcript["screenshot_derivatives_created"] is False
    assert transcript["accepted_for_capture_manifest"] is False
    assert transcript["command_contract_status"] == "not-bound-recapture-required"
    assert transcript["publication_allowed"] is False
    assert transcript["database_contact"] is False
    assert transcript["runtime_daemon_started"] is False
    assert transcript["authority_changed"] is False
    assert not (ROOT / "images" / "elvis.png").exists()


def test_production_authority_docs_forbid_any_return_to_legacy() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "V2_ARCHITECTURE.md",
        MIGRATION_DOCS / "04-migration-roadmap.md",
        PLAN,
        REGISTER,
        MATRIX,
        ROOT / "docs" / "V2_FRESH_TARGET_CUTOVER.md",
    )
    forbidden_transition = re.compile(r"(?:ACTIVE|PAUSED)\s*(?:-->|->)\s*LEGACY")

    for document in documents:
        source = _text(document)
        assert forbidden_transition.search(source) is None, document

    diagram = _text(ROLLBACK_DIAGRAM)
    mermaid_fences = re.findall(r"```mermaid\n(.*?)```", _text(FRESH_CUTOVER), re.S)
    embedded_diagram = diagram.rstrip() + "\n"
    assert embedded_diagram in mermaid_fences
    assert mermaid_fences[-1] == embedded_diagram
    assert mermaid_fences.count(embedded_diagram) == 1
    edges: dict[str, set[str]] = {}
    for source, target in re.findall(
        r"(?m)^\s*([A-Z][A-Z0-9_]*)\s*-->" r"(?:\|[^|]*\|)?\s*([A-Z][A-Z0-9_]*)",
        diagram,
    ):
        edges.setdefault(source, set()).add(target)
    reachable = {"ACTIVE"}
    pending = ["ACTIVE"]
    while pending:
        source = pending.pop()
        for target in edges.get(source, ()):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    assert "LEGACY" not in reachable
    assert "select one writer" not in diagram
    for suffix in ("svg", "png", "excalidraw"):
        artifact = ROLLBACK_DIAGRAM.with_suffix(f".{suffix}")
        assert artifact.stat().st_size > 0

    expected_artifact_set = {
        "mmd": "e3271349f511fd5084fd84e5957871d348b5f36432dcfef164b7fb66e8a973bb",
        "svg": "d3e017a146c9b003dffeb65de471b0c6ed01d6ecae63bae4dac9bbc8f6dc722d",
        "png": "1c05fb57e3382082c4fa994317fb32d9b60630dc020f539437c8dca934bf3296",
        "excalidraw": "5e6c4400bcdcf5fa749805e36ce7560a42d50a55799b34282f271c46d6ffa5d6",
    }
    assert {
        suffix: _sha256(ROLLBACK_DIAGRAM.with_suffix(f".{suffix}"))
        for suffix in expected_artifact_set
    } == expected_artifact_set

    node_labels = {
        match.group(1): re.sub(r"\s+", " ", match.group(2) or match.group(3)).strip()
        for match in re.finditer(
            r'\b([A-Z][A-Z0-9_]*)\s*(?:\["([^"]+)"\]|\{"([^"]+)"\})',
            diagram,
        )
    }
    assert set(node_labels) == {
        "ACTIVE",
        "APPROVE",
        "CLONE",
        "EARLY",
        "IMPORT",
        "LEGACY",
        "PAUSE",
        "PREF",
        "RECEIPT",
        "REAPPROVE",
        "RECOVER",
        "REVIEW",
        "VALIDATE",
    }
    edge_labels = {
        re.sub(r"\s+", " ", label).strip()
        for label in re.findall(r'-->\|"([^"]+)"\|', diagram)
    }
    assert edge_labels == {
        "approved later",
        "failure",
        "granted",
        "not granted",
        "pass",
        "reject or stale",
        "rollback",
        "success",
    }
    diagram_labels = set(node_labels.values()) | edge_labels
    assert all("<br" not in label.lower() for label in diagram_labels)

    svg_source = _text(ROLLBACK_DIAGRAM.with_suffix(".svg"))
    svg_root = ET.fromstring(svg_source)
    svg_text = " ".join(svg_root.itertext())
    assert svg_root.attrib["aria-labelledby"] == "rollback-title rollback-desc"
    assert "signed V1 retirement" in svg_text
    assert "no path returns to legacy authority" in svg_text
    svg_compact = re.sub(r"[^a-z0-9]+", "", svg_text.lower())
    for label in diagram_labels:
        compact_label = re.sub(r"[^a-z0-9]+", "", label.lower())
        assert compact_label in svg_compact, label

    excalidraw_source = _text(ROLLBACK_DIAGRAM.with_suffix(".excalidraw"))
    excalidraw = json.loads(excalidraw_source)
    excalidraw_labels = {
        re.sub(r"\s+", " ", element.get("originalText") or element["text"]).strip()
        for element in excalidraw["elements"]
        if element.get("type") == "text"
    }
    assert diagram_labels <= excalidraw_labels
    assert "<br" not in diagram.lower()
    assert "<br" not in svg_source.lower()
    assert "<br" not in excalidraw_source.lower()
    assert "select one writer" not in svg_source
    assert "select one writer" not in excalidraw_source

    root_index = _text(ROOT / "README.md")
    docs_index = _text(ROOT / "docs" / "README.md")
    for authoritative in (
        "05-v2-production-plan.md",
        "06-v2-production-failure-register.md",
        "07-v2-production-e2e-matrix.md",
    ):
        assert authoritative in root_index
        assert authoritative in docs_index
