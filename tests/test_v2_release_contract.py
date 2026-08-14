from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import v2_operator

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
DOCKERFILE = ROOT / "deploy" / "v2" / "operator.Dockerfile"
COMPOSE = ROOT / "deploy" / "v2" / "compose.preview.yml"
PREVIEW_ENV = ROOT / "deploy" / "v2" / "v2-preview.env.example"

RELEASE_TAG = "v2.0.0-alpha.2"
PROJECT_VERSION = "2.0.0a2"
IMAGE_VERSION = "2.0.0-alpha.2"
IMAGE = "ghcr.io/cluster2600/elvis-v2-operator"
COMMANDS = {
    "bootstrap": "scripts.postgres_bootstrap",
    "cutover-preflight": "scripts.postgres_cutover_preflight",
    "import-snapshot": "scripts.postgres_legacy_snapshot_import",
    "reconcile-snapshot": "scripts.postgres_legacy_snapshot_reconciliation",
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_project_and_release_versions_are_exactly_mapped() -> None:
    project = tomllib.loads(_text(ROOT / "pyproject.toml"))["project"]
    workflow = _text(WORKFLOW)

    assert project["version"] == PROJECT_VERSION
    assert project["requires-python"] == ">=3.14,<3.15"
    assert v2_operator._VERSION == IMAGE_VERSION
    assert f"V2_RELEASE_TAG: {RELEASE_TAG}" in workflow
    assert f"V2_PROJECT_VERSION: {PROJECT_VERSION}" in workflow
    assert f"V2_IMAGE_VERSION: {IMAGE_VERSION}" in workflow


def test_alpha2_version_is_consistent_across_install_surfaces() -> None:
    expected_references = {
        "INSTALL_V2.md": (f"`{RELEASE_TAG}`", f"TAG={RELEASE_TAG}"),
        "RELEASE_NOTES.md": (
            f"# ELVIS V2 operator preview — {RELEASE_TAG}",
            f"{IMAGE}:{IMAGE_VERSION}",
        ),
        "CHANGELOG.md": (f"## {RELEASE_TAG} —",),
        "deploy/v2/compose.preview.yml": (f"{IMAGE}:{IMAGE_VERSION}",),
        "deploy/v2/v2-preview.env.example": (f"{IMAGE}:{IMAGE_VERSION}",),
    }

    for relative, references in expected_references.items():
        content = _text(ROOT / relative)
        for reference in references:
            assert reference in content, f"{reference!r} missing from {relative}"

    non_historical_surfaces = (
        WORKFLOW,
        ROOT / "pyproject.toml",
        ROOT / "trading/__init__.py",
        ROOT / "scripts/v2_operator.py",
        COMPOSE,
        PREVIEW_ENV,
        ROOT / "INSTALL_V2.md",
    )
    for path in non_historical_surfaces:
        content = _text(path)
        assert "2.0.0a1" not in content, path
        assert "2.0.0-alpha.1" not in content, path


def test_dispatcher_exposes_only_bounded_operator_commands() -> None:
    assert {
        name: module for name, (module, _) in v2_operator._COMMANDS.items()
    } == COMMANDS

    source = _text(ROOT / "scripts" / "v2_operator.py")
    for forbidden in ("main.py", "run_elvis", "live", "activate"):
        assert f'"{forbidden}"' not in source


def test_dispatcher_help_is_dependency_free_and_explicitly_no_go() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.v2_operator", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "ACTIVE NO-GO" in result.stdout
    assert "Paper/migration preview only" in result.stdout
    for command in COMMANDS:
        assert command in result.stdout


def test_dispatcher_forwards_arguments_to_one_exact_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_import(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            main=lambda argv: calls.append((name, list(argv))) or 23,
        )

    monkeypatch.setattr(v2_operator.importlib, "import_module", fake_import)

    exit_code = v2_operator.main(
        ["import-snapshot", "--config", "/run/operator/import.json"]
    )

    assert exit_code == 23
    assert calls == [
        (
            "scripts.postgres_legacy_snapshot_import",
            ["--config", "/run/operator/import.json"],
        )
    ]


def test_operator_image_is_python_314_source_only_and_non_runtime() -> None:
    dockerfile = _text(DOCKERFILE)

    assert re.search(r"^FROM python:3\.14-slim@sha256:[0-9a-f]{64}$", dockerfile, re.M)
    assert 'ENTRYPOINT ["python", "-m", "scripts.v2_operator"]' in dockerfile
    assert 'CMD ["--help"]' in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert 'org.elvis.v2.active="NO-GO"' in dockerfile
    assert 'org.elvis.v2.scope="paper-migration-preview"' in dockerfile
    assert "COPY trading ./trading" not in dockerfile
    assert "COPY trading/application/__init__.py" not in dockerfile
    assert "COPY trading/domain/__init__.py" not in dockerfile
    assert "COPY scripts ./scripts" not in dockerfile
    assert "pip install ." not in dockerfile
    assert "pyproject.toml" not in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--only-binary=:all:" in dockerfile
    assert "main.py" not in dockerfile
    assert "run_elvis" not in dockerfile
    assert "COPY LICENSE /licenses/ELVIS-LICENSE" in dockerfile
    assert 'org.opencontainers.image.licenses="BTC_BOT"' in dockerfile
    for forbidden_module in (
        "paper_bot.py",
        "paper_runtime_activation.py",
        "binance_executor.py",
        "exchange_manager.py",
    ):
        assert forbidden_module not in dockerfile

    for module in COMMANDS.values():
        script_name = module.rsplit(".", 1)[-1] + ".py"
        assert f"scripts/{script_name}" in dockerfile
    assert "scripts/v2_opening_plan.py" in dockerfile
    assert "trading/application/fresh_opening.py" in dockerfile

    requirements = _text(ROOT / "deploy" / "v2" / "requirements.operator.txt")
    assert requirements.count("psycopg2-binary==2.9.12") == 1
    assert requirements.count("cryptography==50.0.0") == 1
    assert requirements.count("cffi==2.0.0") == 1
    assert requirements.count("pycparser==3.0") == 1
    assert requirements.count("--hash=sha256:") == 7


def test_preview_compose_pulls_exact_image_and_is_hardened() -> None:
    compose = _text(COMPOSE)
    preview_env = _text(PREVIEW_ENV)

    exact_reference = f"{IMAGE}:{IMAGE_VERSION}"
    assert exact_reference in compose
    assert f"ELVIS_V2_OPERATOR_IMAGE={exact_reference}" in preview_env
    assert "build:" not in compose
    assert "read_only: true" in compose
    assert 'cap_drop: ["ALL"]' in compose
    assert 'security_opt: ["no-new-privileges:true"]' in compose
    assert "65532" in compose
    assert "/var/run/docker.sock" not in compose
    assert "PGSERVICEFILE: /run/operator/pg_service.conf" in compose
    assert "PGPASSFILE: /run/operator/pgpass" in compose


def test_preview_service_file_requires_verified_tls_for_every_service() -> None:
    service_file = _text(ROOT / "deploy/v2/pg_service.preview.conf.example")
    sections = re.findall(r"(?m)^\[[^]]+\]$", service_file)

    assert sections == [
        "[elvis_v2_admin]",
        "[elvis_v2_migrator]",
        "[elvis_v2_readiness]",
        "[elvis_v2_trainer]",
        "[elvis_source_clone]",
        "[elvis_fresh_target_admin]",
        "[elvis_fresh_target_migrator]",
        "[elvis_fresh_target_readiness]",
    ]
    assert service_file.count("sslmode=verify-full") == len(sections)
    assert service_file.count("sslrootcert=/run/operator/ca.crt") == len(sections)
    assert "sslmode=prefer" not in service_file
    assert "sslmode=disable" not in service_file


def test_release_workflow_uses_only_commit_pinned_actions() -> None:
    workflow = _text(WORKFLOW)
    action_uses = re.findall(r"^\s*uses:\s*([^\s#]+)", workflow, re.M)

    assert action_uses
    for action_use in action_uses:
        action, separator, ref = action_use.rpartition("@")
        assert action and separator
        assert re.fullmatch(r"[0-9a-f]{40}", ref), action_use


def test_release_gate_requires_exact_main_sha_and_successful_ci() -> None:
    workflow = _text(WORKFLOW)

    assert f"- {RELEASE_TAG}" in workflow
    assert f"V2_PROJECT_VERSION: {PROJECT_VERSION}" in workflow
    assert '[[ "${tagged_commit}" == "${main_commit}" ]]' in workflow
    assert 'event_commit="$(git rev-parse "${GITHUB_SHA}^{commit}")"' in workflow
    assert '[[ "${tagged_commit}" == "${event_commit}" ]]' in workflow
    assert "commit=%s\\n" in workflow
    assert "release-commit: ${{ steps.release-identity.outputs.commit }}" in workflow
    assert "actions/workflows/ci.yml/runs" in workflow
    assert '-f head_sha="${RELEASE_COMMIT}"' in workflow
    assert ".head_sha == env.RELEASE_COMMIT" in workflow
    assert '.head_branch == "main"' in workflow
    assert '.event == "push"' in workflow
    assert '.name == "CI/CD Pipeline"' in workflow
    assert '.conclusion == "success"' in workflow
    assert re.search(r"publish-image:.*?\n\s+needs: verify\n", workflow, re.S)


def test_release_build_uses_ephemeral_candidate_then_promotes_after_gates() -> None:
    workflow = _text(WORKFLOW)
    publish = workflow[
        workflow.index("  publish-image:\n") : workflow.index("  scan-image:\n")
    ]
    release = workflow[workflow.index("  release:\n") :]
    tags_match = re.search(
        r"^\s{10}tags: \|\n(?P<tags>^\s{12}.+\n)",
        publish,
        re.M,
    )

    assert tags_match is not None
    tags = {line.strip() for line in tags_match.group("tags").splitlines()}
    assert tags == {
        "${{ env.V2_IMAGE }}:candidate-${{ github.run_id }}-${{ github.run_attempt }}",
    }
    assert (
        "org.opencontainers.image.revision="
        "${{ needs.verify.outputs.release-commit }}" in workflow
    )
    assert "docker/metadata-action" not in workflow
    assert "pattern={{major}}" not in workflow
    assert "pattern={{minor}}" not in workflow
    assert "ghcr.io/cluster2600/elvis\n" not in workflow
    assert ":sha-${{ needs.verify.outputs.release-commit }}" not in publish
    assert 'versioned_ref="${V2_IMAGE}:${V2_IMAGE_VERSION}"' in release
    assert 'commit_ref="${V2_IMAGE}:sha-${RELEASE_COMMIT}"' in release
    assert "promote_or_verify()" in release
    assert 'promote_or_verify "${commit_ref}"' in release
    assert 'promote_or_verify "${versioned_ref}"' in release
    assert "docker buildx imagetools create" in release
    assert '[[ "${existing_digest}" == "${IMAGE_DIGEST}" ]]' in release
    assert release.index("docker buildx imagetools create") < release.index(
        'gh release create "${V2_RELEASE_TAG}"'
    )


def test_release_promotion_creates_only_after_explicit_registry_absence() -> None:
    workflow = _text(WORKFLOW)
    release = workflow[workflow.index("  release:\n") :]
    promotion = release[
        release.index("          promote_or_verify() {\n") : release.index(
            '          promote_or_verify "${commit_ref}"\n'
        )
    ]

    assert "--write-out '%{http_code}'" in promotion
    assert 'case "${status}" in' in promotion
    assert "|| true" not in promotion

    status_case = re.search(
        r'^            case "\$\{status\}" in\n(?P<body>.*?)^            esac$',
        promotion,
        re.M | re.S,
    )
    assert status_case is not None

    branches = {}
    for label in ("200", "404", "*"):
        branch = re.search(
            rf"^              {re.escape(label)}\)\n"
            rf"(?P<body>.*?)"
            rf"^                ;;$",
            status_case.group("body"),
            re.M | re.S,
        )
        assert branch is not None, f"missing registry status branch {label}"
        branches[label] = branch.group("body")

    create = "docker buildx imagetools create"
    assert create not in branches["200"]
    assert branches["200"].index('[[ -n "${existing_digest}" ]]') < branches[
        "200"
    ].index('[[ "${existing_digest}" == "${IMAGE_DIGEST}" ]]')
    assert branches["404"].count(create) == 1
    assert create not in branches["*"]
    assert "return 1" in branches["*"]
    assert promotion.count(create) == 1

    post_case = promotion[promotion.index("            esac") :]
    assert 'docker buildx imagetools inspect "${target_ref}"' in post_case
    assert '[[ "${promoted_digest}" == "${IMAGE_DIGEST}" ]]' in post_case


def test_release_build_smokes_and_evidence_precede_release_creation() -> None:
    workflow = _text(WORKFLOW)

    assert "platforms: linux/amd64,linux/arm64" in workflow
    assert "for platform in linux/amd64 linux/arm64" in workflow
    assert (
        "docker compose --env-file .env -f compose.preview.yml config --quiet"
        in workflow
    )
    assert "docker compose --env-file .env -f compose.preview.yml pull" in workflow
    assert 'export ELVIS_V2_OPERATOR_IMAGE="${V2_IMAGE}@${IMAGE_DIGEST}"' in workflow
    assert '== "${V2_IMAGE}@${IMAGE_DIGEST}"' in workflow
    assert "sbom: true" in workflow
    assert "provenance: mode=max" in workflow
    assert "subject-digest: ${{ steps.build.outputs.digest }}" in workflow
    assert "subject-checksums: dist/ATTESTED_SHA256SUMS" in workflow
    assert "SHA256SUMS" in workflow
    assert ".spdx.json" in workflow
    assert ".intoto.jsonl" in workflow
    assert (
        "image-ref: ${{ env.V2_IMAGE }}@${{ needs.publish-image.outputs.image-digest }}"
        in workflow
    )
    assert 'exit-code: "1"' in workflow
    assert "severity: HIGH,CRITICAL" in workflow
    assert "TRIVY_DB_REPOSITORY: ghcr.io/aquasecurity/trivy-db:2" in workflow
    assert "platform: [linux/amd64, linux/arm64]" in workflow
    assert "TRIVY_PLATFORM: ${{ matrix.platform }}" in workflow
    smoke_job = workflow[workflow.index("  smoke:\n") : workflow.index("  assets:\n")]
    assert "docker/login-action" not in smoke_job
    assert "docker logout ghcr.io" in smoke_job
    assert "anonymously on amd64 and arm64" in smoke_job
    public_gate = workflow.index(
        "Require an anonymously pullable public bootstrap image"
    )
    publish_job = workflow.index("  publish-image:\n")
    assert public_gate < publish_job
    verify_job = workflow[:publish_job]
    assert "packages: read" not in verify_job
    assert "users/cluster2600/packages/container/elvis-v2-operator" not in verify_job
    assert 'bootstrap_ref="${V2_IMAGE}:visibility-bootstrap"' in verify_job
    assert 'docker pull "${bootstrap_ref}"' in verify_job
    assert verify_job.index("docker logout ghcr.io") < verify_job.index(
        'docker pull "${bootstrap_ref}"'
    )
    assert "SYFT_PLATFORM: linux/amd64" in workflow
    assert "SYFT_PLATFORM: linux/arm64" in workflow
    assert "-linux-amd64.spdx.json" in workflow
    assert "-linux-arm64.spdx.json" in workflow
    assert "${V2_RELEASE_TAG}.spdx.json" not in workflow
    assert (
        '"elvis-v2-operator-${V2_RELEASE_TAG}.intoto.jsonl" \\\n'
        "            IMAGE_DIGEST.txt \\\n"
        "            > SHA256SUMS" in workflow
    )

    release_needs = re.search(
        r"^  release:\n(?:.*\n)*?    needs: \[(?P<needs>[^]]+)\]",
        workflow,
        re.M,
    )
    assert release_needs is not None
    assert {item.strip() for item in release_needs.group("needs").split(",")} == {
        "verify",
        "publish-image",
        "scan-image",
        "smoke",
        "assets",
    }
    assert workflow.count('gh release create "${V2_RELEASE_TAG}"') == 1
    assert workflow.index("  release:\n") < workflow.index(
        'gh release create "${V2_RELEASE_TAG}"'
    )
    assert "--prerelease" in workflow
    assert "--latest=false" in workflow
    assert "--verify-tag" in workflow


def test_multiarch_smoke_evicts_same_digest_between_platform_pulls() -> None:
    workflow = _text(WORKFLOW)
    smoke = workflow[
        workflow.index(
            "      - name: Import-smoke all four commands anonymously on amd64 and arm64"
        ) : workflow.index("      - name: Smoke a clean-directory Compose install")
    ]

    assert 'image_ref="${V2_IMAGE}@${IMAGE_DIGEST}"' in smoke
    assert "for platform in linux/amd64 linux/arm64; do" in smoke
    assert 'docker pull --platform "${platform}" "${image_ref}"' in smoke

    cleanup = re.search(
        r"cleanup_smoke_image\(\) \{\n" r"(?P<body>.*?)" r"^          \}",
        smoke,
        re.M | re.S,
    )
    assert cleanup is not None
    assert 'docker image rm "${image_ref}"' in cleanup.group("body")
    assert "|| true" in cleanup.group("body")
    assert smoke.index("trap cleanup_smoke_image EXIT") < smoke.index(
        "for platform in linux/amd64 linux/arm64; do"
    )

    strict_eviction = '            docker image rm "${image_ref}" >/dev/null\n'
    loop_end = "          done\n\n          trap - EXIT"
    assert smoke.count(strict_eviction) == 1
    assert loop_end in smoke
    assert smoke.index(strict_eviction) < smoke.index(loop_end)


def test_install_guide_and_notes_keep_active_no_go() -> None:
    install = _text(ROOT / "INSTALL_V2.md")
    notes = _text(ROOT / "RELEASE_NOTES.md")

    assert "ACTIVE: NO-GO" in install
    assert "paper/migration operator preview" in install
    assert (
        "There is deliberately no `run`, `trade`, `live`, or `activate` command"
        in install
    )
    assert "No Python installation and no wheel are required" in install
    assert "public `ghcr.io/cluster2600/elvis-v2-operator`" in install
    assert "without registry credentials" in install
    assert "ACTIVE: NO-GO" in notes
    assert "GitHub prerelease" in notes
    assert "no Python wheel or PyPI publication" in notes
    assert "latest" in notes
    assert install.index("gh attestation verify") < install.index("tar -xzf")
    assert "$(cat IMAGE_DIGEST.txt)" in install
    assert "$(cat ../IMAGE_DIGEST.txt)" not in install


def test_release_bundle_contains_every_documented_operator_input() -> None:
    workflow = _text(WORKFLOW)
    install = _text(ROOT / "INSTALL_V2.md")

    assert "deploy/v2/pg_service.preview.conf.example" in workflow
    assert "deploy/v2/*-v1.example.json" in workflow
    for example in (
        "pg_service.preview.conf.example",
        "bootstrap-stage-v1.example.json",
        "bootstrap-complete-v1.example.json",
        "cutover-preflight-v1.example.json",
        "legacy-snapshot-import-v1.example.json",
        "legacy-snapshot-reconciliation-v1.example.json",
    ):
        assert example in install
