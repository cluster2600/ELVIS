"""Static safety contract for the dormant V2 PostgreSQL rehearsal."""

from __future__ import annotations

import configparser
import ipaddress
import json
import re
from pathlib import Path

import yaml

_REPOSITORY = Path(__file__).resolve().parents[1]
_DEPLOYMENT = _REPOSITORY / "deploy" / "v2"
_COMPOSE = _DEPLOYMENT / "compose.bootstrap.yml"
_PREVIEW_COMPOSE = _DEPLOYMENT / "compose.preview.yml"
_OPERATOR_DOCKERFILE = _DEPLOYMENT / "operator.Dockerfile"
_OPERATOR_REQUIREMENTS = _DEPLOYMENT / "requirements.operator.txt"
_PREVIEW_ENV = _DEPLOYMENT / "v2-preview.env.example"
_PREVIEW_SERVICE_FILE = _DEPLOYMENT / "pg_service.preview.conf.example"
_HBA = _DEPLOYMENT / "postgres" / "pg_hba.conf"
_REHEARSAL_ENTRYPOINT = _DEPLOYMENT / "postgres" / "rehearsal-entrypoint.sh"
_MARKER_WRITER = _DEPLOYMENT / "postgres" / "write-rehearsal-marker.sh"
_HISTORICAL_STAGE_MANIFEST = _DEPLOYMENT / "bootstrap-stage-v1.example.json"
_HISTORICAL_COMPLETE_MANIFEST = _DEPLOYMENT / "bootstrap-complete-v1.example.json"
_STAGE_MANIFEST = _DEPLOYMENT / "bootstrap-stage-v2.example.json"
_COMPLETE_MANIFEST = _DEPLOYMENT / "bootstrap-complete-v2.example.json"
_SERVICE_FILE = _DEPLOYMENT / "pg_service.conf.example"
_CUTOVER_PREFLIGHT_MANIFEST = _DEPLOYMENT / "cutover-preflight-v1.example.json"
_LEGACY_SNAPSHOT_IMPORT_MANIFEST = (
    _DEPLOYMENT / "legacy-snapshot-import-v1.example.json"
)
_LEGACY_SNAPSHOT_RECONCILIATION_MANIFEST = (
    _DEPLOYMENT / "legacy-snapshot-reconciliation-v1.example.json"
)
_BOOTSTRAP_RUNBOOK = _REPOSITORY / "docs" / "V2_POSTGRES_BOOTSTRAP.md"
_BOOTSTRAP_TRUST_DIAGRAM = _REPOSITORY / "diagrams" / "v2-bootstrap-trust-boundary.mmd"
_BOOTSTRAP_FLOW_DIAGRAM = _REPOSITORY / "diagrams" / "v2-bootstrap-operator-flow.mmd"

_DATABASE = "elvis_paper_v2_rehearsal"
_ADMIN_ROLE = "elvis_bootstrap_admin"
_SUBNET = "10.254.90.0/28"
_ROLE_KEYS = (
    "schema_owner",
    "migrator",
    "opening",
    "legacy_runtime",
    "atomic_runtime",
    "activation",
    "readiness",
    "trainer",
)
_LOGIN_ROLE_KEYS = (
    "migrator",
    "readiness",
    "trainer",
)
_LOGIN_ROLES = (
    "elvis_v2_migrator",
    "elvis_v2_readiness",
    "elvis_v2_trainer",
)
_EXPECTED_FILES = {
    _COMPOSE,
    _PREVIEW_COMPOSE,
    _OPERATOR_DOCKERFILE,
    _OPERATOR_REQUIREMENTS,
    _PREVIEW_ENV,
    _PREVIEW_SERVICE_FILE,
    _HBA,
    _REHEARSAL_ENTRYPOINT,
    _MARKER_WRITER,
    _HISTORICAL_STAGE_MANIFEST,
    _HISTORICAL_COMPLETE_MANIFEST,
    _STAGE_MANIFEST,
    _COMPLETE_MANIFEST,
    _SERVICE_FILE,
    _CUTOVER_PREFLIGHT_MANIFEST,
    _LEGACY_SNAPSHOT_IMPORT_MANIFEST,
    _LEGACY_SNAPSHOT_RECONCILIATION_MANIFEST,
}


def _load_compose() -> dict[str, object]:
    document = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _load_manifest(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _hba_rows() -> list[tuple[str, ...]]:
    rows = []
    for raw_line in _HBA.read_text(encoding="utf-8").splitlines():
        line = raw_line.partition("#")[0].strip()
        if line:
            rows.append(tuple(line.split()))
    return rows


def test_bootstrap_runbook_embeds_the_exact_diagram_sources() -> None:
    runbook = _BOOTSTRAP_RUNBOOK.read_text(encoding="utf-8")
    mermaid_fences = re.findall(r"```mermaid\n(.*?)```", runbook, re.S)

    for diagram in (_BOOTSTRAP_TRUST_DIAGRAM, _BOOTSTRAP_FLOW_DIAGRAM):
        source = diagram.read_text(encoding="utf-8")
        assert mermaid_fences.count(source) == 1
        relative = diagram.relative_to(_REPOSITORY)
        assert f"../{relative.as_posix()}" in runbook


def test_v2_deployment_surface_is_exact_and_dormant() -> None:
    assert {path for path in _DEPLOYMENT.rglob("*") if path.is_file()} == (
        _EXPECTED_FILES
    )

    compose = _load_compose()
    services = compose["services"]
    assert isinstance(services, dict)
    assert set(services) == {"postgres", "bootstrap"}

    postgres = services["postgres"]
    bootstrap = services["bootstrap"]
    assert postgres["profiles"] == ["v2-rehearsal"]
    assert bootstrap["profiles"] == ["v2-operator"]
    assert postgres["restart"] == "no"
    assert bootstrap["restart"] == "no"
    assert bootstrap["user"] == (
        "${ELVIS_V2_OPERATOR_UID:?set ELVIS_V2_OPERATOR_UID}:"
        "${ELVIS_V2_OPERATOR_GID:?set ELVIS_V2_OPERATOR_GID}"
    )
    assert "ports" not in postgres
    assert "ports" not in bootstrap
    assert "depends_on" not in bootstrap
    assert bootstrap["read_only"] is True
    assert bootstrap["cap_drop"] == ["ALL"]
    assert bootstrap["security_opt"] == ["no-new-privileges:true"]
    assert bootstrap["entrypoint"] == [
        "python",
        "-m",
        "scripts.v2_operator",
        "bootstrap",
    ]
    assert postgres["entrypoint"] == ["/usr/local/bin/elvis-v2-rehearsal-entrypoint"]
    assert compose["volumes"] == {
        "rehearsal-data": {"labels": {"org.elvis.v2.scope": "fresh-rehearsal-only"}}
    }
    assert postgres["volumes"][0] == "rehearsal-data:/var/lib/postgresql/data"
    assert all("btc_bot" not in volume for volume in postgres["volumes"])

    dockerfile = _OPERATOR_DOCKERFILE.read_text(encoding="utf-8")
    assert dockerfile.splitlines()[0] == (
        "FROM python:3.14-slim@sha256:"
        "b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1"
    )
    assert "--require-hashes" in dockerfile
    assert "requirements.operator.txt" in dockerfile
    assert 'ENTRYPOINT ["python", "-m", "scripts.v2_operator"]' in dockerfile
    assert 'CMD ["--help"]' in dockerfile
    assert "main.py" not in dockerfile

    entrypoint = _REHEARSAL_ENTRYPOINT.read_text(encoding="utf-8")
    marker_writer = _MARKER_WRITER.read_text(encoding="utf-8")
    assert entrypoint.startswith("#!/bin/sh\nset -eu\n")
    assert '"$PGDATA/PG_VERSION"' in entrypoint
    assert '"$PGDATA/.elvis-v2-fresh-rehearsal-v1"' in entrypoint
    assert 'exec /usr/local/bin/docker-entrypoint.sh "$@"' in entrypoint
    assert marker_writer.startswith("#!/bin/sh\nset -eu\n")
    assert "umask 077" in marker_writer
    assert 'mv "$temporary_marker" "$marker_path"' in marker_writer


def test_v2_rehearsal_network_is_internal_and_has_no_host_binding() -> None:
    compose = _load_compose()
    networks = compose["networks"]
    assert set(networks) == {"rehearsal"}
    rehearsal = networks["rehearsal"]
    assert rehearsal["internal"] is True
    assert rehearsal["enable_ipv6"] is False
    assert rehearsal["ipam"]["config"] == [{"subnet": _SUBNET}]

    services = compose["services"]
    assert services["postgres"]["image"] == (
        "postgres:15-alpine@sha256:"
        "3d0f7584ed7d04e27fa050d6683a74746608faf21f202be78460d679cc56461f"
    )
    postgres_network = services["postgres"]["networks"]
    assert postgres_network == {"rehearsal": {"ipv4_address": "10.254.90.2"}}
    assert ipaddress.ip_address("10.254.90.2") in ipaddress.ip_network(_SUBNET)
    assert services["bootstrap"]["networks"] == ["rehearsal"]


def test_v2_rehearsal_uses_external_files_without_literal_credentials() -> None:
    compose = _load_compose()
    postgres = compose["services"]["postgres"]
    bootstrap = compose["services"]["bootstrap"]
    assert postgres["environment"] == {
        "POSTGRES_DB": _DATABASE,
        "POSTGRES_USER": _ADMIN_ROLE,
        "POSTGRES_PASSWORD_FILE": "/run/secrets/postgres_admin_password",
        "POSTGRES_INITDB_ARGS": (
            "--auth-local=scram-sha-256 --auth-host=scram-sha-256"
        ),
    }
    assert bootstrap["environment"] == {
        "PGSERVICEFILE": "/run/operator/pg_service.conf",
        "PGPASSFILE": "/run/operator/pgpass",
    }
    assert set(compose["secrets"]) == {"postgres_admin_password"}

    for path in _EXPECTED_FILES:
        contents = path.read_text(encoding="utf-8")
        assert re.search(r"postgres(?:ql)?://", contents, re.IGNORECASE) is None
        assert re.search(r"(?im)^\s*password\s*=", contents) is None

    for manifest_path in (_STAGE_MANIFEST, _COMPLETE_MANIFEST):
        manifest = _load_manifest(manifest_path)
        serialized = json.dumps(manifest).lower()
        for forbidden_key in ('"password"', '"dsn"', '"host"', '"port"', '"user"'):
            assert forbidden_key not in serialized


def test_v2_hba_is_an_exact_scram_allowlist_followed_by_rejects() -> None:
    expected = [
        ("local", "postgres", _ADMIN_ROLE, "scram-sha-256"),
        ("local", _DATABASE, _ADMIN_ROLE, "scram-sha-256"),
        ("local", "all", "all", "reject"),
        ("host", _DATABASE, _ADMIN_ROLE, _SUBNET, "scram-sha-256"),
        *(("host", _DATABASE, role, _SUBNET, "scram-sha-256") for role in _LOGIN_ROLES),
        ("host", "all", "all", "0.0.0.0/0", "reject"),
        ("host", "all", "all", "::/0", "reject"),
    ]
    assert _hba_rows() == expected


def test_v2_stage_and_complete_manifests_encode_only_service_names() -> None:
    stage = _load_manifest(_STAGE_MANIFEST)
    complete = _load_manifest(_COMPLETE_MANIFEST)
    expected_top_level = {
        "schema_version",
        "expected_database",
        "admin_role",
        "roles",
        "services",
        "opening_admission",
        "adoption",
    }
    expected_service_keys = {"admin", *_ROLE_KEYS}

    for manifest in (stage, complete):
        assert set(manifest) == expected_top_level
        assert manifest["schema_version"] == 2
        assert manifest["expected_database"] == _DATABASE
        assert manifest["admin_role"] == _ADMIN_ROLE
        assert manifest["adoption"] is None
        assert set(manifest["roles"]) == set(_ROLE_KEYS)
        assert set(manifest["services"]) == expected_service_keys
        assert manifest["services"]["schema_owner"] is None
        assert manifest["services"]["opening"] is None
        assert manifest["services"]["legacy_runtime"] is None
        assert manifest["services"]["atomic_runtime"] is None
        assert manifest["services"]["activation"] is None
        assert set(manifest["opening_admission"]) == {
            "candidate_sha256",
            "pin_authority_record_sha256",
            "deployment_incarnation_id",
        }

    assert stage["roles"] == complete["roles"]
    assert tuple(complete["roles"][key] for key in _LOGIN_ROLE_KEYS) == _LOGIN_ROLES
    assert len(set(stage["roles"].values())) == len(_ROLE_KEYS)
    assert stage["services"]["admin"] == "elvis_v2_admin"
    assert all(stage["services"][key] is None for key in _LOGIN_ROLE_KEYS)

    complete_service_ids = tuple(complete["services"][key] for key in _LOGIN_ROLE_KEYS)
    assert all(isinstance(service_id, str) for service_id in complete_service_ids)
    assert len(set(complete_service_ids)) == len(_LOGIN_ROLE_KEYS)
    assert complete_service_ids == _LOGIN_ROLES

    for historical_path in (
        _HISTORICAL_STAGE_MANIFEST,
        _HISTORICAL_COMPLETE_MANIFEST,
    ):
        historical = _load_manifest(historical_path)
        assert historical["schema_version"] == 1
        assert "opening" not in historical["roles"]
        assert "opening" not in historical["services"]


def test_v2_libpq_services_are_exact_separate_identities_without_passwords() -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(_SERVICE_FILE.read_text(encoding="utf-8"))
    expected_sections = ["elvis_v2_admin", *_LOGIN_ROLES]
    assert parser.sections() == expected_sections

    expected_users = [_ADMIN_ROLE, *_LOGIN_ROLES]
    actual_users = []
    for section in expected_sections:
        values = dict(parser[section])
        assert values == {
            "host": "postgres",
            "port": "5432",
            "dbname": _DATABASE,
            "user": values["user"],
        }
        actual_users.append(values["user"])

    assert actual_users == expected_users
    assert len(set(actual_users)) == len(expected_users)
