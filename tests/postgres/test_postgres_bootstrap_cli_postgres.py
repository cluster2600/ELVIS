"""PostgreSQL 15 end-to-end checks for the offline bootstrap CLI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn

from scripts import postgres_bootstrap as cli
from tests.test_fresh_opening import _approval as build_opening_approval
from tests.test_fresh_opening import _intent as build_opening_intent
from tests.test_fresh_opening import _policy as build_opening_policy
from trading.application.fresh_opening import (
    derive_prospective_fresh_opening_candidate,
    encode_detached_fresh_opening_approval,
    encode_fresh_opening_intent,
    encode_fresh_opening_trust_policy,
)

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
_BOOTSTRAP_CREDENTIAL_ROLE_KEYS = (
    "migrator",
    "readiness",
    "trainer",
)
_TERMINAL_LOGIN_ROLE_KEYS = (
    "readiness",
    "trainer",
)
_TERMINAL_NOLOGIN_ROLE_KEYS = tuple(
    role_key for role_key in _ROLE_KEYS if role_key not in _TERMINAL_LOGIN_ROLE_KEYS
)
_SERVICELESS_ROLE_KEYS = (
    "schema_owner",
    "opening",
    "legacy_runtime",
    "atomic_runtime",
    "activation",
)


@dataclass(frozen=True)
class BootstrapCliCluster:
    admin_dsn: str = field(repr=False)
    config: dict[str, object]
    config_path: Path
    resolver: Callable[[str], Callable[[], object]] = field(repr=False)
    roles: dict[str, str]
    passwords: dict[str, str] = field(repr=False)
    services: dict[str, str | None]


def _dsn_identity(dsn: str):
    return frozenset(parse_dsn(dsn).items())


def _config(
    *,
    database: str,
    admin_role: str,
    roles: dict[str, str],
    services: dict[str, str | None],
    opening_admission: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "expected_database": database,
        "admin_role": admin_role,
        "roles": roles,
        "services": services,
        "opening_admission": opening_admission,
        "adoption": None,
    }


def _canonical_sha256(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_opening_authority(tmp_path: Path) -> dict[str, str]:
    policy = build_opening_policy()
    now = datetime.now(timezone.utc)
    intent = build_opening_intent(
        policy,
        approval_id=f"opening-approval-pg-{uuid4().hex}",
        approval_issued_at=now - timedelta(minutes=5),
        approval_expires_at=now + timedelta(minutes=55),
        nonce=uuid4().hex * 2,
    )
    approval = build_opening_approval(intent)
    policy_document = encode_fresh_opening_trust_policy(policy)
    intent_document = encode_fresh_opening_intent(intent)
    approval_document = encode_detached_fresh_opening_approval(approval)
    candidate = derive_prospective_fresh_opening_candidate(
        intent,
        approval,
        policy,
        opening_codec=cli.opening_plan_cli._OPENING_CODEC,
    )
    for name, document in (
        ("opening-intent.json", intent_document),
        ("opening-approval.json", approval_document),
        ("opening-policy.json", policy_document),
    ):
        (tmp_path / name).write_text(document.payload, encoding="utf-8")
    return {
        "candidate_sha256": candidate.candidate_document.sha256,
        "pin_authority_record_sha256": "b" * 64,
        "deployment_incarnation_id": f"deployment-pg-{uuid4().hex}",
    }


def _apply_args(path: Path) -> list[str]:
    config_document = json.loads(path.read_text(encoding="utf-8"))
    policy_path = path.parent / "opening-policy.json"
    policy_document = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_sha256 = _canonical_sha256(policy_document)
    public_key = bytes.fromhex(policy_document["anchors"][0]["ed25519_public_key"])
    return [
        "--config",
        str(path),
        "--pinned-config-sha256",
        _canonical_sha256(config_document),
        "--opening-intent",
        str(path.parent / "opening-intent.json"),
        "--opening-approval",
        str(path.parent / "opening-approval.json"),
        "--opening-trust-policy",
        str(policy_path),
        "--pinned-trust-policy-sha256",
        policy_sha256,
        "--pinned-signer-public-key-sha256",
        hashlib.sha256(public_key).hexdigest(),
        "--apply",
        "--confirm-exclusive-ddl-role-window",
    ]


def _service_resolver(service_dsns: dict[str, str]):
    def resolve(service_name: str):
        dsn = service_dsns[service_name]
        return lambda: psycopg2.connect(dsn)

    return resolve


def _enable_bootstrap_credential_services(cluster: BootstrapCliCluster) -> None:
    config = dict(cluster.config)
    config["services"] = cluster.services
    cluster.config_path.write_text(json.dumps(config), encoding="utf-8")


def _provision_passwords(
    admin_dsn: str,
    roles: dict[str, str],
    passwords: dict[str, str],
) -> None:
    connection = psycopg2.connect(admin_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for role_key in _BOOTSTRAP_CREDENTIAL_ROLE_KEYS:
                role = roles[role_key]
                cursor.execute(
                    sql.SQL("ALTER ROLE {} LOGIN PASSWORD %s").format(
                        sql.Identifier(role)
                    ),
                    (passwords[role_key],),
                )
    finally:
        connection.close()


def _drop_roles(admin_dsn: str, roles: dict[str, str]) -> None:
    connection = psycopg2.connect(admin_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for role in reversed(tuple(roles.values())):
                role_exists_query = (
                    "SELECT EXISTS (" "SELECT 1 FROM pg_roles WHERE rolname = %s" ")"
                )
                cursor.execute(role_exists_query, (role,))
                if not cursor.fetchone()[0]:
                    continue
                cursor.execute(
                    "SELECT pg_terminate_backend(pid) "
                    "FROM pg_stat_activity "
                    "WHERE usename = %s AND pid <> pg_backend_pid()",
                    (role,),
                )
                cursor.execute(
                    sql.SQL("REASSIGN OWNED BY {} TO CURRENT_USER").format(
                        sql.Identifier(role)
                    )
                )
                drop_owned = sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role))
                cursor.execute(drop_owned)
                cursor.execute(
                    sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role))
                )
    finally:
        connection.close()


def _assert_terminal_login_attributes(cluster: BootstrapCliCluster) -> None:
    connection = psycopg2.connect(cluster.admin_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT rolname, rolcanlogin FROM pg_roles WHERE rolname = ANY(%s)",
                (list(cluster.roles.values()),),
            )
            role_attributes = dict(cursor.fetchall())
    finally:
        connection.close()

    assert role_attributes == {
        cluster.roles[role_key]: role_key in _TERMINAL_LOGIN_ROLE_KEYS
        for role_key in _ROLE_KEYS
    }
    assert len(_TERMINAL_LOGIN_ROLE_KEYS) == 2
    assert len(_TERMINAL_NOLOGIN_ROLE_KEYS) == 6


@pytest.fixture
def bootstrap_cli_cluster(
    tmp_path,
    postgres_database_dsn,
    postgres_connection_allowlist,
):
    database_parameters = parse_dsn(postgres_database_dsn)
    database = database_parameters["dbname"]
    admin_role = database_parameters["user"]
    suffix = uuid4().hex[:12]
    prefix = f"ec_{suffix}"
    roles = {
        "schema_owner": f"{prefix}_owner",
        "migrator": f"{prefix}_migrator",
        "opening": f"{prefix}_opening",
        "legacy_runtime": f"{prefix}_legacy",
        "atomic_runtime": f"{prefix}_atomic",
        "activation": f"{prefix}_activation",
        "readiness": f"{prefix}_readiness",
        "trainer": f"{prefix}_trainer",
    }
    services: dict[str, str | None] = {
        "admin": f"{prefix}_admin_service",
        **{role_key: None for role_key in _ROLE_KEYS},
    }
    services.update(
        {
            role_key: f"{prefix}_{role_key}_service"
            for role_key in _BOOTSTRAP_CREDENTIAL_ROLE_KEYS
        }
    )
    passwords = {
        role_key: f"test-only-{suffix}-{index}"
        for index, role_key in enumerate(_BOOTSTRAP_CREDENTIAL_ROLE_KEYS, start=1)
    }
    service_dsns = {services["admin"]: postgres_database_dsn}
    for role_key in _BOOTSTRAP_CREDENTIAL_ROLE_KEYS:
        service_name = services[role_key]
        assert service_name is not None
        parameters = dict(database_parameters)
        parameters.update(user=roles[role_key], password=passwords[role_key])
        role_dsn = make_dsn(**parameters)
        service_dsns[service_name] = role_dsn
        postgres_connection_allowlist.add(_dsn_identity(role_dsn))

    initial_services = dict(services)
    for role_key in _BOOTSTRAP_CREDENTIAL_ROLE_KEYS:
        initial_services[role_key] = None
    opening_admission = _write_opening_authority(tmp_path)
    config = _config(
        database=database,
        admin_role=admin_role,
        roles=roles,
        services=initial_services,
        opening_admission=opening_admission,
    )
    config_path = tmp_path / "bootstrap.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    try:
        yield BootstrapCliCluster(
            admin_dsn=postgres_database_dsn,
            config=config,
            config_path=config_path,
            resolver=_service_resolver(service_dsns),
            roles=roles,
            passwords=passwords,
            services=services,
        )
    finally:
        for role_key in _BOOTSTRAP_CREDENTIAL_ROLE_KEYS:
            service_name = services[role_key]
            assert service_name is not None
            postgres_connection_allowlist.discard(
                _dsn_identity(service_dsns[service_name])
            )
        _drop_roles(postgres_database_dsn, roles)


def test_cli_two_pass_bootstrap_is_complete_and_idempotent(
    bootstrap_cli_cluster,
    capsys,
) -> None:
    cluster = bootstrap_cli_cluster
    arguments = _apply_args(cluster.config_path)

    first_exit = cli.main(
        arguments,
        service_connection_factory=cluster.resolver,
    )
    first_output = json.loads(capsys.readouterr().out)

    assert first_exit == 10
    assert first_output == {
        "status": "CREDENTIALS_REQUIRED",
        "migration_versions": [],
        "verified_role_probes": [],
        "pending_role_credentials": [
            cluster.roles[role_key] for role_key in _BOOTSTRAP_CREDENTIAL_ROLE_KEYS
        ],
        "old_shared_runtime_demoted": False,
    }

    _provision_passwords(
        cluster.admin_dsn,
        cluster.roles,
        cluster.passwords,
    )
    _enable_bootstrap_credential_services(cluster)
    arguments = _apply_args(cluster.config_path)

    second_exit = cli.main(
        arguments,
        service_connection_factory=cluster.resolver,
    )
    second_output = json.loads(capsys.readouterr().out)
    assert second_exit == 0
    assert second_output == {
        "status": "COMPLETE",
        "migration_versions": [1, 2, 3, 4, 5, 6, 7],
        "verified_role_probes": [
            cluster.roles[role_key] for role_key in _TERMINAL_LOGIN_ROLE_KEYS
        ],
        "pending_role_credentials": [],
        "old_shared_runtime_demoted": False,
    }

    repeated_exit = cli.main(
        arguments,
        service_connection_factory=cluster.resolver,
    )
    repeated_output = json.loads(capsys.readouterr().out)
    assert repeated_exit == 0
    assert repeated_output == second_output
    assert all(
        cluster.services[role_key] is None for role_key in _SERVICELESS_ROLE_KEYS
    )
    _assert_terminal_login_attributes(cluster)


def test_cli_reports_terminal_drift_without_repair(
    bootstrap_cli_cluster,
    capsys,
) -> None:
    cluster = bootstrap_cli_cluster
    arguments = _apply_args(cluster.config_path)

    assert (
        cli.main(
            arguments,
            service_connection_factory=cluster.resolver,
        )
        == 10
    )
    _ = capsys.readouterr()
    _provision_passwords(
        cluster.admin_dsn,
        cluster.roles,
        cluster.passwords,
    )
    _enable_bootstrap_credential_services(cluster)
    arguments = _apply_args(cluster.config_path)
    assert (
        cli.main(
            arguments,
            service_connection_factory=cluster.resolver,
        )
        == 0
    )
    _ = capsys.readouterr()

    connection = psycopg2.connect(cluster.admin_dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER ROLE {} CREATEDB").format(
                    sql.Identifier(cluster.roles["trainer"])
                )
            )
    finally:
        connection.close()

    drift_exit = cli.main(
        arguments,
        service_connection_factory=cluster.resolver,
    )
    captured = capsys.readouterr()
    assert drift_exit == 21
    assert captured.out == '{"status":"ERROR","code":"DRIFT"}\n'
    assert captured.err == ""

    connection = psycopg2.connect(cluster.admin_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT rolcreatedb FROM pg_roles WHERE rolname = %s",
                (cluster.roles["trainer"],),
            )
            assert cursor.fetchone() == (True,)
        connection.rollback()
    finally:
        connection.close()
