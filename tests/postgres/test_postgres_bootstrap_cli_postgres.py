"""PostgreSQL 15 end-to-end checks for the offline bootstrap CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn

from scripts import postgres_bootstrap as cli

_ROLE_KEYS = (
    "schema_owner",
    "migrator",
    "legacy_runtime",
    "atomic_runtime",
    "activation",
    "readiness",
    "trainer",
)
_LOGIN_ROLE_KEYS = _ROLE_KEYS[1:]


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
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "expected_database": database,
        "admin_role": admin_role,
        "roles": roles,
        "services": services,
        "adoption": None,
    }


def _apply_args(path: Path) -> list[str]:
    return [
        "--config",
        str(path),
        "--apply",
        "--confirm-exclusive-ddl-role-window",
    ]


def _service_resolver(service_dsns: dict[str, str]):
    def resolve(service_name: str):
        dsn = service_dsns[service_name]
        return lambda: psycopg2.connect(dsn)

    return resolve


def _enable_login_services(cluster: BootstrapCliCluster) -> None:
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
            for role_key in _LOGIN_ROLE_KEYS:
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
        "legacy_runtime": f"{prefix}_legacy",
        "atomic_runtime": f"{prefix}_atomic",
        "activation": f"{prefix}_activation",
        "readiness": f"{prefix}_readiness",
        "trainer": f"{prefix}_trainer",
    }
    services = {
        "admin": f"{prefix}_admin_service",
        "schema_owner": None,
        **{role_key: f"{prefix}_{role_key}_service" for role_key in _LOGIN_ROLE_KEYS},
    }
    passwords = {
        role_key: f"test-only-{suffix}-{index}"
        for index, role_key in enumerate(_LOGIN_ROLE_KEYS, start=1)
    }
    service_dsns = {services["admin"]: postgres_database_dsn}
    for role_key in _LOGIN_ROLE_KEYS:
        parameters = dict(database_parameters)
        parameters.update(user=roles[role_key], password=passwords[role_key])
        role_dsn = make_dsn(**parameters)
        service_dsns[services[role_key]] = role_dsn
        postgres_connection_allowlist.add(_dsn_identity(role_dsn))

    initial_services = dict(services)
    for role_key in _LOGIN_ROLE_KEYS:
        initial_services[role_key] = None
    config = _config(
        database=database,
        admin_role=admin_role,
        roles=roles,
        services=initial_services,
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
        for role_key in _LOGIN_ROLE_KEYS:
            postgres_connection_allowlist.discard(
                _dsn_identity(service_dsns[services[role_key]])
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
            cluster.roles[role_key] for role_key in _LOGIN_ROLE_KEYS
        ],
        "old_shared_runtime_demoted": False,
    }

    _provision_passwords(
        cluster.admin_dsn,
        cluster.roles,
        cluster.passwords,
    )
    _enable_login_services(cluster)

    second_exit = cli.main(
        arguments,
        service_connection_factory=cluster.resolver,
    )
    second_output = json.loads(capsys.readouterr().out)
    assert second_exit == 0
    assert second_output == {
        "status": "COMPLETE",
        "migration_versions": [1, 2, 3, 4, 5, 6],
        "verified_role_probes": [
            cluster.roles[role_key] for role_key in _LOGIN_ROLE_KEYS
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
    _enable_login_services(cluster)
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
