"""Two-cluster PostgreSQL 15 proof for the read-only cut-over preflight."""

from __future__ import annotations

import os
import re
import secrets
import stat
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from psycopg2 import OperationalError, sql
from psycopg2.extensions import make_dsn

from tests.conftest import _ORIGINAL_PSYCOPG2_CONNECT
from trading.application.fresh_target_cutover import (
    FreshTargetBootstrapIntent,
    FreshTargetCutoverBlocker,
    FreshTargetCutoverContext,
    FreshTargetCutoverStatus,
    FreshTargetRoleManifest,
)
from trading.persistence import apply_migrations, load_migrations
from trading.persistence import postgres_cutover_preflight as preflight_module
from trading.persistence.postgres_bootstrap import (
    PostgresBootstrapContext,
    PostgresBootstrapRoles,
)
from trading.persistence.postgres_cutover_preflight import PostgresCutoverPreflight

_POSTGRES_IMAGE = (
    "postgres:15-alpine@"
    "sha256:3d0f7584ed7d04e27fa050d6683a74746608faf21f202be78460d679cc56461f"
)
_REQUIRED_ENV = "ELVIS_TEST_V2_CUTOVER_PREFLIGHT_REQUIRED"
_SOURCE_DATABASE = "elvis_source_clone"
_TARGET_DATABASE = "elvis_fresh_target"
_ADMIN_ROLE = "postgres"
_LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)
_HISTORICAL_HEAD6_AUTHORITY_TABLES = (
    "account_balances",
    "liquidations",
    "margin_history",
    "model_predictions",
    "open_positions",
    "order_events",
    "orders",
    "paper_account_balances",
    "paper_account_batch_manifests",
    "paper_account_postings",
    "paper_account_settlements",
    "paper_account_streams",
    "paper_margin_reservations",
    "paper_runtime_control",
    "paper_runtime_generations",
    "position_streams",
    "schema_migrations",
    "trades",
    "trading_session_resets",
)
_HISTORICAL_HEAD6_LEGACY_SEQUENCES = tuple(
    f"{relation.removeprefix('np.')}_id_seq" for relation in _LEGACY_RELATIONS
)
_HISTORICAL_HEAD6_NON_ACTIVATION_FUNCTIONS = (
    "np.enforce_legacy_paper_runtime_fence()",
    "np.reject_paper_runtime_generation_mutation()",
)
_HISTORICAL_HEAD6_ACTIVATION_FUNCTIONS = (
    "np.acquire_paper_runtime_activation_fence()",
    (
        "np.activate_paper_runtime_generation("
        "text,bigint,bigint,text,text,text,bigint,text)"
    ),
)
_HISTORICAL_HEAD6_LEGACY_PRIVILEGES = {
    "account_balances": ("SELECT", "INSERT", "UPDATE"),
    "liquidations": ("SELECT", "INSERT"),
    "margin_history": ("SELECT", "INSERT"),
    "model_predictions": ("SELECT", "INSERT", "UPDATE"),
    "open_positions": ("SELECT", "INSERT", "DELETE"),
    "trades": ("SELECT", "INSERT", "DELETE"),
    "trading_session_resets": ("SELECT", "INSERT"),
}
_HISTORICAL_HEAD6_ATOMIC_PRIVILEGES = {
    "order_events": ("SELECT", "INSERT"),
    "orders": ("SELECT", "INSERT", "UPDATE"),
    "paper_account_balances": ("SELECT", "INSERT", "UPDATE"),
    "paper_account_batch_manifests": ("SELECT", "INSERT"),
    "paper_account_postings": ("SELECT", "INSERT"),
    "paper_account_settlements": ("SELECT", "INSERT"),
    "paper_account_streams": ("SELECT", "INSERT", "UPDATE"),
    "paper_margin_reservations": ("SELECT", "INSERT", "DELETE"),
    "position_streams": ("SELECT", "INSERT", "UPDATE"),
}
_MUTATING_SQL = re.compile(
    r"^\s*(?:ALTER|CALL|COPY|CREATE|DELETE|DO|DROP|GRANT|INSERT|REVOKE|TRUNCATE|UPDATE)\b",
    re.IGNORECASE,
)
_EXPECTED_RELATION_SHA256 = {
    "np.account_balances": (
        "8319dd40e1de9cb6c9afde5af062324f12487c0d92a295684e5068f4ed59980a"
    ),
    "np.liquidations": (
        "89bde5a159cf0cbad577c937fde5a883598525180f9e33d2497fbe664408bf3a"
    ),
    "np.margin_history": (
        "2bc2d2dac3019d304102cd74dd7e00a3085dce5f1ff5a72599bf17264fe3fcc7"
    ),
    "np.model_predictions": (
        "ca56ec2ecb19adb195deb2575afdf90c56c4f93bf6152650d71f8d8a30ea8558"
    ),
    "np.open_positions": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "np.trades": ("884ad3ae06cdf979bf41f3a22a29b3963bbe130c3294359c7121ecec17bd3842"),
    "np.trading_session_resets": (
        "9a1607d6933347602d173f36bdcccd8dccf1890aa092be72341d5e10a37ce9f4"
    ),
}
_EXPECTED_SOURCE_SHA256 = (
    "907b095095741ed1193f79e5b89d9b0129ae61cff05d35457bb3b4036112c35b"
)

pytestmark = pytest.mark.skipif(
    os.getenv(_REQUIRED_ENV) != "1",
    reason=f"set {_REQUIRED_ENV}=1 to run the isolated two-cluster preflight",
)


def _redact(value: str, secrets_to_redact: tuple[str, ...]) -> str:
    for secret in secrets_to_redact:
        value = value.replace(secret, "<redacted>")
    return value


def _run(
    command: list[str],
    *,
    expected_exit_codes: tuple[int, ...] = (0,),
    secrets_to_redact: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    if any(
        secret and (secret in result.stdout or secret in result.stderr)
        for secret in secrets_to_redact
    ):
        pytest.fail("Docker output exposed a generated test secret", pytrace=False)
    if result.returncode not in expected_exit_codes:
        details = (
            f"command exited {result.returncode}, expected {expected_exit_codes}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        pytest.fail(_redact(details, secrets_to_redact), pytrace=False)
    return result


def _write_private(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o600)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _published_port(container: str) -> int:
    deadline = time.monotonic() + 5.0
    while True:
        result = _run(
            ["docker", "port", container, "5432/tcp"],
            expected_exit_codes=(0, 1),
        )
        output = result.stdout.strip()
        match = re.fullmatch(r"127\.0\.0\.1:([0-9]{1,5})", output)
        if result.returncode == 0 and match is not None:
            port = int(match.group(1))
            assert 1 <= port <= 65_535
            return port
        if time.monotonic() >= deadline:
            pytest.fail(
                "Docker did not publish the expected loopback PostgreSQL port",
                pytrace=False,
            )
        time.sleep(0.05)


def test_published_port_retries_until_docker_exposes_the_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        (
            subprocess.CompletedProcess(
                args=(),
                returncode=1,
                stdout="",
                stderr="mapping is not visible yet",
            ),
            subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout="127.0.0.1:54321\n",
                stderr="",
            ),
        )
    )
    calls: list[tuple[list[str], tuple[int, ...]]] = []
    sleeps: list[float] = []

    def run_once(
        command: list[str],
        *,
        expected_exit_codes: tuple[int, ...] = (0,),
        secrets_to_redact: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        assert secrets_to_redact == ()
        calls.append((command, expected_exit_codes))
        return next(results)

    monkeypatch.setattr(__name__ + "._run", run_once)
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    assert _published_port("test-container") == 54321
    assert calls == [
        (["docker", "port", "test-container", "5432/tcp"], (0, 1)),
        (["docker", "port", "test-container", "5432/tcp"], (0, 1)),
    ]
    assert sleeps == [0.05]


def _wait_for_postgres(container: str, database: str) -> None:
    for _attempt in range(60):
        result = _run(
            [
                "docker",
                "exec",
                container,
                "pg_isready",
                "--username",
                _ADMIN_ROLE,
                "--dbname",
                database,
            ],
            expected_exit_codes=(0, 1, 2),
        )
        if result.returncode == 0:
            return
        import time

        time.sleep(0.25)
    pytest.fail(f"PostgreSQL container {container} did not become ready", pytrace=False)


def _connect(dsn: str):
    deadline = time.monotonic() + 5.0
    while True:
        try:
            connection = _ORIGINAL_PSYCOPG2_CONNECT(dsn)
        except OperationalError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(0.1, remaining))
        else:
            connection.autocommit = False
            return connection


def test_connect_retries_a_transient_published_port_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Connection:
        autocommit = True

    connection = Connection()
    results = iter((OperationalError("port is not ready"), connection))
    sleeps: list[float] = []

    def connect_once(_dsn: str):
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return result

    monotonic = iter((0.0, 0.0))
    monkeypatch.setattr(__name__ + "._ORIGINAL_PSYCOPG2_CONNECT", connect_once)
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(time, "sleep", sleeps.append)

    assert _connect("test-only-dsn") is connection
    assert sleeps == [0.1]


def _create_isolated_network(network: str, label: str, seed: str) -> None:
    start = int(seed[:2], 16)
    for offset in range(256):
        third_octet = (start + offset) % 256
        result = _run(
            [
                "docker",
                "network",
                "create",
                "--subnet",
                f"10.253.{third_octet}.0/28",
                "--label",
                label,
                network,
            ],
            expected_exit_codes=(0, 1),
        )
        if result.returncode == 0:
            return
        if "overlap" not in result.stderr.lower():
            pytest.fail("could not create the isolated Docker network", pytrace=False)
    pytest.fail("no isolated Docker test subnet was available", pytrace=False)


def _assert_project_absent(project: str) -> None:
    for resource, arguments in (
        ("container", ["container", "ls", "--all"]),
        ("network", ["network", "ls"]),
        ("volume", ["volume", "ls"]),
    ):
        result = _run(
            [
                "docker",
                *arguments,
                "--quiet",
                "--filter",
                f"label=org.elvis.v2.cutover-test={project}",
            ]
        )
        assert result.stdout.strip() == "", f"residual {resource} for {project}"


class RecordingCursor:
    def __init__(self, cursor: object, statements: list[str]) -> None:
        self._cursor = cursor
        self._statements = statements

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def __enter__(self) -> "RecordingCursor":
        self._cursor.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        return self._cursor.__exit__(*args)

    def execute(self, query: object, variables: object = None) -> object:
        rendered = (
            query.as_string(self._cursor) if hasattr(query, "as_string") else str(query)
        )
        self._statements.append(rendered)
        if variables is None:
            return self._cursor.execute(query)
        return self._cursor.execute(query, variables)


@dataclass
class ConnectionRecord:
    statements: list[str] = field(default_factory=list)
    commits: int = 0
    rollbacks: int = 0


class RecordingConnection:
    def __init__(self, connection: object, record: ConnectionRecord) -> None:
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "_record", record)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._connection, name, value)

    def cursor(self, *args: object, **kwargs: object) -> RecordingCursor:
        return RecordingCursor(
            self._connection.cursor(*args, **kwargs),
            self._record.statements,
        )

    def commit(self) -> None:
        self._record.commits += 1
        self._connection.commit()

    def rollback(self) -> None:
        self._record.rollbacks += 1
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


@dataclass(frozen=True)
class CutoverPair:
    project: str
    source_container: str
    target_container: str
    source_dsn: str = field(repr=False)
    target_dsn: str = field(repr=False)
    password: str = field(repr=False)
    context: FreshTargetCutoverContext

    def connect(self, dsn: str):
        return _connect(dsn)

    def source_factory(self):
        return self.connect(self.source_dsn)

    def target_factory(self):
        return self.connect(self.target_dsn)

    def inspector(
        self,
        *,
        records: list[ConnectionRecord] | None = None,
        source_factory=None,
        target_factory=None,
    ) -> PostgresCutoverPreflight:
        def recorded(factory):
            def connect():
                record = ConnectionRecord()
                assert records is not None
                records.append(record)
                return RecordingConnection(factory(), record)

            return connect

        source = source_factory or self.source_factory
        target = target_factory or self.target_factory
        if records is not None:
            source = recorded(source)
            target = recorded(target)
        return PostgresCutoverPreflight(source, target)

    def execute(self, dsn: str, statement: object, parameters: object = None) -> None:
        connection = self.connect(dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute(statement, parameters)
        finally:
            connection.close()


def _bootstrap_historical_head6_target(
    target_dsn: str,
    suffix: str,
    *,
    role_prefix: str,
    roles: PostgresBootstrapRoles | None = None,
) -> tuple[PostgresBootstrapContext, str]:
    if roles is None:
        roles = PostgresBootstrapRoles(
            schema_owner=f"{role_prefix}_{suffix}_owner",
            migrator=f"{role_prefix}_{suffix}_migrator",
            legacy_runtime=f"{role_prefix}_{suffix}_legacy",
            atomic_runtime=f"{role_prefix}_{suffix}_atomic",
            activation=f"{role_prefix}_{suffix}_activation",
            readiness=f"{role_prefix}_{suffix}_readiness",
            trainer=f"{role_prefix}_{suffix}_trainer",
        )
    context = PostgresBootstrapContext(
        expected_database=_TARGET_DATABASE,
        admin_role=_ADMIN_ROLE,
        roles=roles,
        adoption=None,
    )
    manifest = (
        ("schema_owner", roles.schema_owner),
        ("migrator", roles.migrator),
        ("legacy_runtime", roles.legacy_runtime),
        ("atomic_runtime", roles.atomic_runtime),
        ("activation", roles.activation),
        ("readiness", roles.readiness),
        ("trainer", roles.trainer),
    )
    login_roles = tuple(role for purpose, role in manifest if purpose != "schema_owner")
    passwords = {
        role: f"test-only-{suffix}-{index}-{secrets.token_hex(8)}"
        for index, role in enumerate(login_roles)
    }
    admin = _connect(target_dsn)
    try:
        with admin.cursor() as cursor:
            for purpose, role in manifest:
                cursor.execute(
                    sql.SQL(
                        "CREATE ROLE {} NOLOGIN NOSUPERUSER NOINHERIT NOCREATEDB "
                        "NOCREATEROLE NOREPLICATION NOBYPASSRLS "
                        "CONNECTION LIMIT -1 PASSWORD NULL"
                    ).format(sql.Identifier(role))
                )
                cursor.execute(
                    sql.SQL("COMMENT ON ROLE {} IS %s").format(sql.Identifier(role)),
                    (f"elvis-postgres-bootstrap:v1:{_TARGET_DATABASE}:{purpose}",),
                )
            cursor.execute(
                sql.SQL("GRANT {} TO {}").format(
                    sql.Identifier(roles.schema_owner),
                    sql.Identifier(roles.migrator),
                )
            )
            cursor.execute(
                sql.SQL("REVOKE CREATE ON DATABASE {} FROM PUBLIC").format(
                    sql.Identifier(_TARGET_DATABASE)
                )
            )
            cursor.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(_TARGET_DATABASE),
                    sql.Identifier(roles.schema_owner),
                )
            )
            cursor.execute(
                sql.SQL("CREATE SCHEMA np AUTHORIZATION {}").format(
                    sql.Identifier(roles.schema_owner)
                )
            )
            cursor.execute(
                "COMMENT ON SCHEMA np IS %s",
                (f"elvis-postgres-bootstrap-schema:v1:{_TARGET_DATABASE}",),
            )
        admin.commit()
        admin.autocommit = True
        with admin.cursor() as cursor:
            for role, password in passwords.items():
                cursor.execute(
                    sql.SQL("ALTER ROLE {} LOGIN PASSWORD %s").format(
                        sql.Identifier(role)
                    ),
                    (password,),
                )
    finally:
        admin.close()

    migrator_dsn = make_dsn(
        target_dsn,
        user=roles.migrator,
        password=passwords[roles.migrator],
    )
    migrator = _connect(migrator_dsn)
    try:
        migrator.autocommit = True
        with migrator.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET ROLE {}").format(sql.Identifier(roles.schema_owner))
            )
        migrator.autocommit = False
        assert apply_migrations(migrator, load_migrations()[:6]) == (
            1,
            2,
            3,
            4,
            5,
            6,
        )
    finally:
        migrator.close()

    admin = _connect(target_dsn)
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER SCHEMA np OWNER TO {}").format(
                    sql.Identifier(roles.schema_owner)
                )
            )
            for table in _HISTORICAL_HEAD6_AUTHORITY_TABLES:
                cursor.execute(
                    sql.SQL("ALTER TABLE np.{} OWNER TO {}").format(
                        sql.Identifier(table),
                        sql.Identifier(roles.schema_owner),
                    )
                )
            for sequence in _HISTORICAL_HEAD6_LEGACY_SEQUENCES:
                cursor.execute(
                    sql.SQL("ALTER SEQUENCE np.{} OWNER TO {}").format(
                        sql.Identifier(sequence),
                        sql.Identifier(roles.schema_owner),
                    )
                )
            for function in _HISTORICAL_HEAD6_NON_ACTIVATION_FUNCTIONS:
                cursor.execute(
                    sql.SQL("ALTER FUNCTION {} OWNER TO {}").format(
                        sql.SQL(function),
                        sql.Identifier(roles.schema_owner),
                    )
                )
            for function in _HISTORICAL_HEAD6_ACTIVATION_FUNCTIONS:
                cursor.execute(
                    sql.SQL("ALTER FUNCTION {} OWNER TO {}").format(
                        sql.SQL(function),
                        sql.Identifier(roles.activation),
                    )
                )

            cursor.execute("REVOKE ALL ON SCHEMA np FROM PUBLIC")
            cursor.execute(
                sql.SQL("GRANT USAGE ON SCHEMA np TO {}").format(
                    sql.SQL(", ").join(sql.Identifier(role) for role in login_roles)
                )
            )
            for role in login_roles:
                cursor.execute(
                    sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA np FROM {}").format(
                        sql.Identifier(role)
                    )
                )
                cursor.execute(
                    sql.SQL("REVOKE ALL ON ALL SEQUENCES IN SCHEMA np FROM {}").format(
                        sql.Identifier(role)
                    )
                )
                cursor.execute(
                    sql.SQL("REVOKE ALL ON ALL FUNCTIONS IN SCHEMA np FROM {}").format(
                        sql.Identifier(role)
                    )
                )
            cursor.execute("REVOKE ALL ON ALL TABLES IN SCHEMA np FROM PUBLIC")
            cursor.execute("REVOKE ALL ON ALL SEQUENCES IN SCHEMA np FROM PUBLIC")
            cursor.execute("REVOKE ALL ON ALL FUNCTIONS IN SCHEMA np FROM PUBLIC")

            for table, privileges in _HISTORICAL_HEAD6_LEGACY_PRIVILEGES.items():
                cursor.execute(
                    sql.SQL("GRANT {} ON TABLE np.{} TO {}").format(
                        sql.SQL(", ").join(sql.SQL(item) for item in privileges),
                        sql.Identifier(table),
                        sql.Identifier(roles.legacy_runtime),
                    )
                )
            for table, privileges in _HISTORICAL_HEAD6_ATOMIC_PRIVILEGES.items():
                cursor.execute(
                    sql.SQL("GRANT {} ON TABLE np.{} TO {}").format(
                        sql.SQL(", ").join(sql.SQL(item) for item in privileges),
                        sql.Identifier(table),
                        sql.Identifier(roles.atomic_runtime),
                    )
                )
            cursor.execute(
                sql.SQL(
                    "GRANT SELECT ON TABLE np.paper_runtime_control, "
                    "np.paper_runtime_generations TO {}"
                ).format(sql.Identifier(roles.atomic_runtime))
            )
            cursor.execute(
                sql.SQL("GRANT SELECT ON TABLE {} TO {}").format(
                    sql.SQL(", ").join(
                        sql.SQL("np.{}").format(sql.Identifier(table))
                        for table in _HISTORICAL_HEAD6_AUTHORITY_TABLES
                    ),
                    sql.Identifier(roles.readiness),
                )
            )
            cursor.execute(
                sql.SQL("GRANT SELECT ON TABLE np.trades TO {}").format(
                    sql.Identifier(roles.trainer)
                )
            )
            cursor.execute(
                sql.SQL("GRANT USAGE ON SEQUENCE {} TO {}").format(
                    sql.SQL(", ").join(
                        sql.SQL("np.{}").format(sql.Identifier(sequence))
                        for sequence in _HISTORICAL_HEAD6_LEGACY_SEQUENCES
                    ),
                    sql.Identifier(roles.legacy_runtime),
                )
            )
            cursor.execute(
                sql.SQL(
                    "GRANT UPDATE(control_key) ON np.paper_runtime_control TO {}"
                ).format(sql.Identifier(roles.atomic_runtime))
            )
            cursor.execute(
                sql.SQL("GRANT UPDATE(id) ON np.open_positions TO {}").format(
                    sql.Identifier(roles.legacy_runtime)
                )
            )
            cursor.execute(
                sql.SQL(
                    "GRANT UPDATE(activation_id) "
                    "ON np.paper_runtime_generations TO {}"
                ).format(sql.Identifier(roles.atomic_runtime))
            )
            cursor.execute(
                sql.SQL("GRANT SELECT ON TABLE {} TO {}").format(
                    sql.SQL(", ").join(
                        sql.SQL("np.{}").format(sql.Identifier(table))
                        for table in _HISTORICAL_HEAD6_AUTHORITY_TABLES
                    ),
                    sql.Identifier(roles.activation),
                )
            )
            for table in _HISTORICAL_HEAD6_AUTHORITY_TABLES:
                cursor.execute(
                    sql.SQL("GRANT UPDATE ON TABLE np.{} TO {}").format(
                        sql.Identifier(table),
                        sql.Identifier(roles.activation),
                    )
                )
            cursor.execute(
                sql.SQL("GRANT INSERT ON np.paper_runtime_generations TO {}").format(
                    sql.Identifier(roles.activation)
                )
            )
            for function in _HISTORICAL_HEAD6_ACTIVATION_FUNCTIONS:
                cursor.execute(
                    sql.SQL("REVOKE ALL ON FUNCTION {} FROM PUBLIC").format(
                        sql.SQL(function)
                    )
                )
                cursor.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(
                        sql.SQL(function),
                        sql.Identifier(roles.activation),
                    )
                )
            cursor.execute(
                sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(
                    sql.Identifier(_TARGET_DATABASE)
                )
            )
            cursor.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                    sql.Identifier(_TARGET_DATABASE),
                    sql.Identifier(_ADMIN_ROLE),
                )
            )
            cursor.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(_TARGET_DATABASE),
                    sql.Identifier(roles.schema_owner),
                )
            )
            cursor.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(_TARGET_DATABASE),
                    sql.SQL(", ").join(sql.Identifier(role) for role in login_roles),
                )
            )
            cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")
        admin.commit()
    finally:
        admin.close()
    return context, migrator_dsn


def _bootstrap_target(target_dsn: str, suffix: str) -> PostgresBootstrapContext:
    context, _migrator_dsn = _bootstrap_historical_head6_target(
        target_dsn,
        suffix,
        role_prefix="ct",
    )
    return context


def _prepare_source(source_dsn: str) -> None:
    connection = _connect(source_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(load_migrations()[0].sql)
            cursor.execute(
                "INSERT INTO np.trades "
                "(id, timestamp, symbol, side, price, quantity, pnl, fee) "
                "VALUES "
                "(7, '2026-08-13 10:11:12', 'BTC|USDT', 'BUY', "
                "123.5, 0.25, 1.5, 0.01), "
                "(8, '2026-08-13 10:11:13', 'ETHUSDT', 'SELL', "
                "234.5, 0.5, -2.5, 0.02), "
                "(9, '2026-08-13 10:11:14', 'XRPUSDT', 'BUY', "
                "0.5, 3.0, 0.0, 0.03)"
            )
            cursor.execute(
                "INSERT INTO np.liquidations "
                "(id, timestamp, symbol, entry_price, liquidation_price, "
                "quantity, leverage, liquidation_fee) VALUES "
                "(3, '2026-08-13 10:11:13', 'ETHUSDT', 10, 8, 2, 3, 0.2)"
            )
            cursor.execute(
                "INSERT INTO np.margin_history "
                "(id, timestamp, balance, used_margin, open_positions) VALUES "
                "(2, '2026-08-13 10:11:14', 1000, 20, 0)"
            )
            cursor.execute(
                "INSERT INTO np.trading_session_resets "
                "(id, reset_timestamp, reason) VALUES "
                "(5, '2026-08-13 10:11:15', 'cutover|fixture')"
            )
            cursor.execute(
                "INSERT INTO np.model_predictions "
                "(id, created_at, symbol, side, model, vote, scored) VALUES "
                "(11, '2026-08-13 10:11:16', 'BTCUSDT', 'BUY', "
                "'model\\name', 'BUY', true)"
            )
            cursor.execute(
                "INSERT INTO np.account_balances "
                "(id, asset, balance, last_updated) VALUES "
                "(13, 'USDT', 1000, '2026-08-13 10:11:17')"
            )
        connection.commit()
    finally:
        connection.close()


def _source_relation_names(dsn: str) -> tuple[str, ...]:
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT relname FROM pg_class "
                "WHERE relnamespace = 'np'::regnamespace AND relkind = 'r' "
                "ORDER BY relname"
            )
            return tuple(row[0] for row in cursor.fetchall())
    finally:
        connection.close()


def _database_snapshot(dsn: str) -> tuple[object, ...]:
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT n.nspname, c.relname, c.relkind, "
                "pg_get_userbyid(c.relowner), c.relacl::text "
                "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'np' ORDER BY c.relkind, c.relname"
            )
            catalog = tuple(cursor.fetchall())
            cursor.execute(
                "SELECT object_kind, object_name FROM ("
                "SELECT 'collation'::text AS object_kind, collname::text AS object_name "
                "FROM pg_collation c JOIN pg_namespace n ON n.oid = c.collnamespace "
                "WHERE n.nspname = 'np' UNION ALL "
                "SELECT 'operator_family', opfname FROM pg_opfamily o "
                "JOIN pg_namespace n ON n.oid = o.opfnamespace "
                "WHERE n.nspname = 'np' UNION ALL "
                "SELECT 'text_search_configuration', cfgname FROM pg_ts_config c "
                "JOIN pg_namespace n ON n.oid = c.cfgnamespace "
                "WHERE n.nspname = 'np' UNION ALL "
                "SELECT 'statistics', stxname FROM pg_statistic_ext s "
                "JOIN pg_namespace n ON n.oid = s.stxnamespace "
                "WHERE n.nspname = 'np' UNION ALL "
                "SELECT 'routine', proname || '(' || pg_get_function_identity_arguments(p.oid) || ')' "
                "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
                "WHERE n.nspname = 'np' UNION ALL "
                "SELECT 'standalone_type', typname FROM pg_type t "
                "JOIN pg_namespace n ON n.oid = t.typnamespace "
                "WHERE n.nspname = 'np' AND t.typrelid = 0 AND t.typelem = 0 UNION ALL "
                "SELECT 'trigger', tgname FROM pg_trigger t "
                "JOIN pg_class c ON c.oid = t.tgrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'np' AND NOT t.tgisinternal"
                ") roots ORDER BY object_kind, object_name"
            )
            roots = tuple(cursor.fetchall())
            cursor.execute(
                "SELECT pg_get_userbyid(a.defaclrole), COALESCE(n.nspname, ''), "
                "a.defaclobjtype, a.defaclacl::text FROM pg_default_acl a "
                "LEFT JOIN pg_namespace n ON n.oid = a.defaclnamespace "
                "WHERE a.defaclnamespace = 0 OR n.nspname = 'np' "
                "ORDER BY 1, 2, 3, 4"
            )
            default_acls = tuple(cursor.fetchall())
            cursor.execute("SELECT to_regclass('np.schema_migrations')")
            migrations = ()
            if cursor.fetchone()[0] is not None:
                cursor.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'np' AND table_name = 'schema_migrations' "
                    "ORDER BY ordinal_position"
                )
                migration_columns = tuple(row[0] for row in cursor.fetchall())
                if migration_columns == (
                    "version",
                    "name",
                    "checksum",
                    "applied_at",
                ):
                    cursor.execute(
                        "SELECT version, name, checksum, applied_at "
                        "FROM np.schema_migrations ORDER BY version"
                    )
                    migrations = tuple(cursor.fetchall())
                else:
                    migrations = ("MALFORMED", migration_columns)
            cursor.execute("SELECT to_regclass('np.paper_runtime_control')")
            control = ()
            if cursor.fetchone()[0] is not None:
                cursor.execute(
                    "SELECT mode, runtime_generation, updated_at "
                    "FROM np.paper_runtime_control ORDER BY control_key"
                )
                control = tuple(cursor.fetchall())
            data = []
            for relation in _LEGACY_RELATIONS:
                cursor.execute(
                    sql.SQL("SELECT * FROM {} ORDER BY id").format(
                        sql.Identifier(*relation.split("."))
                    )
                )
                data.append((relation, tuple(cursor.fetchall())))
        connection.rollback()
        return catalog, roots, default_acls, migrations, control, tuple(data)
    finally:
        connection.close()


@pytest.fixture
def cutover_pair(tmp_path: Path) -> CutoverPair:
    project = f"elvis-v2-cutover-{uuid4().hex[:12]}"
    network = f"{project}-internal"
    source_container = f"{project}-source"
    target_container = f"{project}-target"
    password = secrets.token_hex(24)
    password_file = tmp_path / "postgres-password"
    _write_private(password_file, password + "\n")
    label = f"org.elvis.v2.cutover-test={project}"
    containers = (
        (source_container, _SOURCE_DATABASE),
        (target_container, _TARGET_DATABASE),
    )
    volumes = (
        f"{project}-source-data",
        f"{project}-target-data",
    )

    try:
        _create_isolated_network(network, label, project.rsplit("-", 1)[-1])
        for volume in volumes:
            _run(["docker", "volume", "create", "--label", label, volume])
        for (container, database), volume in zip(containers, volumes):
            _run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--name",
                    container,
                    "--label",
                    label,
                    "--network",
                    network,
                    "--publish",
                    "127.0.0.1:0:5432",
                    "--env",
                    f"POSTGRES_DB={database}",
                    "--env",
                    f"POSTGRES_USER={_ADMIN_ROLE}",
                    "--env",
                    "POSTGRES_PASSWORD_FILE=/run/operator/password",
                    "--volume",
                    f"{password_file}:/run/operator/password:ro",
                    "--volume",
                    f"{volume}:/var/lib/postgresql/data",
                    _POSTGRES_IMAGE,
                ],
                secrets_to_redact=(password,),
            )
            _wait_for_postgres(container, database)

        source_dsn = make_dsn(
            host="127.0.0.1",
            port=_published_port(source_container),
            dbname=_SOURCE_DATABASE,
            user=_ADMIN_ROLE,
            password=password,
            connect_timeout=5,
        )
        target_dsn = make_dsn(
            host="127.0.0.1",
            port=_published_port(target_container),
            dbname=_TARGET_DATABASE,
            user=_ADMIN_ROLE,
            password=password,
            connect_timeout=5,
        )
        _prepare_source(source_dsn)
        suffix = project.rsplit("-", 1)[-1][:10]
        target_context = _bootstrap_target(target_dsn, suffix)
        yield CutoverPair(
            project=project,
            source_container=source_container,
            target_container=target_container,
            source_dsn=source_dsn,
            target_dsn=target_dsn,
            password=password,
            context=FreshTargetCutoverContext(
                source_expected_database=_SOURCE_DATABASE,
                source_expected_role=_ADMIN_ROLE,
                target_bootstrap_intent=FreshTargetBootstrapIntent(
                    expected_database=target_context.expected_database,
                    admin_role=target_context.admin_role,
                    roles=FreshTargetRoleManifest(
                        **{
                            key: getattr(target_context.roles, key)
                            for key in (
                                "schema_owner",
                                "migrator",
                                "legacy_runtime",
                                "atomic_runtime",
                                "activation",
                                "readiness",
                                "trainer",
                            )
                        }
                    ),
                ),
            ),
        )
    finally:
        _run(
            ["docker", "rm", "--force", source_container, target_container],
            expected_exit_codes=(0, 1),
            secrets_to_redact=(password,),
        )
        _run(
            ["docker", "volume", "rm", "--force", *volumes],
            expected_exit_codes=(0, 1),
            secrets_to_redact=(password,),
        )
        _run(
            ["docker", "network", "rm", network],
            expected_exit_codes=(0, 1),
            secrets_to_redact=(password,),
        )
        _assert_project_absent(project)


def test_separate_clusters_are_ready_repeatable_and_strictly_read_only(
    cutover_pair: CutoverPair,
    monkeypatch,
) -> None:
    pair = cutover_pair
    before = (
        _database_snapshot(pair.source_dsn),
        _database_snapshot(pair.target_dsn),
    )
    records: list[ConnectionRecord] = []
    assert _source_relation_names(pair.source_dsn) == tuple(
        relation.removeprefix("np.") for relation in _LEGACY_RELATIONS
    )
    assert "schema_migrations" not in _source_relation_names(pair.source_dsn)

    first = pair.inspector(records=records).inspect(pair.context)
    monkeypatch.setattr(preflight_module, "_FETCH_BATCH_SIZE", 1)
    second = pair.inspector(records=records).inspect(pair.context)

    assert first == second
    assert first.status is FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET
    assert first.blockers == ()
    assert first.source.system_identifier != first.target.system_identifier
    assert tuple(item.name for item in first.source.relations) == _LEGACY_RELATIONS
    assert {item.name: item.sha256 for item in first.source.relations} == (
        _EXPECTED_RELATION_SHA256
    )
    trades = next(item for item in first.source.relations if item.name == "np.trades")
    assert (trades.row_count, trades.pk_min, trades.pk_max) == (3, 7, 9)
    assert first.source.canonical_sha256 == _EXPECTED_SOURCE_SHA256
    assert first.stale_on_return is True
    assert first.snapshot_authoritative is False
    assert records
    assert all(record.commits == 0 for record in records)
    assert all(record.rollbacks == 1 for record in records)
    assert all(
        record.statements
        and record.statements[0].strip().upper()
        == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
        for record in records
    )
    assert all(
        _MUTATING_SQL.search(statement) is None
        for record in records
        for statement in record.statements
    )
    assert (
        _database_snapshot(pair.source_dsn),
        _database_snapshot(pair.target_dsn),
    ) == before


def test_source_failures_block_independently_and_restore_exactly(
    cutover_pair: CutoverPair,
) -> None:
    pair = cutover_pair
    inspector = pair.inspector()

    wrong_identity = inspector.inspect(
        replace(pair.context, source_expected_database="wrong_source")
    )
    assert FreshTargetCutoverBlocker.SOURCE_IDENTITY in wrong_identity.blockers

    pair.execute(
        pair.source_dsn,
        "CREATE ROLE expected_source_role NOLOGIN",
    )
    try:
        context = replace(pair.context, source_expected_role="expected_source_role")

        def session_authorized_factory():
            connection = pair.source_factory()
            with connection.cursor() as cursor:
                cursor.execute("SET SESSION AUTHORIZATION expected_source_role")
            connection.commit()
            return connection

        spoofed = pair.inspector(source_factory=session_authorized_factory).inspect(
            context
        )
        assert FreshTargetCutoverBlocker.SOURCE_IDENTITY in spoofed.blockers
    finally:
        pair.execute(pair.source_dsn, "DROP ROLE expected_source_role")

    same_cluster = pair.inspector(target_factory=pair.source_factory).inspect(
        pair.context
    )
    assert FreshTargetCutoverBlocker.SAME_CLUSTER in same_cluster.blockers

    held_connection = pair.source_factory()
    try:
        with held_connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        busy = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_ACTIVE_SESSIONS in busy.blockers
    finally:
        held_connection.rollback()
        held_connection.close()

    pair.execute(pair.source_dsn, "ALTER TABLE np.trades RENAME TO trades_missing")
    try:
        missing = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in missing.blockers
    finally:
        pair.execute(pair.source_dsn, "ALTER TABLE np.trades_missing RENAME TO trades")

    pair.execute(pair.source_dsn, "ALTER TABLE np.trades ADD COLUMN unexpected TEXT")
    try:
        drifted = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in drifted.blockers
    finally:
        pair.execute(pair.source_dsn, "ALTER TABLE np.trades DROP COLUMN unexpected")

    for replacement in (
        "CREATE INDEX idx_trades_symbol_ts ON np.trades (symbol DESC, timestamp)",
        "CREATE INDEX idx_trades_symbol_ts ON np.trades "
        "(symbol text_pattern_ops, timestamp)",
        'CREATE INDEX idx_trades_symbol_ts ON np.trades (symbol COLLATE "C", timestamp)',
    ):
        pair.execute(pair.source_dsn, "DROP INDEX np.idx_trades_symbol_ts")
        pair.execute(pair.source_dsn, replacement)
        try:
            before_index = _database_snapshot(pair.source_dsn)
            index_drift = inspector.inspect(pair.context)
            assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in index_drift.blockers
            assert _database_snapshot(pair.source_dsn) == before_index
        finally:
            pair.execute(pair.source_dsn, "DROP INDEX np.idx_trades_symbol_ts")
            pair.execute(
                pair.source_dsn,
                "CREATE INDEX idx_trades_symbol_ts " "ON np.trades (symbol, timestamp)",
            )

    pair.execute(pair.source_dsn, "CREATE TABLE np.unexpected_root (id INTEGER)")
    try:
        extra_table = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in extra_table.blockers
    finally:
        pair.execute(pair.source_dsn, "DROP TABLE np.unexpected_root")

    pair.execute(
        pair.source_dsn,
        "CREATE FUNCTION np.unexpected_root() RETURNS INTEGER "
        "LANGUAGE SQL IMMUTABLE AS 'SELECT 1'",
    )
    try:
        extra_function = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in extra_function.blockers
    finally:
        pair.execute(pair.source_dsn, "DROP FUNCTION np.unexpected_root()")

    pair.execute(
        pair.source_dsn,
        "CREATE TYPE np.unexpected_root AS ENUM ('unexpected')",
    )
    try:
        extra_type = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in extra_type.blockers
    finally:
        pair.execute(pair.source_dsn, "DROP TYPE np.unexpected_root")

    pair.execute(pair.source_dsn, "CREATE ROLE cutover_default_acl_outsider NOLOGIN")
    pair.execute(
        pair.source_dsn,
        "ALTER DEFAULT PRIVILEGES FOR ROLE cutover_default_acl_outsider "
        "GRANT SELECT ON TABLES TO PUBLIC",
    )
    try:
        before_default_acl = _database_snapshot(pair.source_dsn)
        default_acl = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in default_acl.blockers
        assert _database_snapshot(pair.source_dsn) == before_default_acl
    finally:
        pair.execute(
            pair.source_dsn,
            "ALTER DEFAULT PRIVILEGES FOR ROLE cutover_default_acl_outsider "
            "REVOKE SELECT ON TABLES FROM PUBLIC",
        )
        pair.execute(pair.source_dsn, "DROP OWNED BY cutover_default_acl_outsider")
        pair.execute(pair.source_dsn, "DROP ROLE cutover_default_acl_outsider")

    pair.execute(
        pair.source_dsn,
        "CREATE COLLATION np.unexpected_root (provider = libc, locale = 'C')",
    )
    try:
        before_collation = _database_snapshot(pair.source_dsn)
        extra_collation = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in extra_collation.blockers
        assert _database_snapshot(pair.source_dsn) == before_collation
    finally:
        pair.execute(pair.source_dsn, "DROP COLLATION np.unexpected_root")

    pair.execute(
        pair.source_dsn,
        "CREATE TEXT SEARCH CONFIGURATION np.unexpected_root "
        "(COPY = pg_catalog.simple)",
    )
    try:
        before_text_search = _database_snapshot(pair.source_dsn)
        extra_text_search = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in extra_text_search.blockers
        assert _database_snapshot(pair.source_dsn) == before_text_search
    finally:
        pair.execute(
            pair.source_dsn,
            "DROP TEXT SEARCH CONFIGURATION np.unexpected_root",
        )

    pair.execute(
        pair.source_dsn,
        "CREATE STATISTICS np.unexpected_root (dependencies) ON symbol, side "
        "FROM np.trades",
    )
    try:
        before_statistics = _database_snapshot(pair.source_dsn)
        extra_statistics = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in extra_statistics.blockers
        assert _database_snapshot(pair.source_dsn) == before_statistics
    finally:
        pair.execute(pair.source_dsn, "DROP STATISTICS np.unexpected_root")

    pair.execute(
        pair.source_dsn,
        "CREATE OPERATOR FAMILY np.unexpected_root USING btree",
    )
    try:
        before_operator_family = _database_snapshot(pair.source_dsn)
        extra_operator_family = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in (
            extra_operator_family.blockers
        )
        assert _database_snapshot(pair.source_dsn) == before_operator_family
    finally:
        pair.execute(
            pair.source_dsn,
            "DROP OPERATOR FAMILY np.unexpected_root USING btree",
        )

    pair.execute(
        pair.source_dsn,
        "CREATE FUNCTION public.elvis_cutover_test_trigger() RETURNS TRIGGER "
        "LANGUAGE plpgsql AS 'BEGIN RETURN NEW; END'",
    )
    pair.execute(
        pair.source_dsn,
        "CREATE TRIGGER elvis_cutover_test_trigger BEFORE INSERT ON np.trades "
        "FOR EACH ROW EXECUTE FUNCTION public.elvis_cutover_test_trigger()",
    )
    try:
        extra_trigger = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_SCHEMA in extra_trigger.blockers
    finally:
        pair.execute(
            pair.source_dsn,
            "DROP TRIGGER elvis_cutover_test_trigger ON np.trades",
        )
        pair.execute(
            pair.source_dsn,
            "DROP FUNCTION public.elvis_cutover_test_trigger()",
        )

    pair.execute(
        pair.source_dsn,
        "INSERT INTO np.open_positions "
        "(id, symbol, side, entry_price, quantity, leverage, entry_time) "
        "VALUES (19, 'BTCUSDT', 'BUY', 10, 1, 2, '2026-08-13 12:00:00')",
    )
    try:
        opened = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_OPEN_POSITIONS in opened.blockers
    finally:
        pair.execute(pair.source_dsn, "DELETE FROM np.open_positions WHERE id = 19")

    pair.execute(pair.source_dsn, "UPDATE np.trades SET symbol = NULL WHERE id = 7")
    try:
        invalid = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_DATA_QUALITY in invalid.blockers
    finally:
        pair.execute(
            pair.source_dsn,
            "UPDATE np.trades SET symbol = 'BTC|USDT' WHERE id = 7",
        )

    pair.execute(
        pair.source_dsn,
        "UPDATE np.trades SET side = '105000.0' WHERE id = 7",
    )
    try:
        poisoned_side = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.SOURCE_DATA_QUALITY in poisoned_side.blockers
    finally:
        pair.execute(pair.source_dsn, "UPDATE np.trades SET side = 'BUY' WHERE id = 7")

    restored = inspector.inspect(pair.context)
    assert restored.status is FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET


def test_target_terminal_mode_migration_and_data_drift_block_without_repair(
    cutover_pair: CutoverPair,
) -> None:
    pair = cutover_pair
    inspector = pair.inspector()
    roles = pair.context.target_bootstrap_intent.roles

    pair.execute(
        pair.target_dsn,
        sql.SQL("ALTER ROLE {} CREATEDB").format(sql.Identifier(roles.trainer)),
    )
    try:
        incomplete = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.TARGET_NOT_COMPLETE in incomplete.blockers
    finally:
        pair.execute(
            pair.target_dsn,
            sql.SQL("ALTER ROLE {} NOCREATEDB").format(sql.Identifier(roles.trainer)),
        )

    pair.execute(
        pair.target_dsn,
        "UPDATE np.schema_migrations SET checksum = %s WHERE version = 1",
        ("f" * 64,),
    )
    try:
        migration_drift = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.TARGET_NOT_COMPLETE in migration_drift.blockers
    finally:
        expected_checksum = load_migrations()[0].checksum
        pair.execute(
            pair.target_dsn,
            "UPDATE np.schema_migrations SET checksum = %s WHERE version = 1",
            (expected_checksum,),
        )

    pair.execute(
        pair.target_dsn,
        "ALTER TABLE np.schema_migrations RENAME TO schema_migrations_missing",
    )
    try:
        before_missing_ledger = _database_snapshot(pair.target_dsn)
        missing_ledger = inspector.inspect(pair.context)
        assert missing_ledger.status is FreshTargetCutoverStatus.BLOCKED
        assert FreshTargetCutoverBlocker.TARGET_NOT_COMPLETE in missing_ledger.blockers
        assert _database_snapshot(pair.target_dsn) == before_missing_ledger
    finally:
        pair.execute(
            pair.target_dsn,
            "ALTER TABLE np.schema_migrations_missing RENAME TO schema_migrations",
        )

    pair.execute(
        pair.target_dsn,
        "ALTER TABLE np.schema_migrations RENAME TO schema_migrations_expected",
    )
    pair.execute(
        pair.target_dsn,
        "CREATE TABLE np.schema_migrations (unexpected TEXT)",
    )
    try:
        before_malformed_ledger = _database_snapshot(pair.target_dsn)
        malformed_ledger = inspector.inspect(pair.context)
        assert malformed_ledger.status is FreshTargetCutoverStatus.BLOCKED
        assert FreshTargetCutoverBlocker.TARGET_NOT_COMPLETE in (
            malformed_ledger.blockers
        )
        assert _database_snapshot(pair.target_dsn) == before_malformed_ledger
    finally:
        pair.execute(pair.target_dsn, "DROP TABLE np.schema_migrations")
        pair.execute(
            pair.target_dsn,
            "ALTER TABLE np.schema_migrations_expected RENAME TO schema_migrations",
        )

    function_connection = pair.connect(pair.target_dsn)
    try:
        with function_connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'np.enforce_legacy_paper_runtime_fence()'::regprocedure)"
            )
            original_function = cursor.fetchone()[0]
        function_connection.rollback()
    finally:
        function_connection.close()
    pair.execute(
        pair.target_dsn,
        """
        CREATE OR REPLACE FUNCTION np.enforce_legacy_paper_runtime_fence()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        BEGIN
            RAISE EXCEPTION USING
                ERRCODE = '55000',
                MESSAGE = 'poisoned historical function was executed';
        END
        $function$
        """,
    )
    try:
        before_poisoned_function = _database_snapshot(pair.target_dsn)
        poisoned_function = inspector.inspect(pair.context)
        assert poisoned_function.status is FreshTargetCutoverStatus.BLOCKED
        assert poisoned_function.target.terminal_catalog_exact is False
        assert FreshTargetCutoverBlocker.TARGET_NOT_COMPLETE in (
            poisoned_function.blockers
        )
        assert _database_snapshot(pair.target_dsn) == before_poisoned_function
    finally:
        pair.execute(pair.target_dsn, original_function)

    pair.execute(
        pair.target_dsn,
        "ALTER TABLE np.trades " "DISABLE TRIGGER legacy_paper_runtime_fence_trades",
    )
    try:
        before_disabled_trigger = _database_snapshot(pair.target_dsn)
        disabled_trigger = inspector.inspect(pair.context)
        assert disabled_trigger.status is FreshTargetCutoverStatus.BLOCKED
        assert disabled_trigger.target.terminal_catalog_exact is False
        assert FreshTargetCutoverBlocker.TARGET_NOT_COMPLETE in (
            disabled_trigger.blockers
        )
        assert _database_snapshot(pair.target_dsn) == before_disabled_trigger
    finally:
        pair.execute(
            pair.target_dsn,
            "ALTER TABLE np.trades "
            "ENABLE ALWAYS TRIGGER legacy_paper_runtime_fence_trades",
        )

    pair.execute(
        pair.target_dsn,
        "UPDATE np.paper_runtime_control SET mode = 'SHADOW' WHERE control_key IS TRUE",
    )
    try:
        wrong_mode = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.TARGET_MODE in wrong_mode.blockers
    finally:
        pair.execute(
            pair.target_dsn,
            "UPDATE np.paper_runtime_control SET mode = 'LEGACY' "
            "WHERE control_key IS TRUE",
        )

    pair.execute(
        pair.target_dsn,
        "UPDATE np.paper_runtime_control SET runtime_generation = 1 "
        "WHERE control_key IS TRUE",
    )
    try:
        wrong_generation = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.TARGET_MODE in wrong_generation.blockers
    finally:
        pair.execute(
            pair.target_dsn,
            "UPDATE np.paper_runtime_control SET runtime_generation = 0 "
            "WHERE control_key IS TRUE",
        )

    pair.execute(
        pair.target_dsn,
        "INSERT INTO np.trades "
        "(id, timestamp, symbol, side, price, quantity, pnl, fee) "
        "VALUES (23, '2026-08-13 14:00:00', 'BTCUSDT', 'BUY', 1, 1, 0, 0)",
    )
    try:
        nonempty = inspector.inspect(pair.context)
        assert FreshTargetCutoverBlocker.TARGET_NOT_EMPTY in nonempty.blockers
        assert "np.trades" in nonempty.target.nonempty_relations
    finally:
        pair.execute(pair.target_dsn, "DELETE FROM np.trades WHERE id = 23")

    restored = inspector.inspect(pair.context)
    assert restored.status is FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET
