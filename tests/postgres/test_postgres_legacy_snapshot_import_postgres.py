"""Two-cluster PostgreSQL 15 proof for the bounded legacy snapshot importer."""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable
from uuid import uuid4

import pytest
from psycopg2 import OperationalError, sql
from psycopg2.extensions import make_dsn

from scripts import postgres_legacy_snapshot_import as import_cli
from tests.conftest import _ORIGINAL_PSYCOPG2_CONNECT
from tests.postgres.legacy_snapshot_import_support import (
    AfterTargetIdentityConnection,
    CommitNotSentConnection,
    CommitUnknownConnection,
    FailBeforeCommitConnection,
    FailOnSequenceConnection,
    RecordingConnection,
    SqlEventLog,
    first_event_index,
    statement_keyword,
)
from tests.postgres.test_postgres_cutover_preflight_postgres import (
    _POSTGRES_IMAGE,
    _bootstrap_historical_head6_target,
    _create_isolated_network,
    _database_snapshot,
    _published_port,
    _run,
    _wait_for_postgres,
    _write_private,
)
from trading.application.fresh_target_cutover import (
    FreshTargetBootstrapIntent,
    FreshTargetCutoverContext,
    FreshTargetCutoverStatus,
    FreshTargetRoleManifest,
)
from trading.application.legacy_snapshot_import import (
    LegacySnapshotImportContext,
    LegacySnapshotImportDisposition,
)
from trading.persistence import load_migrations
from trading.persistence.postgres_bootstrap import (
    PostgresBootstrap,
    PostgresBootstrapContext,
    PostgresBootstrapRoles,
)
from trading.persistence.postgres_cutover_preflight import PostgresCutoverPreflight
from trading.persistence.postgres_legacy_snapshot_import import (
    PostgresLegacySnapshotImport,
    PostgresLegacySnapshotImportBusyError,
    PostgresLegacySnapshotImportCommitUnknown,
    PostgresLegacySnapshotImportConflict,
    PostgresLegacySnapshotImportStorageError,
)

_REQUIRED_ENV = "ELVIS_TEST_V2_LEGACY_SNAPSHOT_IMPORT_REQUIRED"
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
_LEGACY_SEQUENCES = tuple(
    f"{relation.removeprefix('np.')}_id_seq" for relation in _LEGACY_RELATIONS
)
_FORBIDDEN_WRITE_KEYWORDS = frozenset(
    {
        "ALTER",
        "CALL",
        "CREATE",
        "DELETE",
        "DO",
        "DROP",
        "GRANT",
        "REVOKE",
        "TRUNCATE",
        "UPDATE",
    }
)

pytestmark = pytest.mark.skipif(
    os.getenv(_REQUIRED_ENV) != "1",
    reason=f"set {_REQUIRED_ENV}=1 to run the isolated two-cluster importer",
)


def _assert_project_absent(project: str) -> None:
    label = f"org.elvis.v2.snapshot-import-test={project}"
    for resource, arguments in (
        ("container", ["container", "ls", "--all"]),
        ("network", ["network", "ls"]),
        ("volume", ["volume", "ls"]),
    ):
        result = _run(["docker", *arguments, "--quiet", "--filter", f"label={label}"])
        assert result.stdout.strip() == "", f"residual {resource} for {project}"


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


def _execute(dsn: str, statement: object, parameters: object = None) -> None:
    connection = _connect(dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(statement, parameters)
    finally:
        connection.close()


def _bootstrap_target(
    target_dsn: str,
    suffix: str,
    *,
    roles: PostgresBootstrapRoles | None = None,
) -> tuple[PostgresBootstrapContext, str]:
    return _bootstrap_historical_head6_target(
        target_dsn,
        suffix,
        role_prefix="im",
        roles=roles,
    )


def _prepare_source(source_dsn: str) -> None:
    connection = _connect(source_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(load_migrations()[0].sql)
            cursor.execute(
                "INSERT INTO np.account_balances "
                "(id, asset, balance, last_updated) VALUES "
                "(2, 'USDT|cash', '-0'::real, '1999-12-31 23:59:59.123456'), "
                "(19, E'µBTC\\\\cold', '3.402823e+38'::real, "
                "'2038-01-19 03:14:07.999999')"
            )
            cursor.execute(
                "INSERT INTO np.liquidations "
                "(id, timestamp, symbol, entry_price, liquidation_price, "
                "quantity, leverage, liquidation_fee) VALUES "
                "(3, '2001-02-03 04:05:06.000007', E'ETH|\\\\USD', "
                "'1e-30'::real, '2e-30'::real, '3e-30'::real, 125, '-0'::real)"
            )
            cursor.execute(
                "INSERT INTO np.margin_history "
                "(id, timestamp, balance, used_margin, open_positions) VALUES "
                "(5, '2026-08-13 10:11:14.654321', '-0'::real, "
                "'1e-30'::real, 0)"
            )
            cursor.execute(
                "INSERT INTO np.model_predictions "
                "(id, created_at, symbol, side, model, vote, scored) "
                "SELECT (value * 2 + 1)::integer, "
                "'2020-01-02 03:04:05'::timestamp + value * interval '1 microsecond', "
                "'SYM|' || value::text, CASE WHEN value % 2 = 0 THEN 'BUY' ELSE 'SELL' END, "
                "E'model\\\\' || value::text, "
                "CASE WHEN value % 3 = 0 THEN 'HOLD' ELSE 'BUY' END, value % 2 = 0 "
                "FROM generate_series(1, 513) AS value"
            )
            cursor.execute(
                "INSERT INTO np.trades "
                "(id, timestamp, symbol, side, price, quantity, pnl, fee) VALUES "
                "(7, '2026-08-13 10:11:12.111111', 'BTC|USDT', 'BUY', "
                "'1e-30'::real, '2e-30'::real, '-0'::real, '-0'::real), "
                "(19, '2026-08-13 10:11:13.222222', E'ETH\\\\USDT', 'TEST', "
                "'3.402823e+38'::real, '1e-30'::real, '-3.402823e+38'::real, "
                "'1e-30'::real)"
            )
            cursor.execute(
                "INSERT INTO np.trading_session_resets "
                "(id, reset_timestamp, reason) VALUES "
                "(17, '2026-08-13 10:11:15.333333', E'cutover|\\\\fixture µ')"
            )
            sequence_states = {
                "account_balances_id_seq": (100, True),
                "liquidations_id_seq": (2, False),
                "margin_history_id_seq": (5, True),
                "model_predictions_id_seq": (2, False),
                "open_positions_id_seq": (1, False),
                "trades_id_seq": (999, True),
                "trading_session_resets_id_seq": (1, False),
            }
            for sequence, (value, is_called) in sequence_states.items():
                cursor.execute(
                    "SELECT pg_catalog.setval(%s::regclass, %s, %s)",
                    (f"np.{sequence}", value, is_called),
                )
        connection.commit()
    finally:
        connection.close()


def _relation_rows(dsn: str) -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
    connection = _connect(dsn)
    try:
        rows = []
        with connection.cursor() as cursor:
            for relation in _LEGACY_RELATIONS:
                cursor.execute(
                    sql.SQL("SELECT * FROM {} ORDER BY id").format(
                        sql.Identifier(*relation.split("."))
                    )
                )
                rows.append((relation, tuple(cursor.fetchall())))
        connection.rollback()
        return tuple(rows)
    finally:
        connection.close()


def _sequence_states(dsn: str) -> tuple[tuple[str, int, bool, int], ...]:
    connection = _connect(dsn)
    try:
        states = []
        with connection.cursor() as cursor:
            for sequence in _LEGACY_SEQUENCES:
                cursor.execute(
                    sql.SQL("SELECT last_value, is_called FROM np.{}").format(
                        sql.Identifier(sequence)
                    )
                )
                last_value, is_called = cursor.fetchone()
                next_value = last_value + 1 if is_called else last_value
                states.append((sequence, last_value, is_called, next_value))
        connection.rollback()
        return tuple(states)
    finally:
        connection.close()


def _target_boundary(dsn: str) -> tuple[object, ...]:
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT version, name, checksum FROM np.schema_migrations ORDER BY version"
            )
            migrations = tuple(cursor.fetchall())
            cursor.execute(
                "SELECT mode, runtime_generation FROM np.paper_runtime_control "
                "ORDER BY control_key"
            )
            control = tuple(cursor.fetchall())
            cursor.execute(
                "SELECT c.relname, t.tgname, t.tgenabled "
                "FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'np' AND NOT t.tgisinternal "
                "ORDER BY c.relname, t.tgname"
            )
            triggers = tuple(cursor.fetchall())
            cursor.execute(
                "SELECT relation_name, row_count FROM ("
                "SELECT 'orders'::text relation_name, count(*) row_count FROM np.orders "
                "UNION ALL SELECT 'order_events', count(*) FROM np.order_events "
                "UNION ALL SELECT 'position_streams', count(*) FROM np.position_streams "
                "UNION ALL SELECT 'paper_account_streams', count(*) FROM np.paper_account_streams "
                "UNION ALL SELECT 'paper_runtime_generations', count(*) FROM np.paper_runtime_generations"
                ") rows ORDER BY relation_name"
            )
            v2_rows = tuple(cursor.fetchall())
        connection.rollback()
        return migrations, control, triggers, v2_rows
    finally:
        connection.close()


def _preflight_payload(receipt) -> dict[str, object]:
    return {
        "status": receipt.status.value,
        "blockers": [item.value for item in receipt.blockers],
        "stale_on_return": receipt.stale_on_return,
        "snapshot_authoritative": receipt.snapshot_authoritative,
        "source": {
            "system_identifier": str(receipt.source.system_identifier),
            "relations": [
                {
                    "name": relation.name,
                    "row_count": relation.row_count,
                    "pk_min": relation.pk_min,
                    "pk_max": relation.pk_max,
                    "sha256": relation.sha256,
                }
                for relation in receipt.source.relations
            ],
            "other_session_count": receipt.source.other_session_count,
            "open_position_count": receipt.source.open_position_count,
            "semantic_invalid_row_count": receipt.source.semantic_invalid_row_count,
            "canonical_sha256": receipt.source.canonical_sha256,
            "legacy_layout_exact": receipt.source.legacy_layout_exact,
            "identity_exact": receipt.source.identity_exact,
        },
        "target": {
            "system_identifier": str(receipt.target.system_identifier),
            "terminal_catalog_exact": receipt.target.terminal_catalog_exact,
            "migration_versions": list(receipt.target.migration_versions),
            "runtime_mode": receipt.target.runtime_mode,
            "runtime_generation": receipt.target.runtime_generation,
            "nonempty_relations": list(receipt.target.nonempty_relations),
        },
    }


def _import_config(pair: "ImportPair", *, batch_size: int) -> dict[str, object]:
    intent = pair.context.target_bootstrap_intent
    return {
        "schema_version": 1,
        "batch_size": batch_size,
        "source": {
            "service": "source_service",
            "expected_database": pair.context.source_expected_database,
            "expected_role": pair.context.source_expected_role,
        },
        "target": {
            "admin_service": "target_admin_service",
            "migrator_service": "target_migrator_service",
            "bootstrap_context": {
                "expected_database": intent.expected_database,
                "admin_role": intent.admin_role,
                "roles": {
                    name: getattr(intent.roles, name)
                    for name in (
                        "schema_owner",
                        "migrator",
                        "legacy_runtime",
                        "atomic_runtime",
                        "activation",
                        "readiness",
                        "trainer",
                    )
                },
                "adoption": None,
            },
        },
    }


@dataclass(frozen=True)
class ImportPair:
    project: str
    source_dsn: str = field(repr=False)
    target_admin_dsn: str = field(repr=False)
    target_migrator_dsn: str = field(repr=False)
    context: FreshTargetCutoverContext

    def source_factory(self):
        return _connect(self.source_dsn)

    def target_admin_factory(self):
        return _connect(self.target_admin_dsn)

    def target_migrator_factory(self):
        return _connect(self.target_migrator_dsn)

    def preflight(self):
        receipt = PostgresCutoverPreflight(
            self.source_factory,
            self.target_admin_factory,
        ).inspect(self.context)
        assert receipt.status is FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET
        return receipt

    def importer(
        self,
        *,
        evidence: dict[str, list[SqlEventLog]] | None = None,
        source_wrapper: Callable[[object, SqlEventLog], object] = RecordingConnection,
        admin_wrapper: Callable[[object, SqlEventLog], object] = RecordingConnection,
        migrator_wrapper: Callable[[object, SqlEventLog], object] = RecordingConnection,
    ) -> PostgresLegacySnapshotImport:
        def wrap(label: str, factory, wrapper):
            def connect():
                record = SqlEventLog()
                if evidence is not None:
                    evidence.setdefault(label, []).append(record)
                return wrapper(factory(), record)

            return connect

        return PostgresLegacySnapshotImport(
            wrap("source", self.source_factory, source_wrapper),
            wrap("target_admin", self.target_admin_factory, admin_wrapper),
            wrap("target_migrator", self.target_migrator_factory, migrator_wrapper),
        )


@dataclass(frozen=True)
class BootstrappedTarget:
    project: str
    admin_dsn: str = field(repr=False)
    migrator_dsn: str = field(repr=False)
    context: PostgresBootstrapContext
    system_identifier: int


@contextmanager
def _bootstrapped_decoy_target(pair: ImportPair, tmp_path: Path):
    project = f"{pair.project}-decoy"
    network = f"{project}-internal"
    container = f"{project}-target"
    volume = f"{project}-data"
    password = secrets.token_hex(24)
    password_file = tmp_path / "decoy-postgres-password"
    _write_private(password_file, password + "\n")
    label = f"org.elvis.v2.snapshot-import-test={project}"
    intent_roles = pair.context.target_bootstrap_intent.roles
    roles = PostgresBootstrapRoles(
        **{
            name: getattr(intent_roles, name)
            for name in (
                "schema_owner",
                "migrator",
                "legacy_runtime",
                "atomic_runtime",
                "activation",
                "readiness",
                "trainer",
            )
        }
    )

    try:
        _create_isolated_network(network, label, project.rsplit("-", 1)[-1])
        _run(["docker", "volume", "create", "--label", label, volume])
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
                f"POSTGRES_DB={_TARGET_DATABASE}",
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
        _wait_for_postgres(container, _TARGET_DATABASE)
        admin_dsn = make_dsn(
            host="127.0.0.1",
            port=_published_port(container),
            dbname=_TARGET_DATABASE,
            user=_ADMIN_ROLE,
            password=password,
            connect_timeout=5,
        )
        context, migrator_dsn = _bootstrap_target(
            admin_dsn,
            project.rsplit("-", 1)[-1],
            roles=roles,
        )
        inspection = PostgresBootstrap(
            lambda: _connect(admin_dsn)
        ).inspect_historical_terminal(context)
        assert inspection.exact is True
        assert inspection.nonempty_relations == ()
        yield BootstrappedTarget(
            project=project,
            admin_dsn=admin_dsn,
            migrator_dsn=migrator_dsn,
            context=context,
            system_identifier=inspection.system_identifier,
        )
    finally:
        _run(
            ["docker", "rm", "--force", container],
            expected_exit_codes=(0, 1),
            secrets_to_redact=(password,),
        )
        _run(
            ["docker", "volume", "rm", "--force", volume],
            expected_exit_codes=(0, 1),
            secrets_to_redact=(password,),
        )
        _run(
            ["docker", "network", "rm", network],
            expected_exit_codes=(0, 1),
            secrets_to_redact=(password,),
        )
        _assert_project_absent(project)


@pytest.fixture
def import_pair(tmp_path: Path) -> ImportPair:
    project = f"elvis-v2-import-{uuid4().hex[:12]}"
    network = f"{project}-internal"
    source_container = f"{project}-source"
    target_container = f"{project}-target"
    password = secrets.token_hex(24)
    password_file = tmp_path / "postgres-password"
    _write_private(password_file, password + "\n")
    label = f"org.elvis.v2.snapshot-import-test={project}"
    containers = (
        (source_container, _SOURCE_DATABASE),
        (target_container, _TARGET_DATABASE),
    )
    volumes = (f"{project}-source-data", f"{project}-target-data")

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
        target_admin_dsn = make_dsn(
            host="127.0.0.1",
            port=_published_port(target_container),
            dbname=_TARGET_DATABASE,
            user=_ADMIN_ROLE,
            password=password,
            connect_timeout=5,
        )
        _prepare_source(source_dsn)
        suffix = project.rsplit("-", 1)[-1][:10]
        target_context, target_migrator_dsn = _bootstrap_target(
            target_admin_dsn,
            suffix,
        )
        yield ImportPair(
            project=project,
            source_dsn=source_dsn,
            target_admin_dsn=target_admin_dsn,
            target_migrator_dsn=target_migrator_dsn,
            context=FreshTargetCutoverContext(
                source_expected_database=_SOURCE_DATABASE,
                source_expected_role=_ADMIN_ROLE,
                target_bootstrap_intent=FreshTargetBootstrapIntent(
                    expected_database=target_context.expected_database,
                    admin_role=target_context.admin_role,
                    roles=FreshTargetRoleManifest(
                        **{
                            name: getattr(target_context.roles, name)
                            for name in (
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


def _all_events(records: list[SqlEventLog]) -> list[str]:
    return [event for record in records for event in record.events]


def _all_statements(records: list[SqlEventLog]) -> list[str]:
    return [statement for record in records for statement in record.statements]


def _empty_legacy_rows(dsn: str) -> bool:
    return all(not rows for _relation, rows in _relation_rows(dsn))


def _once(wrapper):
    used = False

    def apply(connection: object, record: SqlEventLog):
        nonlocal used
        if used:
            return RecordingConnection(connection, record)
        used = True
        return wrapper(connection, record)

    return apply


def _assert_no_forbidden_sql(statements: list[str]) -> None:
    for statement in statements:
        assert statement_keyword(statement) not in _FORBIDDEN_WRITE_KEYWORDS
        upper = statement.upper()
        assert "ON CONFLICT" not in upper
        assert "DISABLE TRIGGER" not in upper
        assert not re.search(
            r"\b(?:INSERT|COPY)\s+INTO\s+NP\.(?:ORDERS|ORDER_EVENTS|POSITION_STREAMS|PAPER_)",
            upper,
        )


def test_exact_import_and_replay_preserve_rows_sequences_and_authority(
    import_pair: ImportPair,
    tmp_path: Path,
    capsys,
) -> None:
    pair = import_pair
    preflight = pair.preflight()
    source_rows_before = _relation_rows(pair.source_dsn)
    source_sequences_before = _sequence_states(pair.source_dsn)
    source_catalog_before = _database_snapshot(pair.source_dsn)
    target_boundary_before = _target_boundary(pair.target_admin_dsn)
    evidence: dict[str, list[SqlEventLog]] = {}

    imported = pair.importer(evidence=evidence).import_snapshot(
        LegacySnapshotImportContext(pair.context, batch_size=512),
        preflight,
    )

    assert imported.disposition is LegacySnapshotImportDisposition.IMPORTED
    assert imported.source_system_identifier == preflight.source.system_identifier
    assert imported.target_system_identifier == preflight.target.system_identifier
    assert imported.source_canonical_sha256 == preflight.source.canonical_sha256
    assert imported.target_exact is True
    assert imported.runtime_activation_authorized is False
    assert imported.stale_on_return is True
    assert imported.snapshot_authoritative is False
    assert tuple(item.name for item in imported.relations) == _LEGACY_RELATIONS
    assert tuple(item.row_count for item in imported.relations) == tuple(
        item.row_count for item in preflight.source.relations
    )
    assert tuple(item.sha256 for item in imported.relations) == tuple(
        item.sha256 for item in preflight.source.relations
    )
    assert (
        next(
            item for item in imported.relations if item.name == "np.open_positions"
        ).row_count
        == 0
    )
    assert _relation_rows(pair.target_admin_dsn) == source_rows_before
    assert _relation_rows(pair.source_dsn) == source_rows_before
    assert _sequence_states(pair.source_dsn) == source_sequences_before
    assert _database_snapshot(pair.source_dsn) == source_catalog_before
    assert _target_boundary(pair.target_admin_dsn) == target_boundary_before

    config_path = tmp_path / "import.json"
    preflight_path = tmp_path / "preflight.json"
    config_path.write_text(
        json.dumps(_import_config(pair, batch_size=1)),
        encoding="utf-8",
    )
    preflight_path.write_text(
        json.dumps(_preflight_payload(preflight)),
        encoding="utf-8",
    )
    preflight_path.chmod(0o600)
    services = {
        "source_service": pair.source_factory,
        "target_admin_service": pair.target_admin_factory,
        "target_migrator_service": pair.target_migrator_factory,
    }
    assert (
        import_cli.main(
            [
                "--config",
                str(config_path),
                "--preflight-receipt",
                str(preflight_path),
                "--import-snapshot",
                "--confirm-stopped-source-clone",
                "--confirm-exclusive-database-window",
                "--confirm-disposable-target",
            ],
            service_connection_factory=services.__getitem__,
        )
        == 0
    )
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["status"] == "REPLAYED"
    assert cli_payload["stale_on_return"] is True
    assert cli_payload["snapshot_authoritative"] is False
    assert pair.target_migrator_dsn not in json.dumps(cli_payload)
    assert tuple(item.target_sequence_next for item in imported.relations) == tuple(
        state[3] for state in _sequence_states(pair.target_admin_dsn)
    )

    source_records = evidence["source"]
    migrator_records = evidence["target_migrator"]
    source_events = _all_events(source_records)
    migrator_events = _all_events(migrator_records)
    migrator_statements = _all_statements(migrator_records)
    assert (
        source_events[0]
        .upper()
        .startswith("SQL:SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
    )
    assert all(record.commits == 0 for record in source_records)
    assert sum(record.rollbacks for record in source_records) >= 1
    for relation in _LEGACY_RELATIONS:
        relation_selects = [
            statement
            for statement in _all_statements(source_records)
            if statement.upper().startswith("SELECT * FROM")
            and relation.split(".")[1].upper() in statement.upper()
        ]
        assert len(relation_selects) >= 2
    assert 512 in [size for record in source_records for size in record.fetch_sizes]
    target_batch_sizes = [
        size for record in migrator_records for size in record.parameter_batch_sizes
    ]
    assert 512 in target_batch_sizes
    assert all(1 <= size <= 512 for size in target_batch_sizes)
    first_insert = first_event_index(migrator_events, "INSERT INTO")
    assert first_event_index(migrator_events, "current_database") < first_insert
    assert first_event_index(migrator_events, "SET LOCAL ROLE") < first_insert
    assert first_event_index(migrator_events, "LOCK TABLE") < first_insert
    assert first_event_index(migrator_events, "paper_runtime_control") < first_insert
    first_setval = first_event_index(migrator_events, "setval")
    assert first_event_index(migrator_events, "commit:confirmed") < first_setval
    _assert_no_forbidden_sql(migrator_statements)

    replay_evidence: dict[str, list[SqlEventLog]] = {}
    replayed = pair.importer(evidence=replay_evidence).import_snapshot(
        LegacySnapshotImportContext(pair.context, batch_size=1),
        preflight,
    )
    assert replayed.disposition is LegacySnapshotImportDisposition.REPLAYED
    assert replayed.relations == imported.relations
    assert not any(
        statement_keyword(statement) == "INSERT"
        for statement in _all_statements(replay_evidence["target_migrator"])
    )
    assert 1 in [
        size for record in replay_evidence["source"] for size in record.fetch_sizes
    ]
    assert _relation_rows(pair.source_dsn) == source_rows_before
    assert _relation_rows(pair.target_admin_dsn) == source_rows_before
    assert _sequence_states(pair.source_dsn) == source_sequences_before
    assert _database_snapshot(pair.source_dsn) == source_catalog_before
    assert _target_boundary(pair.target_admin_dsn) == target_boundary_before


def test_source_and_target_drift_are_rejected_before_target_mutation(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    preflight = pair.preflight()
    source_before = (_relation_rows(pair.source_dsn), _sequence_states(pair.source_dsn))
    source_catalog_before = _database_snapshot(pair.source_dsn)
    target_rows_before = _relation_rows(pair.target_admin_dsn)
    target_sequences_before = _sequence_states(pair.target_admin_dsn)
    target_boundary_before = _target_boundary(pair.target_admin_dsn)

    identity_evidence: dict[str, list[SqlEventLog]] = {}
    with pytest.raises(PostgresLegacySnapshotImportConflict):
        pair.importer(evidence=identity_evidence).import_snapshot(
            LegacySnapshotImportContext(
                replace(pair.context, source_expected_database="wrong_source_clone")
            ),
            preflight,
        )
    assert not identity_evidence.get("target_migrator")

    def assert_refused(
        expected_exception: type[Exception] | tuple[type[Exception], ...],
        *,
        importer: PostgresLegacySnapshotImport | None = None,
        evidence: dict[str, list[SqlEventLog]] | None = None,
    ) -> None:
        records = evidence if evidence is not None else {}
        candidate = importer or pair.importer(evidence=records)
        target_at_attempt = (
            _relation_rows(pair.target_admin_dsn),
            _sequence_states(pair.target_admin_dsn),
            _target_boundary(pair.target_admin_dsn),
        )
        with pytest.raises(expected_exception):
            candidate.import_snapshot(
                LegacySnapshotImportContext(pair.context),
                preflight,
            )
        assert (
            _relation_rows(pair.target_admin_dsn),
            _sequence_states(pair.target_admin_dsn),
            _target_boundary(pair.target_admin_dsn),
        ) == target_at_attempt
        for target_kind in ("target_admin", "target_migrator"):
            statements = _all_statements(records.get(target_kind, []))
            assert not any(statement_keyword(value) == "INSERT" for value in statements)

    _execute(
        pair.source_dsn,
        "INSERT INTO np.trades "
        "(id, timestamp, symbol, side, price, quantity, pnl, fee) "
        "VALUES (2001, '2026-08-13 12:00:00', 'STALE', 'BUY', 1, 1, 0, 0)",
    )
    try:
        stale_evidence: dict[str, list[SqlEventLog]] = {}
        assert_refused(
            PostgresLegacySnapshotImportConflict,
            evidence=stale_evidence,
        )
        assert not stale_evidence.get("target_migrator")
    finally:
        _execute(pair.source_dsn, "DELETE FROM np.trades WHERE id = 2001")

    held_source = pair.source_factory()
    try:
        with held_source.cursor() as cursor:
            cursor.execute("SELECT 1")
        assert_refused(PostgresLegacySnapshotImportBusyError)
    finally:
        held_source.rollback()
        held_source.close()

    _execute(
        pair.source_dsn,
        "INSERT INTO np.open_positions "
        "(id, symbol, side, entry_price, quantity, leverage, entry_time) "
        "VALUES (29, 'BTCUSDT', 'BUY', 1, 1, 1, '2026-08-13 12:01:00')",
    )
    try:
        assert_refused(PostgresLegacySnapshotImportConflict)
    finally:
        _execute(pair.source_dsn, "DELETE FROM np.open_positions WHERE id = 29")

    _execute(
        pair.source_dsn,
        "INSERT INTO np.trades "
        "(id, timestamp, symbol, side, price, quantity, pnl, fee) "
        "VALUES (2003, '2026-08-13 12:02:00', ' invalid ', 'BUY', 1, 1, 0, 0)",
    )
    try:
        assert_refused(PostgresLegacySnapshotImportConflict)
    finally:
        _execute(pair.source_dsn, "DELETE FROM np.trades WHERE id = 2003")

    _execute(
        pair.source_dsn,
        "ALTER TABLE np.trades ADD COLUMN unexpected TEXT",
    )
    try:
        assert_refused(PostgresLegacySnapshotImportConflict)
    finally:
        _execute(pair.source_dsn, "ALTER TABLE np.trades DROP COLUMN unexpected")

    _execute(
        pair.target_admin_dsn,
        "DELETE FROM np.schema_migrations WHERE version = 6",
    )
    try:
        assert_refused(PostgresLegacySnapshotImportConflict)
    finally:
        migration = load_migrations()[5]
        _execute(
            pair.target_admin_dsn,
            "INSERT INTO np.schema_migrations (version, name, checksum) "
            "VALUES (%s, %s, %s)",
            (migration.version, migration.name, migration.checksum),
        )

    _execute(
        pair.target_admin_dsn,
        "UPDATE np.paper_runtime_control SET mode = 'PAUSED' WHERE control_key IS TRUE",
    )
    try:
        assert_refused(PostgresLegacySnapshotImportConflict)
    finally:
        _execute(
            pair.target_admin_dsn,
            "UPDATE np.paper_runtime_control SET mode = 'LEGACY' "
            "WHERE control_key IS TRUE",
        )

    _execute(
        pair.target_admin_dsn,
        "INSERT INTO np.trades "
        "(id, timestamp, symbol, side, price, quantity, pnl, fee) "
        "VALUES (7, '2026-08-13 12:03:00', 'PARTIAL', 'BUY', 1, 1, 0, 0)",
    )
    try:
        assert_refused(PostgresLegacySnapshotImportConflict)
    finally:
        _execute(pair.target_admin_dsn, "DELETE FROM np.trades WHERE id = 7")

    _execute(
        pair.target_admin_dsn,
        "INSERT INTO np.position_streams "
        "(position_key, execution_scope, stream_version) "
        "VALUES ('foreign-position', 'foreign-scope', 0)",
    )
    try:
        assert_refused(PostgresLegacySnapshotImportConflict)
    finally:
        _execute(
            pair.target_admin_dsn,
            "DELETE FROM np.position_streams WHERE position_key = 'foreign-position'",
        )

    assert (_relation_rows(pair.source_dsn), _sequence_states(pair.source_dsn)) == (
        source_before
    )
    assert _database_snapshot(pair.source_dsn) == source_catalog_before
    assert _relation_rows(pair.target_admin_dsn) == target_rows_before
    assert _sequence_states(pair.target_admin_dsn) == target_sequences_before
    assert _target_boundary(pair.target_admin_dsn) == target_boundary_before


def test_cluster_and_migrator_identity_miswires_fail_before_role_or_insert(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    preflight = pair.preflight()
    target_before = (
        _relation_rows(pair.target_admin_dsn),
        _sequence_states(pair.target_admin_dsn),
        _target_boundary(pair.target_admin_dsn),
    )

    scenarios = []
    same_cluster_evidence: dict[str, list[SqlEventLog]] = {}

    def same_cluster_admin():
        return _connect(pair.source_dsn)

    def same_cluster_migrator():
        return _connect(pair.source_dsn)

    scenarios.append(
        (
            PostgresLegacySnapshotImport(
                pair.source_factory,
                same_cluster_admin,
                same_cluster_migrator,
            ),
            same_cluster_evidence,
        )
    )

    wrong_role_evidence: dict[str, list[SqlEventLog]] = {}

    def wrong_role_migrator():
        record = SqlEventLog()
        wrong_role_evidence.setdefault("target_migrator", []).append(record)
        return RecordingConnection(pair.target_admin_factory(), record)

    scenarios.append(
        (
            PostgresLegacySnapshotImport(
                pair.source_factory,
                pair.target_admin_factory,
                wrong_role_migrator,
            ),
            wrong_role_evidence,
        )
    )

    other_database_dsn = make_dsn(pair.target_migrator_dsn, dbname="postgres")
    other_database_evidence: dict[str, list[SqlEventLog]] = {}

    def other_database_migrator():
        record = SqlEventLog()
        other_database_evidence.setdefault("target_migrator", []).append(record)
        return RecordingConnection(_connect(other_database_dsn), record)

    scenarios.append(
        (
            PostgresLegacySnapshotImport(
                pair.source_factory,
                pair.target_admin_factory,
                other_database_migrator,
            ),
            other_database_evidence,
        )
    )

    for importer, evidence in scenarios:
        with pytest.raises(PostgresLegacySnapshotImportConflict):
            importer.import_snapshot(
                LegacySnapshotImportContext(pair.context),
                preflight,
            )
        statements = _all_statements(evidence.get("target_migrator", []))
        assert not any("SET LOCAL ROLE" in value.upper() for value in statements)
        assert not any(statement_keyword(value) == "INSERT" for value in statements)
        assert (
            _relation_rows(pair.target_admin_dsn),
            _sequence_states(pair.target_admin_dsn),
            _target_boundary(pair.target_admin_dsn),
        ) == target_before


def test_admin_and_migrator_on_distinct_exact_targets_are_rejected_before_insert(
    import_pair: ImportPair,
    tmp_path: Path,
) -> None:
    pair = import_pair
    preflight = pair.preflight()
    target_a_before = (
        _relation_rows(pair.target_admin_dsn),
        _sequence_states(pair.target_admin_dsn),
        _target_boundary(pair.target_admin_dsn),
        _database_snapshot(pair.target_admin_dsn),
    )

    with _bootstrapped_decoy_target(pair, tmp_path) as target_b:
        assert target_b.context.expected_database == _TARGET_DATABASE
        assert target_b.context.roles == PostgresBootstrapRoles(
            **{
                name: getattr(pair.context.target_bootstrap_intent.roles, name)
                for name in (
                    "schema_owner",
                    "migrator",
                    "legacy_runtime",
                    "atomic_runtime",
                    "activation",
                    "readiness",
                    "trainer",
                )
            }
        )
        assert target_b.system_identifier != preflight.target.system_identifier
        target_b_before = (
            _relation_rows(target_b.admin_dsn),
            _sequence_states(target_b.admin_dsn),
            _target_boundary(target_b.admin_dsn),
            _database_snapshot(target_b.admin_dsn),
        )
        evidence: dict[str, list[SqlEventLog]] = {}

        def target_b_migrator():
            record = SqlEventLog()
            evidence.setdefault("target_migrator", []).append(record)
            return RecordingConnection(_connect(target_b.migrator_dsn), record)

        importer = PostgresLegacySnapshotImport(
            pair.source_factory,
            pair.target_admin_factory,
            target_b_migrator,
        )
        with pytest.raises(PostgresLegacySnapshotImportConflict):
            importer.import_snapshot(
                LegacySnapshotImportContext(pair.context),
                preflight,
            )

        statements = _all_statements(evidence["target_migrator"])
        assert not any("SET LOCAL ROLE" in value.upper() for value in statements)
        assert not any(statement_keyword(value) == "INSERT" for value in statements)
        assert (
            _relation_rows(target_b.admin_dsn),
            _sequence_states(target_b.admin_dsn),
            _target_boundary(target_b.admin_dsn),
            _database_snapshot(target_b.admin_dsn),
        ) == target_b_before
        assert (
            _relation_rows(pair.target_admin_dsn),
            _sequence_states(pair.target_admin_dsn),
            _target_boundary(pair.target_admin_dsn),
            _database_snapshot(pair.target_admin_dsn),
        ) == target_a_before


def test_terminal_drift_after_admin_inspection_is_rejected_before_insert(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    preflight = pair.preflight()
    context = LegacySnapshotImportContext(pair.context)
    rows_before = _relation_rows(pair.target_admin_dsn)
    sequences_before = _sequence_states(pair.target_admin_dsn)
    roles = pair.context.target_bootstrap_intent.roles
    migration = load_migrations()[5]

    def drift(kind: str) -> None:
        if kind == "migration_checksum":
            _execute(
                pair.target_admin_dsn,
                "UPDATE np.schema_migrations SET checksum = %s WHERE version = 6",
                ("0" * 64,),
            )
        elif kind == "table_acl":
            _execute(
                pair.target_admin_dsn,
                sql.SQL("GRANT INSERT ON TABLE np.trades TO {}").format(
                    sql.Identifier(roles.readiness)
                ),
            )
        else:
            assert kind == "role_attribute"
            _execute(
                pair.target_admin_dsn,
                sql.SQL("ALTER ROLE {} CREATEDB").format(
                    sql.Identifier(roles.readiness)
                ),
            )

    def restore(kind: str) -> None:
        if kind == "migration_checksum":
            _execute(
                pair.target_admin_dsn,
                "UPDATE np.schema_migrations SET checksum = %s WHERE version = 6",
                (migration.checksum,),
            )
        elif kind == "table_acl":
            _execute(
                pair.target_admin_dsn,
                sql.SQL("REVOKE INSERT ON TABLE np.trades FROM {}").format(
                    sql.Identifier(roles.readiness)
                ),
            )
        else:
            _execute(
                pair.target_admin_dsn,
                sql.SQL("ALTER ROLE {} NOCREATEDB").format(
                    sql.Identifier(roles.readiness)
                ),
            )

    for kind in ("migration_checksum", "table_acl", "role_attribute"):
        evidence: dict[str, list[SqlEventLog]] = {}
        drift_applied = False

        def target_admin_factory():
            record = SqlEventLog()
            evidence.setdefault("target_admin", []).append(record)
            return RecordingConnection(pair.target_admin_factory(), record)

        def drifting_migrator_factory():
            nonlocal drift_applied
            assert evidence.get("target_admin")
            assert "close" in _all_events(evidence["target_admin"])
            if not drift_applied:
                drift(kind)
                drift_applied = True
            record = SqlEventLog()
            evidence.setdefault("target_migrator", []).append(record)
            return RecordingConnection(pair.target_migrator_factory(), record)

        importer = PostgresLegacySnapshotImport(
            pair.source_factory,
            target_admin_factory,
            drifting_migrator_factory,
        )
        try:
            with pytest.raises(PostgresLegacySnapshotImportConflict):
                importer.import_snapshot(context, preflight)
        finally:
            if drift_applied:
                restore(kind)

        statements = _all_statements(evidence.get("target_migrator", []))
        assert not any(statement_keyword(value) == "INSERT" for value in statements)
        assert _relation_rows(pair.target_admin_dsn) == rows_before
        assert _sequence_states(pair.target_admin_dsn) == sequences_before
        assert (
            pair.preflight().status is FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET
        )


def test_sequence_transaction_rechecks_terminal_and_rows_before_setval(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    preflight = pair.preflight()
    context = LegacySnapshotImportContext(pair.context)
    source_rows = _relation_rows(pair.source_dsn)
    initial_sequences = _sequence_states(pair.target_admin_dsn)
    roles = pair.context.target_bootstrap_intent.roles
    migration = load_migrations()[5]

    def drift(kind: str) -> None:
        if kind == "migration_checksum":
            _execute(
                pair.target_admin_dsn,
                "UPDATE np.schema_migrations SET checksum = %s WHERE version = 6",
                ("0" * 64,),
            )
        elif kind == "legacy_row":
            _execute(
                pair.target_admin_dsn,
                "INSERT INTO np.trades "
                "(id, timestamp, symbol, side, price, quantity, pnl, fee) "
                "VALUES (2005, '2026-08-13 12:05:00', 'FOREIGN', 'BUY', 1, 1, 0, 0)",
            )
        else:
            assert kind == "table_acl"
            _execute(
                pair.target_admin_dsn,
                sql.SQL("GRANT INSERT ON TABLE np.trades TO {}").format(
                    sql.Identifier(roles.readiness)
                ),
            )

    def restore(kind: str) -> None:
        if kind == "migration_checksum":
            _execute(
                pair.target_admin_dsn,
                "UPDATE np.schema_migrations SET checksum = %s WHERE version = 6",
                (migration.checksum,),
            )
        elif kind == "legacy_row":
            _execute(pair.target_admin_dsn, "DELETE FROM np.trades WHERE id = 2005")
        else:
            _execute(
                pair.target_admin_dsn,
                sql.SQL("REVOKE INSERT ON TABLE np.trades FROM {}").format(
                    sql.Identifier(roles.readiness)
                ),
            )

    for kind in ("migration_checksum", "legacy_row", "table_acl"):
        evidence: dict[str, list[SqlEventLog]] = {}
        migrator_open_count = 0
        drift_applied = False

        def migrator_wrapper(connection: object, record: SqlEventLog):
            nonlocal drift_applied, migrator_open_count
            migrator_open_count += 1
            if migrator_open_count == 2:
                drift(kind)
                drift_applied = True
            return RecordingConnection(connection, record)

        try:
            with pytest.raises(PostgresLegacySnapshotImportConflict):
                pair.importer(
                    evidence=evidence,
                    migrator_wrapper=migrator_wrapper,
                ).import_snapshot(context, preflight)
        finally:
            if drift_applied:
                restore(kind)

        events = _all_events(evidence["target_migrator"])
        assert migrator_open_count == 2
        assert "commit:confirmed" in events
        assert not any("setval" in event for event in events)
        assert _sequence_states(pair.target_admin_dsn) == initial_sequences
        assert _relation_rows(pair.target_admin_dsn) == source_rows
        terminal = PostgresBootstrap(
            pair.target_admin_factory
        ).inspect_historical_terminal(
            PostgresBootstrapContext(
                expected_database=pair.context.target_bootstrap_intent.expected_database,
                admin_role=pair.context.target_bootstrap_intent.admin_role,
                roles=PostgresBootstrapRoles(
                    **{
                        name: getattr(pair.context.target_bootstrap_intent.roles, name)
                        for name in (
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
                adoption=None,
            )
        )
        assert terminal.exact is True
        assert terminal.nonempty_relations == tuple(
            sorted(relation for relation, rows in source_rows if rows)
        )


def test_exhausted_source_primary_key_is_rejected_before_target_io(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    _execute(
        pair.source_dsn,
        "INSERT INTO np.trades "
        "(id, timestamp, symbol, side, price, quantity, pnl, fee) "
        "VALUES (2147483647, '2026-08-13 12:06:00', 'INTMAX', 'BUY', 1, 1, 0, 0)",
    )
    preflight = pair.preflight()
    target_before = (
        _relation_rows(pair.target_admin_dsn),
        _sequence_states(pair.target_admin_dsn),
        _target_boundary(pair.target_admin_dsn),
        _database_snapshot(pair.target_admin_dsn),
    )
    evidence: dict[str, list[SqlEventLog]] = {}

    with pytest.raises(PostgresLegacySnapshotImportConflict):
        pair.importer(evidence=evidence).import_snapshot(
            LegacySnapshotImportContext(pair.context),
            preflight,
        )

    assert not evidence.get("target_admin")
    assert not evidence.get("target_migrator")
    assert (
        _relation_rows(pair.target_admin_dsn),
        _sequence_states(pair.target_admin_dsn),
        _target_boundary(pair.target_admin_dsn),
        _database_snapshot(pair.target_admin_dsn),
    ) == target_before


def test_update_committed_between_identity_and_lock_blocks_sequence_writes(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    preflight = pair.preflight()
    context = LegacySnapshotImportContext(pair.context)
    source_rows = _relation_rows(pair.source_dsn)
    initial_sequences = _sequence_states(pair.target_admin_dsn)
    evidence: dict[str, list[SqlEventLog]] = {}
    migrator_open_count = 0

    def commit_drift() -> None:
        _execute(
            pair.target_admin_dsn,
            "UPDATE np.trades SET symbol = 'TOCTOU' WHERE id = 7",
        )

    def migrator_wrapper(connection: object, record: SqlEventLog):
        nonlocal migrator_open_count
        migrator_open_count += 1
        if migrator_open_count == 2:
            return AfterTargetIdentityConnection(
                connection,
                record,
                callback=commit_drift,
            )
        return RecordingConnection(connection, record)

    with pytest.raises(PostgresLegacySnapshotImportConflict):
        pair.importer(
            evidence=evidence,
            migrator_wrapper=migrator_wrapper,
        ).import_snapshot(context, preflight)

    records = evidence["target_migrator"]
    events = _all_events(records)
    assert migrator_open_count == 2
    assert len(records) == 2
    assert "commit:confirmed" in records[0].events
    sequence_events = records[1].events
    identity_index = first_event_index(
        sequence_events,
        "target-identity:snapshot-acquired",
    )
    drift_index = first_event_index(
        sequence_events,
        "target-identity:external-drift-committed",
    )
    lock_index = first_event_index(sequence_events, "LOCK TABLE")
    assert identity_index < drift_index < lock_index
    assert not any("setval" in event for event in events)
    assert _sequence_states(pair.target_admin_dsn) == initial_sequences

    expected_rows = tuple(
        (
            relation,
            tuple(
                (
                    row[:2] + ("TOCTOU",) + row[3:]
                    if relation == "np.trades" and row[0] == 7
                    else row
                )
                for row in rows
            ),
        )
        for relation, rows in source_rows
    )
    assert _relation_rows(pair.target_admin_dsn) == expected_rows


def test_precommit_commit_unknown_and_sequence_interruptions_resume_exactly(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    preflight = pair.preflight()
    context = LegacySnapshotImportContext(pair.context)
    source_before = (_relation_rows(pair.source_dsn), _sequence_states(pair.source_dsn))
    initial_target_sequences = _sequence_states(pair.target_admin_dsn)

    before_commit_evidence: dict[str, list[SqlEventLog]] = {}

    def fail_second_insert(connection: object, record: SqlEventLog):
        return FailBeforeCommitConnection(
            connection,
            record,
            fail_on_insert_number=2,
        )

    with pytest.raises(PostgresLegacySnapshotImportStorageError):
        pair.importer(
            evidence=before_commit_evidence,
            migrator_wrapper=fail_second_insert,
        ).import_snapshot(context, preflight)
    assert _empty_legacy_rows(pair.target_admin_dsn)
    assert _sequence_states(pair.target_admin_dsn) == initial_target_sequences
    failed_events = _all_events(before_commit_evidence["target_migrator"])
    assert "insert:failed" in failed_events
    assert "rollback" in failed_events
    assert not any("setval" in event for event in failed_events)

    not_sent_evidence: dict[str, list[SqlEventLog]] = {}
    with pytest.raises(PostgresLegacySnapshotImportCommitUnknown):
        pair.importer(
            evidence=not_sent_evidence,
            migrator_wrapper=_once(CommitNotSentConnection),
        ).import_snapshot(context, preflight)
    assert _empty_legacy_rows(pair.target_admin_dsn)
    assert _sequence_states(pair.target_admin_dsn) == initial_target_sequences

    acknowledgement_lost_evidence: dict[str, list[SqlEventLog]] = {}
    recovered = pair.importer(
        evidence=acknowledgement_lost_evidence,
        migrator_wrapper=_once(CommitUnknownConnection),
    ).import_snapshot(context, preflight)
    assert recovered.disposition is LegacySnapshotImportDisposition.REPLAYED
    ack_events = _all_events(acknowledgement_lost_evidence["target_migrator"])
    assert "commit:unknown" in ack_events
    assert first_event_index(ack_events, "commit:unknown") < first_event_index(
        ack_events, "setval"
    )
    assert _relation_rows(pair.target_admin_dsn) == source_before[0]

    # Rewind only the test target to the admitted empty state so the distinct
    # after-row-commit sequence-interruption path can be exercised independently.
    for relation in reversed(_LEGACY_RELATIONS):
        _execute(
            pair.target_admin_dsn,
            sql.SQL("DELETE FROM {}").format(sql.Identifier(*relation.split("."))),
        )
    for sequence, last_value, is_called, _next_value in initial_target_sequences:
        _execute(
            pair.target_admin_dsn,
            "SELECT pg_catalog.setval(%s::regclass, %s, %s)",
            (f"np.{sequence}", last_value, is_called),
        )

    sequence_evidence: dict[str, list[SqlEventLog]] = {}

    def fail_third_setval(connection: object, record: SqlEventLog):
        return FailOnSequenceConnection(
            connection,
            record,
            fail_on_setval_number=3,
        )

    with pytest.raises(PostgresLegacySnapshotImportStorageError):
        pair.importer(
            evidence=sequence_evidence,
            migrator_wrapper=fail_third_setval,
        ).import_snapshot(context, preflight)
    assert _relation_rows(pair.target_admin_dsn) == source_before[0]
    sequence_events = _all_events(sequence_evidence["target_migrator"])
    assert first_event_index(sequence_events, "commit:confirmed") < first_event_index(
        sequence_events, "setval:failed"
    )

    resumed = pair.importer().import_snapshot(context, preflight)
    assert resumed.disposition is LegacySnapshotImportDisposition.REPLAYED
    assert _relation_rows(pair.target_admin_dsn) == source_before[0]
    assert (_relation_rows(pair.source_dsn), _sequence_states(pair.source_dsn)) == (
        source_before
    )
