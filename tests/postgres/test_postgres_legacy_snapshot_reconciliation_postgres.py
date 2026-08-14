"""PostgreSQL 15 proof for the read-only legacy snapshot reconciliation."""

from __future__ import annotations

import os
import re
import secrets
import struct
from contextlib import contextmanager
from decimal import Decimal
from typing import Callable

import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn

from tests.postgres.legacy_snapshot_import_support import (
    RecordingConnection,
    RecordingCursor,
    SqlEventLog,
    statement_keyword,
)
from tests.postgres.test_postgres_legacy_snapshot_import_postgres import (
    ImportPair,
    _bootstrapped_decoy_target,
)
from tests.postgres.test_postgres_legacy_snapshot_import_postgres import (
    import_pair as _import_pair_fixture,
)
from trading.application.legacy_snapshot_import import (
    LegacySnapshotImportContext,
    LegacySnapshotImportReceipt,
)
from trading.application.legacy_snapshot_reconciliation import (
    LegacyOpeningCandidateSource,
    LegacySnapshotReconciliationContext,
    LegacySnapshotReconciliationDisposition,
    LegacySnapshotReconciliationFindingKind,
    legacy_operator_equity_hypothesis_balances,
    legacy_snapshot_import_receipt_sha256,
)
from trading.persistence.postgres_bootstrap import PostgresBootstrap
from trading.persistence.postgres_legacy_snapshot_reconciliation import (
    PostgresLegacySnapshotReconciliation,
    PostgresLegacySnapshotReconciliationConflict,
    PostgresLegacySnapshotReconciliationStorageError,
)

# flake8: noqa: E501


_REQUIRED_ENV = "ELVIS_TEST_V2_LEGACY_SNAPSHOT_RECONCILIATION_REQUIRED"
_FORBIDDEN_SQL_KEYWORDS = frozenset(
    {
        "ALTER",
        "CALL",
        "COMMENT",
        "COPY",
        "CREATE",
        "DELETE",
        "DO",
        "DROP",
        "GRANT",
        "INSERT",
        "REASSIGN",
        "REINDEX",
        "RESET",
        "REVOKE",
        "SECURITY",
        "SETVAL",
        "TRUNCATE",
        "UPDATE",
    }
)

pytestmark = pytest.mark.skipif(
    os.getenv(_REQUIRED_ENV) != "1",
    reason=f"set {_REQUIRED_ENV}=1 to run the reconciliation review proof",
)

# Re-export the c3c3a disposable two-cluster fixture for this module.
import_pair = _import_pair_fixture


class _SessionRecordingCursor(RecordingCursor):
    def __init__(self, cursor: object, record: SqlEventLog) -> None:
        super().__init__(cursor, record)
        self._capture_session_rows = False

    def execute(self, query: object, variables: object = None) -> object:
        result = super().execute(query, variables)
        self._capture_session_rows = "FROM PG_STAT_ACTIVITY" in (
            self._record.statements[-1].upper()
        )
        return result

    def fetchall(self) -> object:
        rows = self._cursor.fetchall()
        if self._capture_session_rows:
            self._record.events.append(
                f"session-rows:{tuple(tuple(row) for row in rows)!r}"
            )
        return rows


class _SessionRecordingConnection(RecordingConnection):
    def cursor(self, *args: object, **kwargs: object) -> RecordingCursor:
        return _SessionRecordingCursor(
            self._connection.cursor(*args, **kwargs),
            self._record,
        )


def _context(
    pair: ImportPair,
    import_receipt: LegacySnapshotImportReceipt,
    *,
    starting: str = "1000",
    quantum: str = "0.01",
    collateral_asset: str = "USDT",
) -> LegacySnapshotReconciliationContext:
    return LegacySnapshotReconciliationContext(
        import_context=LegacySnapshotImportContext(pair.context, batch_size=512),
        config_document_sha256="a" * 64,
        import_receipt_sha256=legacy_snapshot_import_receipt_sha256(import_receipt),
        execution_scope="paper:compatibility",
        account_key="paper:primary",
        owner_generation=1,
        collateral_asset=collateral_asset,
        margin_quantum=Decimal(quantum),
        hypothesis_starting_collateral=Decimal(starting),
    )


def _import(pair: ImportPair):
    preflight = pair.preflight()
    receipt = pair.importer().import_snapshot(
        LegacySnapshotImportContext(pair.context, batch_size=512),
        preflight,
    )
    return receipt


def _prepare_source_case(
    pair: ImportPair,
    *,
    collateral_balance: str,
    trade_rows: tuple[tuple[object, ...], ...] = (),
    liquidation_rows: tuple[tuple[object, ...], ...] = (),
    reset_timestamp: str | None = None,
    extra_balances: tuple[tuple[str, str], ...] = (),
) -> None:
    for relation in (
        "account_balances",
        "liquidations",
        "margin_history",
        "model_predictions",
        "open_positions",
        "trades",
        "trading_session_resets",
    ):
        _execute(
            pair.source_dsn,
            sql.SQL("DELETE FROM np.{}").format(sql.Identifier(relation)),
        )
    _execute(
        pair.source_dsn,
        "INSERT INTO np.account_balances (asset, balance, last_updated) "
        "VALUES ('USDT', %s::real, '2026-08-13 10:00:00')",
        (collateral_balance,),
    )
    for asset, balance in extra_balances:
        _execute(
            pair.source_dsn,
            "INSERT INTO np.account_balances (asset, balance, last_updated) "
            "VALUES (%s, %s::real, '2026-08-13 10:00:00')",
            (asset, balance),
        )
    if reset_timestamp is not None:
        _execute(
            pair.source_dsn,
            "INSERT INTO np.trading_session_resets (reset_timestamp, reason) "
            "VALUES (%s, 'test')",
            (reset_timestamp,),
        )
    for timestamp, pnl, fee in trade_rows:
        _execute(
            pair.source_dsn,
            "INSERT INTO np.trades "
            "(timestamp, symbol, side, price, quantity, pnl, fee) "
            "VALUES (%s, 'BTCUSDT', 'BUY', 1, 1, %s::real, %s::real)",
            (timestamp, pnl, fee),
        )
    for timestamp, fee in liquidation_rows:
        _execute(
            pair.source_dsn,
            "INSERT INTO np.liquidations "
            "(timestamp, symbol, entry_price, liquidation_price, quantity, "
            "leverage, liquidation_fee) VALUES "
            "(%s, 'BTCUSDT', 1, 1, 1, 1, %s::real)",
            (timestamp, fee),
        )


def _execute(dsn: str, statement: object, parameters: object = None) -> None:
    connection = pair_connect(dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(statement, parameters)
    finally:
        connection.close()


def _fetchone(dsn: str, statement: object) -> tuple[object, ...]:
    connection = pair_connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(statement)
            row = tuple(cursor.fetchone())
        connection.rollback()
        return row
    finally:
        connection.close()


def pair_connect(dsn: str):
    # Keep the same patched-connect escape hatch used by the c3c3a harness.
    from tests.conftest import _ORIGINAL_PSYCOPG2_CONNECT

    connection = _ORIGINAL_PSYCOPG2_CONNECT(dsn)
    connection.autocommit = False
    return connection


def _authority_snapshot(dsn: str) -> tuple[object, ...]:
    """Byte-shaped logical proof over all authority rows, sequences and catalog."""

    connection = pair_connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT n.nspname, c.relname, c.relkind, c.relowner, "
                "c.relacl::text, c.reloptions::text FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'np' ORDER BY c.relkind, c.relname"
            )
            catalog = tuple(cursor.fetchall())
            cursor.execute(
                "SELECT schemaname, tablename, tableowner, tablespace, "
                "hasindexes, hasrules, hastriggers, rowsecurity "
                "FROM pg_tables WHERE schemaname = 'np' ORDER BY tablename"
            )
            table_catalog = tuple(cursor.fetchall())
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'np' AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            )
            tables = tuple(row[0] for row in cursor.fetchall())
            data = []
            for table in tables:
                cursor.execute(
                    sql.SQL(
                        "SELECT row_to_json(row)::text FROM (SELECT * FROM {} "
                        "ORDER BY ctid) row"
                    ).format(sql.Identifier("np", table))
                )
                data.append((table, tuple(row[0] for row in cursor.fetchall())))
            cursor.execute(
                "SELECT sequence_name FROM information_schema.sequences "
                "WHERE sequence_schema = 'np' ORDER BY sequence_name"
            )
            sequences = tuple(row[0] for row in cursor.fetchall())
            sequence_data = []
            for sequence in sequences:
                cursor.execute(
                    sql.SQL("SELECT last_value, is_called FROM np.{}").format(
                        sql.Identifier(sequence)
                    )
                )
                sequence_data.append((sequence, tuple(cursor.fetchone())))
        connection.rollback()
        return catalog, table_catalog, tuple(data), tuple(sequence_data)
    finally:
        connection.close()


def _all_statements(evidence: dict[str, list[SqlEventLog]]) -> tuple[str, ...]:
    return tuple(
        statement
        for records in evidence.values()
        for record in records
        for statement in record.statements
    )


def _session_events(evidence: dict[str, list[SqlEventLog]]) -> tuple[str, ...]:
    return tuple(
        event
        for records in evidence.values()
        for record in records
        for event in record.events
        if event.startswith("session-rows:")
    )


def _assert_read_only(evidence: dict[str, list[SqlEventLog]]) -> None:
    statements = _all_statements(evidence)
    assert statements
    for statement in statements:
        keyword = statement_keyword(statement)
        assert keyword not in _FORBIDDEN_SQL_KEYWORDS
        assert not re.search(r"(?i)\bSET\s+(?:LOCAL\s+)?ROLE\b", statement)
        assert not re.search(r"(?i)\bLOCK\s+TABLE\b", statement)
    for records in evidence.values():
        assert all(record.commits == 0 for record in records)


def _readiness_factory(pair: ImportPair):
    readiness = pair.context.target_bootstrap_intent.roles.readiness
    password = f"test-only-{secrets.token_hex(24)}"
    connection = pair.target_admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER ROLE {} PASSWORD %s").format(sql.Identifier(readiness)),
                (password,),
            )
    finally:
        connection.close()
    parameters = parse_dsn(pair.target_admin_dsn)
    parameters.update(user=readiness, password=password)
    dsn = make_dsn(**parameters)

    def connect():
        return pair_connect(dsn)

    return connect


def _reconciler(
    pair: ImportPair,
    evidence: dict[str, list[SqlEventLog]],
    *,
    readiness_wrapper: Callable[
        [object, SqlEventLog], object
    ] = _SessionRecordingConnection,
    admin_wrapper: Callable[
        [object, SqlEventLog], object
    ] = _SessionRecordingConnection,
) -> PostgresLegacySnapshotReconciliation:
    def wrap(label: str, factory, wrapper):
        def connect():
            record = SqlEventLog()
            evidence.setdefault(label, []).append(record)
            return wrapper(factory(), record)

        return connect

    return PostgresLegacySnapshotReconciliation(
        wrap("admin", pair.target_admin_factory, admin_wrapper),
        wrap("readiness", _readiness_factory(pair), readiness_wrapper),
    )


@contextmanager
def _active_session(dsn: str):
    connection = pair_connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        yield connection
    finally:
        connection.rollback()
        connection.close()


def test_head6_readiness_marker_and_historical_inspector_are_live_canaries(
    import_pair: ImportPair,
    monkeypatch,
) -> None:
    pair = import_pair
    _prepare_source_case(pair, collateral_balance="1000")
    receipt = _import(pair)
    readiness = pair.context.target_bootstrap_intent.roles.readiness
    expected_marker = (
        f"elvis-postgres-bootstrap:v1:"
        f"{pair.context.target_bootstrap_intent.expected_database}:readiness"
    )
    assert _fetchone(
        pair.target_admin_dsn,
        sql.SQL(
            "SELECT pg_catalog.shobj_description(oid, 'pg_authid') "
            "FROM pg_catalog.pg_roles WHERE rolname = {}"
        ).format(sql.Literal(readiness)),
    ) == (expected_marker,)

    def reject_v2_inspection(*_args, **_kwargs):
        raise AssertionError("the V2 terminal inspector must remain unreachable")

    monkeypatch.setattr(PostgresBootstrap, "inspect_terminal", reject_v2_inspection)
    result = _reconciler(pair, {}).reconcile(_context(pair, receipt), receipt)

    assert result.disposition is (
        LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
    )


def test_reset_window_decision_is_read_only_and_keeps_fee_evidence_separate(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    _prepare_source_case(
        pair,
        collateral_balance="1010",
        extra_balances=(("BNB", "0"), ("BTC", "0")),
        reset_timestamp="2026-08-13 12:00:00",
        trade_rows=(
            ("2026-08-13 11:59:59", "99", "9"),
            ("2026-08-13 12:00:00", "10", "0.75"),
        ),
        liquidation_rows=(
            ("2026-08-13 11:59:59", "8"),
            ("2026-08-13 12:00:00", "1.5"),
        ),
    )
    receipt = _import(pair)
    before = _authority_snapshot(pair.target_admin_dsn)
    evidence: dict[str, list[SqlEventLog]] = {}

    result = _reconciler(pair, evidence).reconcile(_context(pair, receipt), receipt)

    assert (
        result.disposition is LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
    ), (
        result.findings,
        _session_events(evidence),
    )
    assert {item.kind for item in result.findings} == {
        LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN
    }
    assert result.evidence.reset_timestamp == "2026-08-13T12:00:00.000000"
    assert result.evidence.hypothesis_realised_pnl == Decimal("10")
    assert result.evidence.hypothesis_trade_fees == Decimal("0.75")
    assert result.evidence.hypothesis_liquidation_fees == Decimal("1.5")
    assert (
        result.evidence.candidates[0].balances == result.evidence.candidates[1].balances
    )
    assert result.config_document_sha256 == "a" * 64
    assert result.import_receipt_sha256 == legacy_snapshot_import_receipt_sha256(
        receipt
    )
    assert result.account_opening_authorized is False
    assert result.account_provisioning_authorized is False
    assert result.runtime_activation_authorized is False
    assert result.stale_on_return is True
    assert result.snapshot_authoritative is False
    assert result.coherent_snapshot_observed is False
    assert result.source_provenance_authenticated is False
    assert result.target_observations_authenticated is False
    assert result.database_window_enforced is False
    assert _authority_snapshot(pair.target_admin_dsn) == before
    _assert_read_only(evidence)


def test_no_reset_usdt_only_diverges_from_operator_hypothesis_asset_set(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    _prepare_source_case(
        pair,
        collateral_balance="1002",
        trade_rows=(("2026-08-13 12:00:00", "1", "0.25"),),
    )
    receipt = _import(pair)
    evidence: dict[str, list[SqlEventLog]] = {}

    result = _reconciler(pair, evidence).reconcile(_context(pair, receipt), receipt)

    assert {item.kind for item in result.findings} == {
        LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN,
        LegacySnapshotReconciliationFindingKind.CANDIDATE_MISMATCH,
    }, evidence
    assert (
        result.disposition is LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
    )
    assert result.evidence.reset_timestamp is None
    assert result.evidence.hypothesis_realised_pnl == Decimal("1")
    assert tuple(candidate.source for candidate in result.evidence.candidates) == tuple(
        LegacyOpeningCandidateSource
    )
    assert LegacySnapshotReconciliationFindingKind.CANDIDATE_MISMATCH in {
        finding.kind for finding in result.findings
    }
    _assert_read_only(evidence)


@pytest.mark.parametrize(
    ("extra_balances", "expected_assets"),
    (
        ((("BTC", "0"),), ("BTC", "USDT")),
        ((("BNB", "0"), ("BTC", "2")), ("BNB", "BTC", "USDT")),
        (
            (("BNB", "0"), ("BTC", "0"), ("ETH", "0")),
            ("BNB", "BTC", "ETH", "USDT"),
        ),
    ),
)
def test_incomplete_nonzero_or_unknown_assets_require_an_operator_decision(
    import_pair: ImportPair,
    extra_balances: tuple[tuple[str, str], ...],
    expected_assets: tuple[str, ...],
) -> None:
    pair = import_pair
    _prepare_source_case(
        pair,
        collateral_balance="1000",
        extra_balances=extra_balances,
    )
    receipt = _import(pair)
    evidence: dict[str, list[SqlEventLog]] = {}
    result = _reconciler(pair, evidence).reconcile(_context(pair, receipt), receipt)
    assert (
        result.disposition is LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
    )
    assert (
        tuple(balance.asset for balance in result.evidence.candidates[0].balances)
        == expected_assets
    )
    assert result.evidence.candidates[0].opening_payload_sha256 != (
        result.evidence.candidates[1].opening_payload_sha256
    )
    assert {item.kind for item in result.findings} == {
        LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN,
        LegacySnapshotReconciliationFindingKind.CANDIDATE_MISMATCH,
    }
    _assert_read_only(evidence)


def test_equal_known_zero_assets_still_require_operator_decision(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    _prepare_source_case(
        pair,
        collateral_balance="1000",
        extra_balances=(("BNB", "0"), ("BTC", "0")),
    )
    receipt = _import(pair)
    evidence: dict[str, list[SqlEventLog]] = {}

    result = _reconciler(pair, evidence).reconcile(_context(pair, receipt), receipt)

    assert (
        result.disposition is LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
    )
    assert {item.kind for item in result.findings} == {
        LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN
    }
    assert tuple(
        balance.asset for balance in result.evidence.candidates[0].balances
    ) == ("BNB", "BTC", "USDT")
    assert result.evidence.candidates[0].balances == (
        result.evidence.candidates[1].balances
    )
    _assert_read_only(evidence)


@pytest.mark.parametrize(
    ("mutation", "finding"),
    (
        (
            "INSERT INTO np.open_positions "
            "(symbol, side, entry_price, quantity, leverage) "
            "VALUES ('BTCUSDT', 'BUY', 1, 1, 1)",
            LegacySnapshotReconciliationFindingKind.TARGET_OPEN_POSITION,
        ),
        (
            "UPDATE np.paper_runtime_control SET mode = 'SHADOW'",
            LegacySnapshotReconciliationFindingKind.TARGET_RUNTIME_CONTROL_DRIFT,
        ),
    ),
)
def test_open_position_and_mode_drift_block_without_mutation(
    import_pair: ImportPair,
    mutation: str,
    finding: LegacySnapshotReconciliationFindingKind,
) -> None:
    pair = import_pair
    _prepare_source_case(pair, collateral_balance="1000")
    receipt = _import(pair)
    _execute(pair.target_admin_dsn, mutation)
    before = _authority_snapshot(pair.target_admin_dsn)
    evidence: dict[str, list[SqlEventLog]] = {}
    result = _reconciler(pair, evidence).reconcile(_context(pair, receipt), receipt)
    assert result.disposition is LegacySnapshotReconciliationDisposition.BLOCKED
    assert finding in {item.kind for item in result.findings}
    assert _authority_snapshot(pair.target_admin_dsn) == before
    _assert_read_only(evidence)


def test_concurrent_session_blocks_and_storage_error_is_redacted(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    _prepare_source_case(pair, collateral_balance="1000")
    receipt = _import(pair)
    evidence: dict[str, list[SqlEventLog]] = {}
    with _active_session(pair.target_admin_dsn):
        result = _reconciler(pair, evidence).reconcile(_context(pair, receipt), receipt)
    assert result.disposition is LegacySnapshotReconciliationDisposition.BLOCKED
    assert LegacySnapshotReconciliationFindingKind.TARGET_ACTIVE_SESSIONS in {
        item.kind for item in result.findings
    }
    _assert_read_only(evidence)

    secret = "postgresql://private:secret@example.invalid/elvis"

    def explode():
        raise RuntimeError(secret)

    def explode_readiness():
        raise RuntimeError(secret)

    with pytest.raises(PostgresLegacySnapshotReconciliationStorageError) as exc:
        PostgresLegacySnapshotReconciliation(explode, explode_readiness).reconcile(
            _context(pair, receipt), receipt
        )
    assert secret not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


@pytest.mark.parametrize(
    ("mutation", "finding"),
    (
        (
            "UPDATE np.account_balances SET balance = balance + 1 "
            "WHERE asset = 'USDT'",
            LegacySnapshotReconciliationFindingKind.TARGET_LEGACY_ROWS_DRIFT,
        ),
        (
            "SELECT pg_catalog.setval('np.account_balances_id_seq'::regclass, "
            "2000000, true)",
            LegacySnapshotReconciliationFindingKind.TARGET_SEQUENCE_DRIFT,
        ),
        (
            "CREATE TABLE np.unexpected_reconciliation_drift (id integer)",
            LegacySnapshotReconciliationFindingKind.TARGET_CATALOG_DRIFT,
        ),
        (
            "INSERT INTO np.position_streams "
            "(position_key, execution_scope) VALUES ('unexpected', 'paper:test')",
            LegacySnapshotReconciliationFindingKind.TARGET_V2_STATE_PRESENT,
        ),
    ),
)
def test_row_hash_sequence_catalog_and_v2_drift_block_without_repair(
    import_pair: ImportPair,
    mutation: str,
    finding: LegacySnapshotReconciliationFindingKind,
) -> None:
    pair = import_pair
    _prepare_source_case(pair, collateral_balance="1000")
    receipt = _import(pair)
    _execute(pair.target_admin_dsn, mutation)
    before = _authority_snapshot(pair.target_admin_dsn)
    evidence: dict[str, list[SqlEventLog]] = {}

    result = _reconciler(pair, evidence).reconcile(_context(pair, receipt), receipt)

    assert result.disposition is LegacySnapshotReconciliationDisposition.BLOCKED
    assert finding in {item.kind for item in result.findings}
    assert _authority_snapshot(pair.target_admin_dsn) == before
    _assert_read_only(evidence)


def test_real_to_decimal_quantization_is_exposed_without_silent_rounding(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    _prepare_source_case(pair, collateral_balance="0.1")
    receipt = _import(pair)
    evidence: dict[str, list[SqlEventLog]] = {}

    result = _reconciler(pair, evidence).reconcile(
        _context(pair, receipt, starting="0.1", quantum="0.01"),
        receipt,
    )

    assert result.disposition is (
        LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
    )
    assert LegacySnapshotReconciliationFindingKind.QUANTIZATION_REQUIRED in {
        item.kind for item in result.findings
    }
    imported = result.evidence.candidates[0].balances[0].available
    hypothesis = next(
        balance.available
        for balance in result.evidence.candidates[1].balances
        if balance.asset == "USDT"
    )
    exact_float4 = struct.unpack("!f", bytes.fromhex("3dcccccd"))[0]
    assert imported == Decimal.from_float(exact_float4)
    assert hypothesis == Decimal.from_float(0.1)
    assert imported != Decimal("0.1")
    assert hypothesis != Decimal("0.1")
    assert result.evidence.candidates[1].balances == (
        legacy_operator_equity_hypothesis_balances(
            _context(pair, receipt, starting="0.1", quantum="0.01"),
            Decimal("0"),
        )
    )
    _assert_read_only(evidence)


def test_ordered_binary64_hypothesis_is_not_postgres_sum_real_semantics(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    _prepare_source_case(
        pair,
        collateral_balance="1",
        extra_balances=(("BNB", "0"), ("BTC", "0")),
        trade_rows=(
            ("2026-08-13 12:00:00", "16777216", "0"),
            ("2026-08-13 12:00:01", "1", "0"),
            ("2026-08-13 12:00:02", "-16777216", "0"),
        ),
    )
    receipt = _import(pair)
    before = _authority_snapshot(pair.target_admin_dsn)
    evidence: dict[str, list[SqlEventLog]] = {}

    result = _reconciler(pair, evidence).reconcile(
        _context(pair, receipt, starting="0", quantum="1"),
        receipt,
    )

    assert _fetchone(
        pair.target_admin_dsn,
        "SELECT pg_catalog.sum(pnl) FROM np.trades",
    ) == (0.0,)
    assert result.evidence.hypothesis_realised_pnl == Decimal("1")
    assert result.evidence.candidates[1].balances == (
        legacy_operator_equity_hypothesis_balances(
            _context(pair, receipt, starting="0", quantum="1"),
            Decimal("1"),
        )
    )
    assert result.evidence.candidates[0].balances == (
        result.evidence.candidates[1].balances
    )
    assert (
        result.disposition is LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
    )
    assert {item.kind for item in result.findings} == {
        LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN
    }
    assert _authority_snapshot(pair.target_admin_dsn) == before
    _assert_read_only(evidence)


def test_overlong_imported_asset_blocks_before_opening_codec(
    import_pair: ImportPair,
) -> None:
    pair = import_pair
    _prepare_source_case(
        pair,
        collateral_balance="1000",
        extra_balances=(("A" * 65, "0"),),
    )
    receipt = _import(pair)
    before = _authority_snapshot(pair.target_admin_dsn)
    evidence: dict[str, list[SqlEventLog]] = {}

    result = _reconciler(pair, evidence).reconcile(
        _context(pair, receipt),
        receipt,
    )

    assert result.disposition is LegacySnapshotReconciliationDisposition.BLOCKED
    assert {item.kind for item in result.findings} == {
        LegacySnapshotReconciliationFindingKind.OPENING_EVIDENCE_UNREPRESENTABLE
    }
    assert all(not candidate.available for candidate in result.evidence.candidates)
    assert _authority_snapshot(pair.target_admin_dsn) == before
    _assert_read_only(evidence)


@pytest.mark.parametrize(
    ("collateral_asset", "finding"),
    (
        (
            "BTC",
            LegacySnapshotReconciliationFindingKind.HYPOTHESIS_COLLATERAL_UNSUPPORTED,
        ),
        ("USDT", LegacySnapshotReconciliationFindingKind.COLLATERAL_MISSING),
    ),
)
def test_unsupported_or_missing_collateral_blocks(
    import_pair: ImportPair,
    collateral_asset: str,
    finding: LegacySnapshotReconciliationFindingKind,
) -> None:
    pair = import_pair
    _prepare_source_case(
        pair,
        collateral_balance="1000",
        extra_balances=((("BTC", "1"),) if collateral_asset == "BTC" else ()),
    )
    if finding is LegacySnapshotReconciliationFindingKind.COLLATERAL_MISSING:
        _execute(pair.source_dsn, "DELETE FROM np.account_balances")
        _execute(
            pair.source_dsn,
            "INSERT INTO np.account_balances (asset, balance) VALUES ('BTC', 1)",
        )
    receipt = _import(pair)
    evidence: dict[str, list[SqlEventLog]] = {}

    result = _reconciler(pair, evidence).reconcile(
        _context(pair, receipt, collateral_asset=collateral_asset),
        receipt,
    )

    assert result.disposition is LegacySnapshotReconciliationDisposition.BLOCKED
    assert finding in {item.kind for item in result.findings}
    _assert_read_only(evidence)


def test_wrong_role_and_cross_cluster_target_conflict_without_mutation(
    import_pair: ImportPair,
    tmp_path,
) -> None:
    pair = import_pair
    _prepare_source_case(pair, collateral_balance="1000")
    receipt = _import(pair)
    before = _authority_snapshot(pair.target_admin_dsn)

    def wrong_admin_factory():
        return pair.target_admin_factory()

    def wrong_readiness_factory():
        return pair.target_admin_factory()

    with pytest.raises(PostgresLegacySnapshotReconciliationConflict):
        PostgresLegacySnapshotReconciliation(
            wrong_admin_factory,
            wrong_readiness_factory,
        ).reconcile(_context(pair, receipt), receipt)
    assert _authority_snapshot(pair.target_admin_dsn) == before

    with _bootstrapped_decoy_target(pair, tmp_path) as decoy:
        decoy_pair = ImportPair(
            project=decoy.project,
            source_dsn=pair.source_dsn,
            target_admin_dsn=decoy.admin_dsn,
            target_migrator_dsn=decoy.migrator_dsn,
            context=pair.context,
        )
        decoy_before = _authority_snapshot(decoy.admin_dsn)
        with pytest.raises(PostgresLegacySnapshotReconciliationConflict):
            _reconciler(decoy_pair, {}).reconcile(_context(pair, receipt), receipt)
        assert _authority_snapshot(decoy.admin_dsn) == decoy_before
