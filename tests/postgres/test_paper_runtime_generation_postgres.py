"""PostgreSQL 15 proofs for immutable paper-runtime generation provenance."""

import psycopg2
import pytest

from trading.persistence import apply_migrations, load_migrations
from trading.persistence.paper_account_readiness import (
    _runtime_generation_catalog_is_exact,
)

SCOPE = "paper:test"
ACCOUNT = "paper-main"
OWNER_GENERATION = 7
OPENING_SHA = "a" * 64
BATCH_SHA = "b" * 64


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _insert_opening(cursor, *, account=ACCOUNT, opening_sha=OPENING_SHA):
    cursor.execute(
        """
        INSERT INTO np.paper_account_streams (
            account_key,
            execution_scope,
            owner_generation,
            collateral_asset,
            opening_version,
            opening_payload,
            opening_payload_sha256
        ) VALUES (%s, %s, %s, 'USDT', 1, '{}'::jsonb, %s)
        """,
        (account, SCOPE, OWNER_GENERATION, opening_sha),
    )


def _insert_epoch(
    cursor,
    generation=1,
    *,
    activation_id="activation-1",
    account=ACCOUNT,
    opening_sha=OPENING_SHA,
):
    cursor.execute(
        """
        INSERT INTO np.paper_runtime_generations (
            runtime_generation,
            activation_id,
            execution_scope,
            account_key,
            owner_generation,
            opening_version,
            opening_payload_sha256
        ) VALUES (%s, %s, %s, %s, %s, 1, %s)
        """,
        (
            generation,
            activation_id,
            SCOPE,
            account,
            OWNER_GENERATION,
            opening_sha,
        ),
    )


def _insert_order_ack(cursor):
    cursor.execute(
        """
        INSERT INTO np.position_streams (
            position_key, execution_scope, stream_version
        ) VALUES ('position-1', %s, 1)
        """,
        (SCOPE,),
    )
    cursor.execute(
        """
        INSERT INTO np.orders (
            client_order_id, decision_id, position_key, execution_scope,
            symbol, position_effect, instruction_version,
            instruction_payload, instruction_payload_sha256
        ) VALUES (
            'order-1', 'decision-1', 'position-1', %s,
            'BTCUSDT', 'OPEN', 1, '{}'::jsonb, %s
        )
        """,
        (SCOPE, "c" * 64),
    )
    cursor.execute(
        """
        INSERT INTO np.order_events (
            position_key, position_version, client_order_id, event_id,
            event_type, event_version, event_payload, event_payload_sha256,
            occurred_at
        ) VALUES (
            'position-1', 1, 'order-1', 'submission-1',
            'SUBMISSION_ACKNOWLEDGED', 1, '{}'::jsonb, %s,
            '2026-08-12T12:00:00Z'
        )
        """,
        ("d" * 64,),
    )


def _insert_legacy_manifest_without_external_refs(cursor):
    """Seed a shape-valid V1 row while bypassing only unrelated old FKs."""
    cursor.execute("SET LOCAL session_replication_role = replica")
    cursor.execute(
        """
        INSERT INTO np.paper_account_batch_manifests (
            account_key,
            client_order_id,
            execution_scope,
            owner_generation,
            opening_version,
            opening_payload_sha256,
            position_key,
            instruction_payload_sha256,
            submission_event_id,
            submission_event_type,
            submission_position_version,
            submission_observed_at,
            submission_event_payload_sha256,
            first_account_version,
            last_account_version,
            last_position_version,
            fill_count,
            batch_version,
            batch_payload,
            batch_payload_sha256
        ) VALUES (
            %s, 'order-1', %s, %s, 1, %s,
            'position-1', %s, 'submission-1', 'SUBMISSION_ACKNOWLEDGED',
            1, '2026-08-12T12:00:00Z', %s,
            1, 1, 2, 1, 1, '{"legacy":true}'::jsonb, %s
        )
        """,
        (
            ACCOUNT,
            SCOPE,
            OWNER_GENERATION,
            OPENING_SHA,
            "c" * 64,
            "d" * 64,
            BATCH_SHA,
        ),
    )
    cursor.execute("SET LOCAL session_replication_role = origin")


def test_generation_schema_is_exact_and_fresh_database_has_no_epoch(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            assert _runtime_generation_catalog_is_exact(cursor) is True
            cursor.execute("SELECT COUNT(*) FROM np.paper_runtime_generations")
            assert cursor.fetchone() == (0,)
            cursor.execute("""
                SELECT ordinal_position, column_name, udt_name, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'np'
                  AND table_name = 'paper_account_batch_manifests'
                  AND column_name = 'runtime_generation'
                """)
            assert cursor.fetchall() == [(22, "runtime_generation", "int8", "YES")]
    finally:
        connection.rollback()
        connection.close()


def test_version_four_upgrade_preserves_v1_payload_hash_and_null_generation(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        assert apply_migrations(connection, migrations[:4]) == (1, 2, 3, 4)
        with connection.cursor() as cursor:
            _insert_opening(cursor)
            _insert_legacy_manifest_without_external_refs(cursor)
            cursor.execute("""
                SELECT batch_version, batch_payload::text,
                       batch_payload_sha256::text
                FROM np.paper_account_batch_manifests
                """)
            before = cursor.fetchone()
        connection.commit()

        assert apply_migrations(connection, migrations) == (5, 6)
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT batch_version, batch_payload::text,
                       batch_payload_sha256::text, runtime_generation
                FROM np.paper_account_batch_manifests
                """)
            after = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) FROM np.paper_runtime_generations")
            assert cursor.fetchone() == (0,)

        assert after[:3] == before
        assert after[3] is None
    finally:
        connection.rollback()
        connection.close()


@pytest.mark.parametrize(
    ("generation", "activation_id", "sqlstate"),
    (
        (0, "activation-zero", "23514"),
        (-1, "activation-negative", "23514"),
        (1, "", "23514"),
        (1, " padded ", "23514"),
    ),
)
def test_epoch_rejects_nonpositive_generation_and_unclean_activation_id(
    migrated_postgres_dsn,
    generation,
    activation_id,
    sqlstate,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            _insert_opening(cursor)
            with pytest.raises(psycopg2.Error) as raised:
                _insert_epoch(
                    cursor,
                    generation,
                    activation_id=activation_id,
                )
        assert raised.value.pgcode == sqlstate
    finally:
        connection.rollback()
        connection.close()


def test_activation_id_is_globally_unique(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            _insert_opening(cursor)
            _insert_epoch(cursor, 1, activation_id="same-activation")
            with pytest.raises(psycopg2.Error) as raised:
                _insert_epoch(cursor, 2, activation_id="same-activation")
        assert raised.value.pgcode == "23505"
    finally:
        connection.rollback()
        connection.close()


def test_epoch_rejects_infinite_activation_timestamp(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            _insert_opening(cursor)
            cursor.execute("SAVEPOINT before_infinite_epoch")
            with pytest.raises(psycopg2.Error) as raised:
                cursor.execute(
                    """
                    INSERT INTO np.paper_runtime_generations (
                        runtime_generation,
                        activation_id,
                        execution_scope,
                        account_key,
                        owner_generation,
                        opening_version,
                        opening_payload_sha256,
                        activated_at
                    ) VALUES (1, 'activation-infinite', %s, %s, %s, 1, %s,
                              'infinity'::timestamptz)
                    """,
                    (SCOPE, ACCOUNT, OWNER_GENERATION, OPENING_SHA),
                )
            assert raised.value.pgcode == "23514"
            cursor.execute("ROLLBACK TO SAVEPOINT before_infinite_epoch")
            cursor.execute("SELECT COUNT(*) FROM np.paper_runtime_generations")
            assert cursor.fetchone() == (0,)
    finally:
        connection.rollback()
        connection.close()


@pytest.mark.parametrize(
    ("mutation", "sqlstate"),
    (
        (
            "UPDATE np.paper_runtime_generations "
            "SET activation_id = 'activation-tampered'",
            "55000",
        ),
        (
            "UPDATE np.paper_runtime_generations "
            "SET activated_at = clock_timestamp()",
            "55000",
        ),
        ("DELETE FROM np.paper_runtime_generations", "55000"),
        ("TRUNCATE np.paper_runtime_generations", "0A000"),
        ("TRUNCATE np.paper_runtime_generations CASCADE", "55000"),
    ),
)
def test_generation_registry_is_append_only_and_failed_mutations_leave_no_delta(
    migrated_postgres_dsn,
    mutation,
    sqlstate,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            _insert_opening(cursor)
            _insert_epoch(cursor)
            cursor.execute("""
                SELECT runtime_generation, activation_id, activated_at
                FROM np.paper_runtime_generations
                """)
            before = cursor.fetchall()
            cursor.execute("SAVEPOINT before_forbidden_mutation")
            with pytest.raises(psycopg2.Error) as raised:
                cursor.execute(mutation)
            assert raised.value.pgcode == sqlstate
            cursor.execute("ROLLBACK TO SAVEPOINT before_forbidden_mutation")
            cursor.execute("""
                SELECT runtime_generation, activation_id, activated_at
                FROM np.paper_runtime_generations
                """)
            assert cursor.fetchall() == before
    finally:
        connection.rollback()
        connection.close()


def test_generation_append_only_trigger_is_always_enabled(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            _insert_opening(cursor)
            _insert_epoch(cursor)
            cursor.execute("SET LOCAL session_replication_role = replica")
            cursor.execute("SAVEPOINT before_replica_delete")
            with pytest.raises(psycopg2.Error) as raised:
                cursor.execute("DELETE FROM np.paper_runtime_generations")
            assert raised.value.pgcode == "55000"
            cursor.execute("ROLLBACK TO SAVEPOINT before_replica_delete")
            cursor.execute("SELECT COUNT(*) FROM np.paper_runtime_generations")
            assert cursor.fetchone() == (1,)
    finally:
        connection.rollback()
        connection.close()


@pytest.mark.parametrize(
    ("account", "opening_sha"),
    (
        ("missing-account", OPENING_SHA),
        (ACCOUNT, "f" * 64),
    ),
)
def test_epoch_requires_exact_account_opening_provenance(
    migrated_postgres_dsn,
    account,
    opening_sha,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            _insert_opening(cursor)
            with pytest.raises(psycopg2.Error) as raised:
                _insert_epoch(cursor, account=account, opening_sha=opening_sha)
        assert raised.value.pgcode == "23503"
    finally:
        connection.rollback()
        connection.close()


def test_manifest_version_generation_check_has_no_v2_null_gap(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            _insert_opening(cursor)
            _insert_legacy_manifest_without_external_refs(cursor)

            with pytest.raises(psycopg2.Error) as raised:
                cursor.execute("""
                    UPDATE np.paper_account_batch_manifests
                    SET batch_version = 2, runtime_generation = NULL
                    """)
            assert raised.value.pgcode == "23514"
            connection.rollback()
    finally:
        connection.rollback()
        connection.close()


def test_manifest_generation_fk_binds_the_complete_opening_identity(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            _insert_opening(cursor)
            _insert_epoch(cursor)
            _insert_order_ack(cursor)
            _insert_legacy_manifest_without_external_refs(cursor)
            cursor.execute("""
                UPDATE np.paper_account_batch_manifests
                SET batch_version = 2, runtime_generation = 1
                """)
            cursor.execute("""
                SELECT batch_version, runtime_generation
                FROM np.paper_account_batch_manifests
                """)
            assert cursor.fetchone() == (2, 1)

            _insert_opening(cursor, account="paper-other")
            _insert_epoch(
                cursor,
                2,
                activation_id="activation-2",
                account="paper-other",
            )
            with pytest.raises(psycopg2.Error) as raised:
                cursor.execute("""
                    UPDATE np.paper_account_batch_manifests
                    SET runtime_generation = 2
                    """)
            assert raised.value.pgcode == "23503"
    finally:
        connection.rollback()
        connection.close()


def test_readiness_catalog_rejects_manifest_generation_check_tamper(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            assert _runtime_generation_catalog_is_exact(cursor) is True
            cursor.execute("""
                ALTER TABLE np.paper_account_batch_manifests
                DROP CONSTRAINT paper_account_batch_manifests_version_known
                """)
            assert _runtime_generation_catalog_is_exact(cursor) is False
    finally:
        connection.rollback()
        connection.close()
