"""Unwired PostgreSQL snapshot assessment before a paper-runtime fence."""

from collections import Counter
from collections.abc import Callable

import psycopg2

from trading.application.paper_account_readiness import (
    LegacyRelationWatermark,
    MigrationIdentity,
    PaperAccountReadinessAssessment,
    PaperAccountReadinessContext,
    PaperAccountReadinessFinding,
    PaperAccountReadinessFindingKind,
)
from trading.domain.order_lifecycle import OrderLifecycleState
from trading.domain.paper_accounting import PaperAccountState
from trading.domain.positions import PositionState
from trading.persistence.migration_runner import load_migrations
from trading.persistence.order_position_journal import (
    _READ_TRANSACTION_SQL,
    JournalRepositoryError,
    PostgresOrderPositionJournal,
    _replay_stream,
)
from trading.persistence.paper_account_journal import (
    PaperAccountJournalError,
    PaperAccountReplayError,
    _replay_account_locked,
)

_ACCOUNT_KEY_MAX_LENGTH = 255
_POSITION_KEY_MAX_LENGTH = 255
_CLIENT_ORDER_ID_MAX_LENGTH = 255
_LOWER_HEX = frozenset("0123456789abcdef")
_SCHEMA_MIGRATION_RELATION = "np.schema_migrations"
_LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)
_RUNTIME_CONTROL_RELATION = "np.paper_runtime_control"
_RUNTIME_GENERATION_RELATION = "np.paper_runtime_generations"
_RUNTIME_CONTROL_FUNCTION = "enforce_legacy_paper_runtime_fence"
_RUNTIME_CONTROL_TRIGGER_PREFIX = "legacy_paper_runtime_fence_"
_RUNTIME_GENERATION_FUNCTION = "reject_paper_runtime_generation_mutation"
_RUNTIME_GENERATION_TRIGGER = "paper_runtime_generations_append_only"
_CURRENT_OPENING_PROVENANCE_FUNCTION = "require_current_paper_fresh_opening_provenance"
_CURRENT_OPENING_PROVENANCE_TRIGGER = (
    "paper_runtime_generations_require_fresh_opening_provenance"
)
_FRESH_OPENING_MUTATION_FUNCTION = "reject_paper_fresh_opening_mutation"
_ACCOUNT_OPENING_IDENTITY_FUNCTION = "protect_paper_account_opening_identity"
_RUNTIME_ACTIVATION_FENCE_FUNCTION = "acquire_paper_runtime_activation_fence"
_RUNTIME_ACTIVATION_MUTATION_FUNCTION = "activate_paper_runtime_generation"
_RUNTIME_CONTROL_MODES = frozenset({"LEGACY", "SHADOW", "PAUSED", "ACTIVE"})
_EXPECTED_RUNTIME_CONTROL_FUNCTION_SOURCE = """DECLARE
    current_mode TEXT;
    current_generation BIGINT;
BEGIN
    BEGIN
        SELECT mode, runtime_generation
        INTO STRICT current_mode, current_generation
        FROM np.paper_runtime_control
        WHERE control_key IS TRUE
        FOR SHARE;
    EXCEPTION
        WHEN NO_DATA_FOUND OR TOO_MANY_ROWS OR undefined_table THEN
            RAISE EXCEPTION USING
                ERRCODE = '55000',
                MESSAGE = 'paper runtime control is unavailable';
    END;

    IF current_mode IS NULL
       OR current_generation IS NULL
       OR current_generation < 0
       OR current_mode NOT IN ('LEGACY', 'SHADOW', 'PAUSED', 'ACTIVE') THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper runtime control is invalid';
    END IF;

    IF current_mode IN ('PAUSED', 'ACTIVE') THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = FORMAT(
                'legacy paper writes are fenced in %s mode',
                current_mode
            );
    END IF;

    RETURN NULL;
END"""
_EXPECTED_RUNTIME_GENERATION_FUNCTION_SOURCE = """BEGIN
    RAISE EXCEPTION USING
        ERRCODE = '55000',
        MESSAGE = 'paper runtime generations are append-only';
END"""
_EXPECTED_RUNTIME_ACTIVATION_FENCE_FUNCTION_SOURCE = """BEGIN
    LOCK TABLE
        ONLY np.account_balances,
        ONLY np.liquidations,
        ONLY np.margin_history,
        ONLY np.model_predictions,
        ONLY np.open_positions,
        ONLY np.order_events,
        ONLY np.orders,
        ONLY np.paper_account_balances,
        ONLY np.paper_account_batch_manifests,
        ONLY np.paper_account_postings,
        ONLY np.paper_account_settlements,
        ONLY np.paper_account_streams,
        ONLY np.paper_margin_reservations,
        ONLY np.paper_runtime_control,
        ONLY np.paper_runtime_generations,
        ONLY np.position_streams,
        ONLY np.schema_migrations,
        ONLY np.trades,
        ONLY np.trading_session_resets
    IN SHARE MODE NOWAIT;

    PERFORM 1
    FROM np.paper_runtime_control
    WHERE control_key IS TRUE
    FOR UPDATE NOWAIT;

    PERFORM 1
    FROM np.paper_account_streams
    ORDER BY account_key
    FOR UPDATE NOWAIT;

    PERFORM 1
    FROM np.position_streams
    ORDER BY position_key
    FOR UPDATE NOWAIT;
END"""
_EXPECTED_RUNTIME_ACTIVATION_MUTATION_FUNCTION_SOURCE = """DECLARE
    activated_mode TEXT;
    activated_generation BIGINT;
BEGIN
    PERFORM np.acquire_paper_runtime_activation_fence();

    IF expected_mode IS NULL
       OR expected_generation IS NULL
       OR NOT (
           (expected_mode = 'LEGACY' AND expected_generation = 0)
           OR (expected_mode = 'PAUSED' AND expected_generation > 0)
       )
       OR expected_generation >= 9223372036854775807
       OR target_generation IS NULL
       OR requested_activation_id IS NULL
       OR requested_activation_id = ''
       OR requested_activation_id <> BTRIM(requested_activation_id)
       OR LENGTH(requested_activation_id) > 255
       OR requested_execution_scope IS NULL
       OR requested_execution_scope = ''
       OR requested_execution_scope <> BTRIM(requested_execution_scope)
       OR LENGTH(requested_execution_scope) > 128
       OR requested_account_key IS NULL
       OR requested_account_key = ''
       OR requested_account_key <> BTRIM(requested_account_key)
       OR LENGTH(requested_account_key) > 255
       OR requested_owner_generation IS NULL
       OR requested_owner_generation <= 0
       OR requested_opening_payload_sha256 IS NULL
       OR requested_opening_payload_sha256 !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper runtime activation arguments are invalid';
    END IF;

    IF target_generation <> expected_generation + 1 THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper runtime activation arguments are invalid';
    END IF;

    INSERT INTO np.paper_runtime_generations (
        runtime_generation,
        activation_id,
        execution_scope,
        account_key,
        owner_generation,
        opening_version,
        opening_payload_sha256
    ) VALUES (
        target_generation,
        requested_activation_id,
        requested_execution_scope,
        requested_account_key,
        requested_owner_generation,
        1,
        requested_opening_payload_sha256
    );

    UPDATE np.paper_runtime_control AS control_row
    SET
        mode = 'ACTIVE',
        runtime_generation = target_generation,
        updated_at = clock_timestamp()
    WHERE control_row.control_key IS TRUE
      AND control_row.mode = expected_mode
      AND control_row.runtime_generation = expected_generation
    RETURNING control_row.mode, control_row.runtime_generation
    INTO activated_mode, activated_generation;

    IF NOT FOUND THEN
        RAISE EXCEPTION USING
            ERRCODE = 'PT001',
            MESSAGE = 'paper runtime activation compare-and-set failed';
    END IF;

    RETURN QUERY
    SELECT activated_mode, activated_generation;
END"""
_DURABLE_BUSINESS_RELATIONS = tuple(
    sorted(
        _LEGACY_RELATIONS
        + (
            "np.order_events",
            "np.orders",
            "np.paper_account_balances",
            "np.paper_account_batch_manifests",
            "np.paper_account_postings",
            "np.paper_account_settlements",
            "np.paper_account_streams",
            "np.paper_fresh_opening_admissions",
            "np.paper_fresh_opening_nonces",
            "np.paper_fresh_opening_provisionings",
            "np.paper_margin_reservations",
            _RUNTIME_CONTROL_RELATION,
            _RUNTIME_GENERATION_RELATION,
            "np.position_streams",
        )
    )
)
_SCHEMA_DRIFT_SQLSTATES = frozenset(
    {
        "42703",  # undefined_column
        "42704",  # undefined_object
        "42804",  # datatype_mismatch
        "42809",  # wrong_object_type
        "42883",  # undefined_function
        "42P01",  # undefined_table
    }
)
_TERMINAL_LIFECYCLE_STATES = frozenset(
    {
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.FAILED,
    }
)

_SELECT_MIGRATION_RELATION_SQL = "SELECT to_regclass(%s)"
_SELECT_APPLIED_MIGRATIONS_SQL = """
SELECT version, name, checksum
FROM np.schema_migrations
ORDER BY version
"""
_SELECT_MIGRATION_COLUMNS_SQL = """
SELECT
    ordinal_position,
    column_name,
    udt_name,
    is_nullable,
    CASE
        WHEN column_default IS NULL THEN 'none'
        WHEN LOWER(column_default) IN ('now()', 'current_timestamp') THEN 'now'
        ELSE 'other'
    END,
    character_maximum_length
FROM information_schema.columns
WHERE table_schema = 'np'
  AND table_name = 'schema_migrations'
ORDER BY ordinal_position
"""
_SELECT_MIGRATION_CONSTRAINTS_SQL = """
SELECT
    constraint_row.contype,
    constraint_row.conkey,
    constraint_row.condeferrable,
    constraint_row.condeferred,
    constraint_row.convalidated
FROM pg_constraint constraint_row
JOIN pg_class table_row
  ON table_row.oid = constraint_row.conrelid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = 'schema_migrations'
ORDER BY constraint_row.conname
"""
_SELECT_MIGRATION_RELATION_KIND_SQL = """
SELECT
    table_row.relkind,
    table_row.relpersistence,
    table_row.relhasrules,
    table_row.relhastriggers,
    table_row.relrowsecurity,
    table_row.relforcerowsecurity,
    EXISTS (
        SELECT 1
        FROM pg_inherits inheritance_row
        WHERE inheritance_row.inhrelid = table_row.oid
           OR inheritance_row.inhparent = table_row.oid
    ),
    EXISTS (
        SELECT 1
        FROM pg_policy policy_row
        WHERE policy_row.polrelid = table_row.oid
    )
FROM pg_class table_row
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = 'schema_migrations'
"""
_SELECT_DURABLE_RELATIONS_SQL = """
SELECT
    FORMAT('%%I.%%I', namespace_row.nspname, table_row.relname),
    table_row.relkind,
    table_row.relpersistence,
    table_row.relhasrules,
    table_row.relrowsecurity,
    table_row.relforcerowsecurity,
    EXISTS (
        SELECT 1
        FROM pg_trigger trigger_row
        WHERE trigger_row.tgrelid = table_row.oid
          AND NOT trigger_row.tgisinternal
    ),
    EXISTS (
        SELECT 1
        FROM pg_inherits inheritance_row
        WHERE inheritance_row.inhrelid = table_row.oid
           OR inheritance_row.inhparent = table_row.oid
    ),
    EXISTS (
        SELECT 1
        FROM pg_policy policy_row
        WHERE policy_row.polrelid = table_row.oid
    )
FROM pg_class table_row
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = ANY(%s)
ORDER BY table_row.relname
"""
_SELECT_RUNTIME_CONTROL_COLUMNS_SQL = """
SELECT
    ordinal_position,
    column_name,
    udt_name,
    is_nullable,
    CASE
        WHEN column_default IS NULL THEN 'none'
        WHEN LOWER(column_default) = 'true' THEN 'true'
        WHEN LOWER(column_default) IN ('now()', 'current_timestamp') THEN 'now'
        ELSE 'other'
    END
FROM information_schema.columns
WHERE table_schema = 'np'
  AND table_name = 'paper_runtime_control'
ORDER BY ordinal_position
"""
_SELECT_RUNTIME_CONTROL_CONSTRAINTS_SQL = """
SELECT
    constraint_row.conname,
    constraint_row.contype,
    constraint_row.conkey,
    constraint_row.condeferrable,
    constraint_row.condeferred,
    constraint_row.convalidated,
    pg_get_expr(constraint_row.conbin, constraint_row.conrelid)
FROM pg_constraint constraint_row
JOIN pg_class table_row
  ON table_row.oid = constraint_row.conrelid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = 'paper_runtime_control'
ORDER BY constraint_row.conname
"""
_SELECT_RUNTIME_CONTROL_FUNCTION_SQL = """
SELECT
    routine_row.prosecdef,
    routine_row.provolatile,
    routine_row.proleakproof,
    routine_row.proisstrict,
    routine_row.pronargs,
    routine_row.prorettype = 'trigger'::regtype,
    language_row.lanname,
    routine_row.proconfig,
    routine_row.prosrc,
    routine_row.proowner = control_row.relowner
FROM pg_proc routine_row
JOIN pg_namespace namespace_row
  ON namespace_row.oid = routine_row.pronamespace
JOIN pg_language language_row
  ON language_row.oid = routine_row.prolang
JOIN pg_class control_row
  ON control_row.relname = 'paper_runtime_control'
JOIN pg_namespace control_namespace
  ON control_namespace.oid = control_row.relnamespace
 AND control_namespace.nspname = 'np'
WHERE namespace_row.nspname = 'np'
  AND routine_row.proname = 'enforce_legacy_paper_runtime_fence'
ORDER BY routine_row.oid
"""
_SELECT_RUNTIME_CONTROL_TRIGGERS_SQL = """
SELECT
    table_row.relname,
    trigger_row.tgname,
    trigger_row.tgenabled,
    trigger_row.tgtype,
    routine_namespace.nspname,
    routine_row.proname,
    trigger_row.tgqual IS NULL,
    trigger_row.tgnargs = 0,
    trigger_row.tgattr = ''::int2vector,
    trigger_row.tgconstraint = 0,
    NOT trigger_row.tgdeferrable,
    NOT trigger_row.tginitdeferred,
    trigger_row.tgoldtable IS NULL,
    trigger_row.tgnewtable IS NULL
FROM pg_trigger trigger_row
JOIN pg_class table_row
  ON table_row.oid = trigger_row.tgrelid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
JOIN pg_proc routine_row
  ON routine_row.oid = trigger_row.tgfoid
JOIN pg_namespace routine_namespace
  ON routine_namespace.oid = routine_row.pronamespace
WHERE namespace_row.nspname = 'np'
  AND NOT trigger_row.tgisinternal
ORDER BY table_row.relname, trigger_row.tgname
"""
_SELECT_RUNTIME_CONTROL_SQL = """
SELECT control_key, mode, runtime_generation
FROM np.paper_runtime_control
"""
_SELECT_RUNTIME_GENERATION_COLUMNS_SQL = """
SELECT
    ordinal_position,
    column_name,
    udt_name,
    is_nullable,
    CASE
        WHEN column_default IS NULL THEN 'none'
        WHEN LOWER(column_default) = 'clock_timestamp()' THEN 'clock_timestamp'
        ELSE 'other'
    END,
    character_maximum_length
FROM information_schema.columns
WHERE table_schema = 'np'
  AND table_name = 'paper_runtime_generations'
ORDER BY ordinal_position
"""
_SELECT_RUNTIME_GENERATION_CONSTRAINTS_SQL = """
SELECT
    constraint_row.conname,
    constraint_row.contype,
    constraint_row.conkey,
    constraint_row.condeferrable,
    constraint_row.condeferred,
    constraint_row.convalidated,
    pg_get_expr(constraint_row.conbin, constraint_row.conrelid)
FROM pg_constraint constraint_row
JOIN pg_class table_row
  ON table_row.oid = constraint_row.conrelid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = 'paper_runtime_generations'
ORDER BY constraint_row.conname
"""
_SELECT_RUNTIME_GENERATION_FKS_SQL = """
SELECT
    constraint_row.conname,
    FORMAT('%I.%I', target_namespace.nspname, target_row.relname),
    constraint_row.confkey,
    constraint_row.confupdtype,
    constraint_row.confdeltype,
    constraint_row.confmatchtype
FROM pg_constraint constraint_row
JOIN pg_class table_row
  ON table_row.oid = constraint_row.conrelid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
JOIN pg_class target_row
  ON target_row.oid = constraint_row.confrelid
JOIN pg_namespace target_namespace
  ON target_namespace.oid = target_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = 'paper_runtime_generations'
  AND constraint_row.contype = 'f'
ORDER BY constraint_row.conname
"""
_SELECT_RUNTIME_GENERATION_FUNCTION_SQL = """
SELECT
    routine_row.prosecdef,
    routine_row.provolatile,
    routine_row.proleakproof,
    routine_row.proisstrict,
    routine_row.pronargs,
    routine_row.prorettype = 'trigger'::regtype,
    language_row.lanname,
    routine_row.proconfig,
    routine_row.prosrc,
    routine_row.proowner = generation_row.relowner
FROM pg_proc routine_row
JOIN pg_namespace namespace_row
  ON namespace_row.oid = routine_row.pronamespace
JOIN pg_language language_row
  ON language_row.oid = routine_row.prolang
JOIN pg_class generation_row
  ON generation_row.relname = 'paper_runtime_generations'
JOIN pg_namespace generation_namespace
  ON generation_namespace.oid = generation_row.relnamespace
 AND generation_namespace.nspname = 'np'
WHERE namespace_row.nspname = 'np'
  AND routine_row.proname = 'reject_paper_runtime_generation_mutation'
ORDER BY routine_row.oid
"""
_SELECT_RUNTIME_ACTIVATION_FUNCTIONS_SQL = """
SELECT
    routine_row.proname,
    pg_get_function_identity_arguments(routine_row.oid),
    pg_get_function_result(routine_row.oid),
    routine_row.prosecdef,
    routine_row.provolatile,
    routine_row.proleakproof,
    routine_row.proisstrict,
    routine_row.proretset,
    routine_row.prokind,
    routine_row.proparallel,
    language_row.lanname,
    routine_row.proconfig,
    routine_row.prosrc,
    routine_row.proowner,
    (
        SELECT COUNT(*) = 22
           AND BOOL_AND(
               has_table_privilege(
                   routine_row.proowner,
                   table_row.oid,
                   %s
               )
               OR has_table_privilege(
                   routine_row.proowner,
                   table_row.oid,
                   %s
               )
               OR has_table_privilege(
                   routine_row.proowner,
                   table_row.oid,
                   %s
               )
           )
        FROM pg_class table_row
        JOIN pg_namespace table_namespace
          ON table_namespace.oid = table_row.relnamespace
        WHERE table_namespace.nspname = 'np'
          AND table_row.relname = ANY(%s)
    )
    AND has_table_privilege(
        routine_row.proowner,
        control_row.oid,
        %s
    )
    AND has_table_privilege(
        routine_row.proowner,
        control_row.oid,
        %s
    )
    AND has_table_privilege(
        routine_row.proowner,
        generation_row.oid,
        %s
    )
    AND has_table_privilege(
        routine_row.proowner,
        account_stream_row.oid,
        %s
    )
    AND has_table_privilege(
        routine_row.proowner,
        account_stream_row.oid,
        %s
    )
    AND has_table_privilege(
        routine_row.proowner,
        position_stream_row.oid,
        %s
    )
    AND has_table_privilege(
        routine_row.proowner,
        position_stream_row.oid,
        %s
    ),
    (
        SELECT COUNT(*) = 1
           AND BOOL_AND(
               function_acl.grantor = routine_row.proowner
               AND function_acl.grantee = routine_row.proowner
               AND function_acl.privilege_type = 'EXECUTE'
               AND NOT function_acl.is_grantable
           )
        FROM aclexplode(
            COALESCE(
                routine_row.proacl,
                acldefault('f', routine_row.proowner)
            )
        ) AS function_acl
    ),
    has_schema_privilege(routine_row.proowner, namespace_row.oid, %s)
FROM pg_proc routine_row
JOIN pg_namespace namespace_row
  ON namespace_row.oid = routine_row.pronamespace
JOIN pg_language language_row
  ON language_row.oid = routine_row.prolang
JOIN pg_class control_row
  ON control_row.relname = 'paper_runtime_control'
JOIN pg_namespace control_namespace
  ON control_namespace.oid = control_row.relnamespace
 AND control_namespace.nspname = 'np'
JOIN pg_class generation_row
  ON generation_row.relname = 'paper_runtime_generations'
JOIN pg_namespace generation_namespace
  ON generation_namespace.oid = generation_row.relnamespace
 AND generation_namespace.nspname = 'np'
JOIN pg_class account_stream_row
  ON account_stream_row.relname = 'paper_account_streams'
JOIN pg_namespace account_stream_namespace
  ON account_stream_namespace.oid = account_stream_row.relnamespace
 AND account_stream_namespace.nspname = 'np'
JOIN pg_class position_stream_row
  ON position_stream_row.relname = 'position_streams'
JOIN pg_namespace position_stream_namespace
  ON position_stream_namespace.oid = position_stream_row.relnamespace
 AND position_stream_namespace.nspname = 'np'
WHERE namespace_row.nspname = 'np'
  AND routine_row.proname IN (
      'acquire_paper_runtime_activation_fence',
      'activate_paper_runtime_generation'
  )
ORDER BY routine_row.proname, routine_row.oid
"""
_SELECT_RUNTIME_GENERATION_TRIGGER_SQL = """
SELECT
    table_row.relname,
    trigger_row.tgname,
    trigger_row.tgenabled,
    trigger_row.tgtype,
    routine_namespace.nspname,
    routine_row.proname,
    trigger_row.tgqual IS NULL,
    trigger_row.tgnargs = 0,
    trigger_row.tgattr = ''::int2vector,
    trigger_row.tgconstraint = 0,
    NOT trigger_row.tgdeferrable,
    NOT trigger_row.tginitdeferred,
    trigger_row.tgoldtable IS NULL,
    trigger_row.tgnewtable IS NULL
FROM pg_trigger trigger_row
JOIN pg_class table_row
  ON table_row.oid = trigger_row.tgrelid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
JOIN pg_proc routine_row
  ON routine_row.oid = trigger_row.tgfoid
JOIN pg_namespace routine_namespace
  ON routine_namespace.oid = routine_row.pronamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = 'paper_runtime_generations'
  AND NOT trigger_row.tgisinternal
ORDER BY trigger_row.tgname
"""
_SELECT_CURRENT_OPENING_PROVENANCE_FUNCTIONS_SQL = """
SELECT
    routine_row.proname,
    pg_get_function_identity_arguments(routine_row.oid),
    pg_get_function_result(routine_row.oid),
    routine_row.prosecdef,
    routine_row.provolatile,
    routine_row.proleakproof,
    routine_row.proisstrict,
    routine_row.proretset,
    routine_row.prokind,
    routine_row.proparallel,
    language_row.lanname,
    routine_row.proconfig,
    encode(sha256(convert_to(routine_row.prosrc, 'UTF8')), 'hex'),
    routine_row.proowner = provisioning_row.relowner,
    NOT EXISTS (
        SELECT 1
        FROM aclexplode(
            COALESCE(
                routine_row.proacl,
                acldefault('f', routine_row.proowner)
            )
        ) function_acl
        WHERE function_acl.grantee = 0
          AND function_acl.privilege_type = 'EXECUTE'
    )
FROM pg_proc routine_row
JOIN pg_namespace namespace_row
  ON namespace_row.oid = routine_row.pronamespace
JOIN pg_language language_row
  ON language_row.oid = routine_row.prolang
JOIN pg_class provisioning_row
  ON provisioning_row.relname = 'paper_fresh_opening_provisionings'
JOIN pg_namespace provisioning_namespace
  ON provisioning_namespace.oid = provisioning_row.relnamespace
 AND provisioning_namespace.nspname = 'np'
WHERE namespace_row.nspname = 'np'
  AND routine_row.proname IN (
      'paper_fresh_opening_target_is_current',
      'require_current_paper_fresh_opening_provenance'
  )
ORDER BY routine_row.proname, routine_row.oid
"""
_SELECT_RUNTIME_MANIFEST_COLUMN_SQL = """
SELECT
    ordinal_position,
    column_name,
    udt_name,
    is_nullable,
    CASE WHEN column_default IS NULL THEN 'none' ELSE 'other' END
FROM information_schema.columns
WHERE table_schema = 'np'
  AND table_name = 'paper_account_batch_manifests'
  AND column_name = 'runtime_generation'
"""
_SELECT_RUNTIME_MANIFEST_CONSTRAINTS_SQL = """
SELECT
    constraint_row.conname,
    constraint_row.contype,
    constraint_row.conkey,
    constraint_row.condeferrable,
    constraint_row.condeferred,
    constraint_row.convalidated,
    pg_get_expr(constraint_row.conbin, constraint_row.conrelid)
FROM pg_constraint constraint_row
JOIN pg_class table_row
  ON table_row.oid = constraint_row.conrelid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = 'paper_account_batch_manifests'
  AND 22 = ANY(constraint_row.conkey)
ORDER BY constraint_row.conname
"""
_SELECT_RUNTIME_MANIFEST_FKS_SQL = """
SELECT
    constraint_row.conname,
    FORMAT('%I.%I', target_namespace.nspname, target_row.relname),
    constraint_row.confkey,
    constraint_row.confupdtype,
    constraint_row.confdeltype,
    constraint_row.confmatchtype
FROM pg_constraint constraint_row
JOIN pg_class table_row
  ON table_row.oid = constraint_row.conrelid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
JOIN pg_class target_row
  ON target_row.oid = constraint_row.confrelid
JOIN pg_namespace target_namespace
  ON target_namespace.oid = target_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = 'paper_account_batch_manifests'
  AND constraint_row.conname =
      'paper_account_batch_manifests_runtime_generation_fk'
"""
_SELECT_RUNTIME_GENERATIONS_SQL = """
SELECT
    runtime_generation,
    activation_id,
    execution_scope,
    account_key,
    owner_generation,
    opening_version,
    opening_payload_sha256
FROM np.paper_runtime_generations
ORDER BY runtime_generation
"""
_SELECT_RUNTIME_MANIFEST_GENERATIONS_SQL = """
SELECT
    account_key,
    client_order_id,
    execution_scope,
    owner_generation,
    opening_version,
    opening_payload_sha256,
    batch_version,
    runtime_generation
FROM np.paper_account_batch_manifests
ORDER BY account_key, client_order_id
"""
_SELECT_ACCOUNT_IDENTITIES_SQL = """
SELECT account_key, execution_scope
FROM np.paper_account_streams
ORDER BY account_key
"""
_SELECT_FRESH_OPENING_PROVENANCE_SQL = """
SELECT
    provisioning_row.control_key,
    provisioning_row.execution_scope,
    provisioning_row.account_key,
    provisioning_row.owner_generation,
    provisioning_row.opening_version,
    provisioning_row.opening_payload_sha256,
    provisioning_row.candidate_payload_sha256,
    provisioning_row.pin_authority_record_sha256,
    provisioning_row.deployment_incarnation_id,
    provisioning_row.database_incarnation_id,
    provisioning_row.migration_version,
    provisioning_row.migration_name,
    provisioning_row.migration_checksum,
    provisioning_row.terminal_catalog_sha256,
    provisioning_row.runtime_mode,
    provisioning_row.runtime_generation,
    provisioning_row.authority_transition_sequence,
    provisioning_row.writer_fence,
    provisioning_row.runtime_activation_authorized,
    provisioning_row.trading_authorized,
    provisioning_row.stale_on_return,
    provisioning_row.provisioning_receipt_payload_sha256,
    admission_row.control_key,
    admission_row.candidate_payload_sha256,
    admission_row.pin_authority_record_sha256,
    admission_row.deployment_incarnation_id,
    stream_row.execution_scope,
    stream_row.account_key,
    stream_row.owner_generation,
    stream_row.opening_version,
    stream_row.opening_payload_sha256,
    np.paper_fresh_opening_target_is_current()
FROM np.paper_fresh_opening_provisionings provisioning_row
LEFT JOIN np.paper_fresh_opening_admissions admission_row
  ON admission_row.candidate_payload_sha256 =
        provisioning_row.candidate_payload_sha256
 AND admission_row.pin_authority_record_sha256 =
        provisioning_row.pin_authority_record_sha256
 AND admission_row.deployment_incarnation_id =
        provisioning_row.deployment_incarnation_id
LEFT JOIN np.paper_account_streams stream_row
  ON stream_row.execution_scope = provisioning_row.execution_scope
 AND stream_row.account_key = provisioning_row.account_key
 AND stream_row.owner_generation = provisioning_row.owner_generation
 AND stream_row.opening_version = provisioning_row.opening_version
 AND stream_row.opening_payload_sha256 =
        provisioning_row.opening_payload_sha256
ORDER BY provisioning_row.control_key
"""
_SELECT_POSITION_IDENTITIES_SQL = """
SELECT position_key, execution_scope
FROM np.position_streams
ORDER BY position_key
"""
_SELECT_ORDER_REFERENCES_SQL = """
SELECT position_key, execution_scope, client_order_id
FROM np.orders
ORDER BY position_key, client_order_id
"""
_SELECT_MANIFEST_REFERENCES_SQL = """
SELECT account_key, execution_scope, position_key, client_order_id
FROM np.paper_account_batch_manifests
ORDER BY account_key, client_order_id
"""
_SELECT_LEGACY_WATERMARK_SQL = {
    relation: f"SELECT COUNT(*), MAX(id) FROM {relation}"
    for relation in _LEGACY_RELATIONS
}


class PaperAccountReadinessError(RuntimeError):
    """Base class for dormant pre-fence assessment failures."""


class PaperAccountReadinessInputError(PaperAccountReadinessError, ValueError):
    """Raised before I/O when the assessment context is invalid."""


class PaperAccountReadinessStorageError(PaperAccountReadinessError):
    """Raised when no complete assessment can be obtained from PostgreSQL."""


def _finding(
    kind: PaperAccountReadinessFindingKind,
    subject_kind: str,
    subject_id: str,
) -> PaperAccountReadinessFinding:
    return PaperAccountReadinessFinding(kind, subject_kind, subject_id)


def _expected_migrations() -> tuple[MigrationIdentity, ...]:
    try:
        migrations = load_migrations()
        return tuple(
            MigrationIdentity(item.version, item.name, item.checksum)
            for item in migrations
        )
    except Exception as exc:
        raise PaperAccountReadinessStorageError(
            "packaged migration evidence cannot be loaded"
        ) from exc


def _one_row(raw: object, field: str, length: int) -> tuple[object, ...]:
    if not isinstance(raw, (tuple, list)) or len(raw) != length:
        raise PaperAccountReadinessStorageError(
            f"PostgreSQL returned an invalid {field} row"
        )
    return tuple(raw)


def _stored_key(value: object, field: str, maximum: int) -> str:
    if type(value) is not str:
        raise ValueError(f"stored {field} is not text")
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"stored {field} is not canonical")
    if "\x00" in value or any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        raise ValueError(f"stored {field} is not representable")
    return value


def _stored_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"stored {field} is not a lowercase SHA-256")
    return value


def _decode_identities(
    rows: object,
    *,
    field: str,
    maximum: int,
) -> tuple[tuple[str, str], ...]:
    try:
        values = tuple(
            (
                _stored_key(_one_row(row, field, 2)[0], field, maximum),
                _stored_key(
                    _one_row(row, field, 2)[1],
                    "execution scope",
                    128,
                ),
            )
            for row in rows
        )
    except (PaperAccountReadinessStorageError, TypeError, ValueError) as exc:
        raise PaperAccountReplayError(f"stored {field} inventory is invalid") from exc
    if len({identity[0] for identity in values}) != len(values):
        raise PaperAccountReplayError(f"stored {field} inventory repeats an identity")
    return tuple(sorted(values))


def _raw_migration_drift() -> PaperAccountReadinessFinding:
    return _finding(
        PaperAccountReadinessFindingKind.MIGRATION_DRIFT,
        "migration_ledger",
        _SCHEMA_MIGRATION_RELATION,
    )


def _migration_metadata_is_exact(cursor: object) -> bool:
    try:
        cursor.execute(_SELECT_MIGRATION_RELATION_KIND_SQL)
        relation_rows = tuple(cursor.fetchall())
        if relation_rows != (("r", "p", False, False, False, False, False, False),):
            return False

        cursor.execute(_SELECT_MIGRATION_COLUMNS_SQL)
        columns = tuple(tuple(row) for row in cursor.fetchall())
        if columns != (
            (1, "version", "int4", "NO", "none", None),
            (2, "name", "text", "NO", "none", None),
            (3, "checksum", "bpchar", "NO", "none", 64),
            (4, "applied_at", "timestamptz", "NO", "now", None),
        ):
            return False

        cursor.execute(_SELECT_MIGRATION_CONSTRAINTS_SQL)
        constraints = tuple(cursor.fetchall())
        if len(constraints) != 1:
            return False
        constraint = _one_row(
            constraints[0],
            "migration ledger constraint",
            5,
        )
        return (
            constraint[0] == "p"
            and tuple(constraint[1]) == (1,)
            and constraint[2:] == (False, False, True)
        )
    except PaperAccountReadinessStorageError, TypeError, ValueError:
        return False


def _read_migration_evidence(
    cursor: object,
) -> tuple[tuple[MigrationIdentity, ...], tuple[PaperAccountReadinessFinding, ...]]:
    cursor.execute(_SELECT_MIGRATION_RELATION_SQL, (_SCHEMA_MIGRATION_RELATION,))
    relation_row = cursor.fetchone()
    relation = _one_row(relation_row, "migration relation", 1)[0]
    if relation is None:
        return (), ()
    if type(relation) is not str or relation != _SCHEMA_MIGRATION_RELATION:
        return (), (_raw_migration_drift(),)
    if not _migration_metadata_is_exact(cursor):
        return (), (_raw_migration_drift(),)

    try:
        cursor.execute(_SELECT_APPLIED_MIGRATIONS_SQL)
        raw_rows = tuple(cursor.fetchall())
    except psycopg2.Error as exc:
        if getattr(exc, "pgcode", None) in _SCHEMA_DRIFT_SQLSTATES:
            return (), (_raw_migration_drift(),)
        raise

    decoded = []
    for raw in raw_rows:
        try:
            row = _one_row(raw, "migration ledger", 3)
            identity = MigrationIdentity(row[0], row[1], row[2])
            if identity.version != len(decoded) + 1:
                raise ValueError("migration versions are not contiguous")
        except PaperAccountReadinessStorageError, TypeError, ValueError:
            return tuple(decoded), (_raw_migration_drift(),)
        decoded.append(identity)
    return tuple(decoded), ()


def _durable_business_relations_are_authoritative(cursor: object) -> bool:
    cursor.execute(
        _SELECT_DURABLE_RELATIONS_SQL,
        ([relation.removeprefix("np.") for relation in _DURABLE_BUSINESS_RELATIONS],),
    )
    rows = tuple(tuple(row) for row in cursor.fetchall())
    expected = tuple(
        (
            relation,
            "r",
            "p",
            False,
            False,
            False,
            relation in _LEGACY_RELATIONS
            or relation
            in {
                "np.paper_account_streams",
                "np.paper_fresh_opening_admissions",
                "np.paper_fresh_opening_nonces",
                "np.paper_fresh_opening_provisionings",
                _RUNTIME_GENERATION_RELATION,
            },
            False,
            False,
        )
        for relation in _DURABLE_BUSINESS_RELATIONS
    )
    return rows == expected


def _runtime_control_catalog_is_exact(cursor: object) -> bool:
    try:
        cursor.execute(_SELECT_RUNTIME_CONTROL_COLUMNS_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            (1, "control_key", "bool", "NO", "true"),
            (2, "mode", "text", "NO", "none"),
            (3, "runtime_generation", "int8", "NO", "none"),
            (4, "updated_at", "timestamptz", "NO", "now"),
        ):
            return False

        cursor.execute(_SELECT_RUNTIME_CONTROL_CONSTRAINTS_SQL)
        constraints = tuple(tuple(row) for row in cursor.fetchall())
        if constraints != (
            (
                "paper_runtime_control_generation_nonnegative",
                "c",
                [3],
                False,
                False,
                True,
                "(runtime_generation >= 0)",
            ),
            (
                "paper_runtime_control_mode",
                "c",
                [2],
                False,
                False,
                True,
                "(mode = ANY (ARRAY['LEGACY'::text, 'SHADOW'::text, "
                "'PAUSED'::text, 'ACTIVE'::text]))",
            ),
            (
                "paper_runtime_control_pkey",
                "p",
                [1],
                False,
                False,
                True,
                None,
            ),
            (
                "paper_runtime_control_singleton",
                "c",
                [1],
                False,
                False,
                True,
                "control_key",
            ),
        ):
            return False

        cursor.execute(_SELECT_RUNTIME_CONTROL_FUNCTION_SQL)
        function_rows = tuple(tuple(row) for row in cursor.fetchall())
        if len(function_rows) != 1:
            return False
        function = function_rows[0]
        if function[:8] != (
            True,
            "v",
            False,
            False,
            0,
            True,
            "plpgsql",
            ["search_path=pg_catalog, pg_temp"],
        ):
            return False
        if type(function[8]) is not str:
            return False
        if function[8].strip() != _EXPECTED_RUNTIME_CONTROL_FUNCTION_SOURCE:
            return False
        if function[9] is not True:
            return False

        cursor.execute(_SELECT_RUNTIME_CONTROL_TRIGGERS_SQL)
        trigger_rows = tuple(tuple(row) for row in cursor.fetchall())
        expected_triggers = tuple(
            sorted(
                tuple(
                    (
                        relation.removeprefix("np."),
                        _RUNTIME_CONTROL_TRIGGER_PREFIX + relation.removeprefix("np."),
                        "A",
                        62,
                        "np",
                        _RUNTIME_CONTROL_FUNCTION,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    )
                    for relation in _LEGACY_RELATIONS
                )
                + (
                    (
                        "paper_account_streams",
                        "paper_account_streams_opening_identity_immutable",
                        "A",
                        27,
                        "np",
                        _ACCOUNT_OPENING_IDENTITY_FUNCTION,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ),
                    (
                        "paper_account_streams",
                        "paper_account_streams_opening_identity_truncate",
                        "A",
                        34,
                        "np",
                        _ACCOUNT_OPENING_IDENTITY_FUNCTION,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ),
                    (
                        "paper_fresh_opening_admissions",
                        "paper_fresh_opening_admissions_append_only",
                        "A",
                        58,
                        "np",
                        _FRESH_OPENING_MUTATION_FUNCTION,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ),
                    (
                        "paper_fresh_opening_nonces",
                        "paper_fresh_opening_nonces_append_only",
                        "A",
                        58,
                        "np",
                        _FRESH_OPENING_MUTATION_FUNCTION,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ),
                    (
                        "paper_fresh_opening_provisionings",
                        "paper_fresh_opening_provisionings_append_only",
                        "A",
                        58,
                        "np",
                        _FRESH_OPENING_MUTATION_FUNCTION,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ),
                    (
                        _RUNTIME_GENERATION_RELATION.removeprefix("np."),
                        _RUNTIME_GENERATION_TRIGGER,
                        "A",
                        58,
                        "np",
                        _RUNTIME_GENERATION_FUNCTION,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ),
                    (
                        _RUNTIME_GENERATION_RELATION.removeprefix("np."),
                        _CURRENT_OPENING_PROVENANCE_TRIGGER,
                        "A",
                        7,
                        "np",
                        _CURRENT_OPENING_PROVENANCE_FUNCTION,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ),
                )
            )
        )
        return trigger_rows == expected_triggers
    except PaperAccountReadinessStorageError, TypeError, ValueError:
        return False


def _runtime_generation_catalog_is_exact(cursor: object) -> bool:
    try:
        cursor.execute(_SELECT_CURRENT_OPENING_PROVENANCE_FUNCTIONS_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            (
                "paper_fresh_opening_target_is_current",
                "",
                "boolean",
                True,
                "v",
                False,
                False,
                False,
                "f",
                "u",
                "plpgsql",
                ["search_path=pg_catalog, pg_temp"],
                "68c05eeedb12d92795adc39652e80b52055afcd76e3fd9d4fbc57d373bf2abf1",
                True,
                True,
            ),
            (
                "require_current_paper_fresh_opening_provenance",
                "",
                "trigger",
                True,
                "v",
                False,
                False,
                False,
                "f",
                "u",
                "plpgsql",
                ["search_path=pg_catalog, pg_temp"],
                "34b067d3fdaedb59b4afef3a60413f9a7b57c27bfcf9a628d865c7f37df7a747",
                True,
                True,
            ),
        ):
            return False

        cursor.execute(_SELECT_RUNTIME_GENERATION_COLUMNS_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            (1, "runtime_generation", "int8", "NO", "none", None),
            (2, "activation_id", "varchar", "NO", "none", 255),
            (3, "execution_scope", "varchar", "NO", "none", 128),
            (4, "account_key", "varchar", "NO", "none", 255),
            (5, "owner_generation", "int8", "NO", "none", None),
            (6, "opening_version", "int2", "NO", "none", None),
            (7, "opening_payload_sha256", "bpchar", "NO", "none", 64),
            (8, "activated_at", "timestamptz", "NO", "clock_timestamp", None),
        ):
            return False

        cursor.execute(_SELECT_RUNTIME_GENERATION_CONSTRAINTS_SQL)
        constraints = tuple(tuple(row) for row in cursor.fetchall())
        if constraints != (
            (
                "paper_runtime_generations_account_key_clean",
                "c",
                [4],
                False,
                False,
                True,
                "(((account_key)::text = btrim((account_key)::text)) AND "
                "((account_key)::text <> ''::text))",
            ),
            (
                "paper_runtime_generations_activated_at_finite",
                "c",
                [8],
                False,
                False,
                True,
                "isfinite(activated_at)",
            ),
            (
                "paper_runtime_generations_activation_id_clean",
                "c",
                [2],
                False,
                False,
                True,
                "(((activation_id)::text = btrim((activation_id)::text)) AND "
                "((activation_id)::text <> ''::text))",
            ),
            (
                "paper_runtime_generations_activation_id_uq",
                "u",
                [2],
                False,
                False,
                True,
                None,
            ),
            (
                "paper_runtime_generations_execution_scope_clean",
                "c",
                [3],
                False,
                False,
                True,
                "(((execution_scope)::text = btrim((execution_scope)::text)) AND "
                "((execution_scope)::text <> ''::text))",
            ),
            (
                "paper_runtime_generations_fresh_opening_provisioning_fk",
                "f",
                [3, 4, 5, 6, 7],
                False,
                False,
                True,
                None,
            ),
            (
                "paper_runtime_generations_generation_positive",
                "c",
                [1],
                False,
                False,
                True,
                "(runtime_generation > 0)",
            ),
            (
                "paper_runtime_generations_manifest_ref_uq",
                "u",
                [1, 3, 4, 5, 6, 7],
                False,
                False,
                True,
                None,
            ),
            (
                "paper_runtime_generations_opening_fk",
                "f",
                [3, 4, 5, 6, 7],
                False,
                False,
                True,
                None,
            ),
            (
                "paper_runtime_generations_opening_sha256_valid",
                "c",
                [7],
                False,
                False,
                True,
                "(opening_payload_sha256 ~ '^[0-9a-f]{64}$'::text)",
            ),
            (
                "paper_runtime_generations_opening_version_known",
                "c",
                [6],
                False,
                False,
                True,
                "(opening_version = 1)",
            ),
            (
                "paper_runtime_generations_owner_generation_positive",
                "c",
                [5],
                False,
                False,
                True,
                "(owner_generation > 0)",
            ),
            (
                "paper_runtime_generations_pkey",
                "p",
                [1],
                False,
                False,
                True,
                None,
            ),
        ):
            return False

        cursor.execute(_SELECT_RUNTIME_GENERATION_FKS_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            (
                "paper_runtime_generations_fresh_opening_provisioning_fk",
                "np.paper_fresh_opening_provisionings",
                [6, 7, 8, 10, 20],
                "a",
                "r",
                "f",
            ),
            (
                "paper_runtime_generations_opening_fk",
                "np.paper_account_streams",
                [2, 1, 3, 7, 9],
                "a",
                "r",
                "s",
            ),
        ):
            return False

        cursor.execute(_SELECT_RUNTIME_GENERATION_FUNCTION_SQL)
        function_rows = tuple(tuple(row) for row in cursor.fetchall())
        if len(function_rows) != 1:
            return False
        function = function_rows[0]
        if function[:8] != (
            True,
            "v",
            False,
            False,
            0,
            True,
            "plpgsql",
            ["search_path=pg_catalog, pg_temp"],
        ):
            return False
        if type(function[8]) is not str:
            return False
        if function[8].strip() != _EXPECTED_RUNTIME_GENERATION_FUNCTION_SOURCE:
            return False
        if function[9] is not True:
            return False

        cursor.execute(_SELECT_RUNTIME_GENERATION_TRIGGER_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            (
                "paper_runtime_generations",
                _RUNTIME_GENERATION_TRIGGER,
                "A",
                58,
                "np",
                _RUNTIME_GENERATION_FUNCTION,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ),
            (
                "paper_runtime_generations",
                _CURRENT_OPENING_PROVENANCE_TRIGGER,
                "A",
                7,
                "np",
                _CURRENT_OPENING_PROVENANCE_FUNCTION,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ),
        ):
            return False

        cursor.execute(_SELECT_RUNTIME_MANIFEST_COLUMN_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            (22, "runtime_generation", "int8", "YES", "none"),
        ):
            return False

        cursor.execute(_SELECT_RUNTIME_MANIFEST_CONSTRAINTS_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            (
                "paper_account_batch_manifests_runtime_generation_fk",
                "f",
                [22, 3, 1, 4, 5, 6],
                False,
                False,
                True,
                None,
            ),
            (
                "paper_account_batch_manifests_version_known",
                "c",
                [18, 22],
                False,
                False,
                True,
                "(((batch_version = 1) AND (runtime_generation IS NULL)) OR "
                "((batch_version = 2) AND (runtime_generation IS NOT NULL) AND "
                "(runtime_generation > 0)))",
            ),
        ):
            return False

        cursor.execute(_SELECT_RUNTIME_MANIFEST_FKS_SQL)
        return tuple(tuple(row) for row in cursor.fetchall()) == (
            (
                "paper_account_batch_manifests_runtime_generation_fk",
                "np.paper_runtime_generations",
                [1, 3, 4, 5, 6, 7],
                "a",
                "r",
                "s",
            ),
        )
    except PaperAccountReadinessStorageError, TypeError, ValueError:
        return False


def _runtime_activation_capabilities_catalog_is_exact(cursor: object) -> bool:
    try:
        cursor.execute(
            _SELECT_RUNTIME_ACTIVATION_FUNCTIONS_SQL,
            (
                "UPDATE",
                "DELETE",
                "TRUNCATE",
                [
                    relation.removeprefix("np.")
                    for relation in _DURABLE_BUSINESS_RELATIONS
                    + (_SCHEMA_MIGRATION_RELATION,)
                ],
                "SELECT",
                "UPDATE",
                "INSERT",
                "SELECT",
                "UPDATE",
                "SELECT",
                "UPDATE",
                "USAGE",
            ),
        )
        function_rows = tuple(tuple(row) for row in cursor.fetchall())
        if len(function_rows) != 2:
            return False

        fence_function, mutation_function = function_rows
        if fence_function[:12] != (
            _RUNTIME_ACTIVATION_FENCE_FUNCTION,
            "",
            "void",
            True,
            "v",
            False,
            False,
            False,
            "f",
            "u",
            "plpgsql",
            ["search_path=pg_catalog, pg_temp"],
        ):
            return False
        if type(fence_function[12]) is not str or (
            fence_function[12].strip()
            != _EXPECTED_RUNTIME_ACTIVATION_FENCE_FUNCTION_SOURCE
        ):
            return False
        if type(fence_function[13]) is not int:
            return False
        if fence_function[14:] != (True, True, True):
            return False

        if mutation_function[:12] != (
            _RUNTIME_ACTIVATION_MUTATION_FUNCTION,
            "expected_mode text, expected_generation bigint, target_generation "
            "bigint, requested_activation_id text, requested_execution_scope "
            "text, requested_account_key text, requested_owner_generation bigint, "
            "requested_opening_payload_sha256 text",
            "TABLE(mode text, runtime_generation bigint)",
            True,
            "v",
            False,
            False,
            True,
            "f",
            "u",
            "plpgsql",
            ["search_path=pg_catalog, pg_temp"],
        ):
            return False
        if type(mutation_function[12]) is not str or (
            mutation_function[12].strip()
            != _EXPECTED_RUNTIME_ACTIVATION_MUTATION_FUNCTION_SOURCE
        ):
            return False
        return (
            type(mutation_function[13]) is int
            and mutation_function[13] == fence_function[13]
            and mutation_function[14:] == (True, True, True)
        )
    except PaperAccountReadinessStorageError, TypeError, ValueError:
        return False


def _read_runtime_control(
    cursor: object,
) -> tuple[str, int] | None:
    try:
        cursor.execute(_SELECT_RUNTIME_CONTROL_SQL)
        rows = tuple(cursor.fetchall())
        if len(rows) != 1:
            return None
        row = _one_row(rows[0], "paper runtime control", 3)
        if row[0] is not True or type(row[1]) is not str:
            return None
        if row[1] not in _RUNTIME_CONTROL_MODES:
            return None
        if type(row[2]) is not int or not 0 <= row[2] <= (1 << 63) - 1:
            return None
        return row[1], row[2]
    except PaperAccountReadinessStorageError, TypeError, ValueError:
        return None


def _runtime_generation_evidence_is_exact(
    cursor: object,
    *,
    context: PaperAccountReadinessContext,
    runtime_mode: str,
    runtime_generation: int,
) -> bool:
    try:
        cursor.execute(_SELECT_RUNTIME_GENERATIONS_SQL)
        rows = tuple(cursor.fetchall())
        if runtime_mode == "LEGACY" and runtime_generation != 0:
            return False
        if runtime_mode == "ACTIVE" and runtime_generation == 0:
            return False
        if len(rows) != runtime_generation:
            return False
        activation_ids = set()
        generations = {}
        for expected_generation, raw in enumerate(rows, start=1):
            row = _one_row(raw, "paper runtime generation", 7)
            if type(row[0]) is not int or row[0] != expected_generation:
                return False
            activation_id = _stored_key(row[1], "activation ID", 255)
            if activation_id in activation_ids:
                return False
            activation_ids.add(activation_id)
            if (
                _stored_key(row[2], "execution scope", 128) != context.execution_scope
                or _stored_key(row[3], "account key", _ACCOUNT_KEY_MAX_LENGTH)
                != context.account_key
                or type(row[4]) is not int
                or row[4] != context.owner_generation
                or type(row[5]) is not int
                or row[5] != 1
                or type(row[6]) is not str
                or row[6] != context.opening_payload_sha256
            ):
                return False
            generations[expected_generation] = (
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
            )

        cursor.execute(_SELECT_RUNTIME_MANIFEST_GENERATIONS_SQL)
        manifest_rows = tuple(cursor.fetchall())
        if runtime_generation == 0:
            return not manifest_rows
        for raw in manifest_rows:
            row = _one_row(raw, "paper account batch generation", 8)
            account_key = _stored_key(
                row[0], "manifest account key", _ACCOUNT_KEY_MAX_LENGTH
            )
            _stored_key(row[1], "manifest client order ID", _CLIENT_ORDER_ID_MAX_LENGTH)
            execution_scope = _stored_key(row[2], "manifest execution scope", 128)
            owner_generation = row[3]
            opening_version = row[4]
            batch_version = row[6]
            manifest_generation = row[7]
            if (
                type(owner_generation) is not int
                or type(opening_version) is not int
                or type(batch_version) is not int
                or type(manifest_generation) is not int
                or batch_version != 2
                or not 1 <= manifest_generation <= runtime_generation
                or type(row[5]) is not str
            ):
                return False
            if generations.get(manifest_generation) != (
                execution_scope,
                account_key,
                owner_generation,
                opening_version,
                row[5],
            ):
                return False
        return True
    except PaperAccountReadinessStorageError, TypeError, ValueError:
        return False


def _decode_order_references(rows: object) -> tuple[tuple[str, str, str], ...]:
    try:
        references = tuple(
            (
                _stored_key(
                    _one_row(row, "order reference", 3)[0],
                    "position key",
                    _POSITION_KEY_MAX_LENGTH,
                ),
                _stored_key(
                    _one_row(row, "order reference", 3)[1],
                    "execution scope",
                    128,
                ),
                _stored_key(
                    _one_row(row, "order reference", 3)[2],
                    "client order ID",
                    _CLIENT_ORDER_ID_MAX_LENGTH,
                ),
            )
            for row in rows
        )
    except (PaperAccountReadinessStorageError, TypeError, ValueError) as exc:
        raise JournalRepositoryError("stored order inventory is invalid") from exc
    return tuple(sorted(references))


def _decode_manifest_references(
    rows: object,
) -> tuple[tuple[str, str, str, str], ...]:
    try:
        references = tuple(
            (
                _stored_key(
                    _one_row(row, "manifest reference", 4)[0],
                    "account key",
                    _ACCOUNT_KEY_MAX_LENGTH,
                ),
                _stored_key(
                    _one_row(row, "manifest reference", 4)[1],
                    "execution scope",
                    128,
                ),
                _stored_key(
                    _one_row(row, "manifest reference", 4)[2],
                    "position key",
                    _POSITION_KEY_MAX_LENGTH,
                ),
                _stored_key(
                    _one_row(row, "manifest reference", 4)[3],
                    "client order ID",
                    _CLIENT_ORDER_ID_MAX_LENGTH,
                ),
            )
            for row in rows
        )
    except (PaperAccountReadinessStorageError, TypeError, ValueError) as exc:
        raise PaperAccountReplayError("stored manifest inventory is invalid") from exc
    return tuple(sorted(references))


def _migration_only_assessment(
    *,
    context: PaperAccountReadinessContext,
    expected: tuple[MigrationIdentity, ...],
    applied: tuple[MigrationIdentity, ...],
    findings: tuple[PaperAccountReadinessFinding, ...],
) -> PaperAccountReadinessAssessment:
    return PaperAccountReadinessAssessment(
        context=context,
        expected_migrations=expected,
        applied_migrations=applied,
        account_version=None,
        legacy_watermarks=(),
        findings=findings,
    )


def _schema_drift_assessment(
    context: PaperAccountReadinessContext,
    expected: tuple[MigrationIdentity, ...],
    applied: tuple[MigrationIdentity, ...],
) -> PaperAccountReadinessAssessment:
    return _migration_only_assessment(
        context=context,
        expected=expected,
        applied=applied,
        findings=(_raw_migration_drift(),),
    )


def _read_legacy_watermarks(cursor: object) -> tuple[LegacyRelationWatermark, ...]:
    result = []
    for relation in _LEGACY_RELATIONS:
        cursor.execute(_SELECT_LEGACY_WATERMARK_SQL[relation])
        row = _one_row(cursor.fetchone(), f"{relation} watermark", 2)
        try:
            result.append(LegacyRelationWatermark(relation, row[0], row[1]))
        except (TypeError, ValueError) as exc:
            raise PaperAccountReplayError(
                f"stored {relation} watermark is invalid"
            ) from exc
    return tuple(result)


def _fresh_opening_provenance_finding(
    cursor: object,
    *,
    context: PaperAccountReadinessContext,
    terminal_migration: MigrationIdentity,
) -> PaperAccountReadinessFinding | None:
    cursor.execute(_SELECT_FRESH_OPENING_PROVENANCE_SQL)
    rows = tuple(cursor.fetchall())
    if not rows:
        return _finding(
            PaperAccountReadinessFindingKind.OPENING_PROVISIONING_ABSENT,
            "fresh_opening_provisioning",
            "np.paper_fresh_opening_provisionings",
        )

    try:
        if len(rows) != 1:
            raise ValueError("fresh opening provisioning is not a singleton")
        row = _one_row(rows[0], "paper fresh opening provenance", 32)
        execution_scope = _stored_key(row[1], "execution scope", 128)
        account_key = _stored_key(row[2], "account key", _ACCOUNT_KEY_MAX_LENGTH)
        opening_sha256 = _stored_sha256(row[5], "opening payload digest")
        candidate_sha256 = _stored_sha256(row[6], "candidate payload digest")
        pin_sha256 = _stored_sha256(row[7], "pin authority record digest")
        deployment_incarnation = _stored_key(
            row[8], "deployment incarnation", _ACCOUNT_KEY_MAX_LENGTH
        )
        database_incarnation_sha256 = _stored_sha256(
            row[9], "database incarnation digest"
        )
        migration_name = _stored_key(row[11], "migration name", 255)
        migration_checksum = _stored_sha256(row[12], "migration checksum")
        terminal_catalog_sha256 = _stored_sha256(row[13], "terminal catalog digest")
        provisioning_receipt_sha256 = _stored_sha256(
            row[21], "provisioning receipt digest"
        )
        admission_candidate_sha256 = _stored_sha256(
            row[23], "admission candidate digest"
        )
        admission_pin_sha256 = _stored_sha256(row[24], "admission pin authority digest")
        admission_deployment_incarnation = _stored_key(
            row[25], "admission deployment incarnation", _ACCOUNT_KEY_MAX_LENGTH
        )
        stream_execution_scope = _stored_key(row[26], "stream execution scope", 128)
        stream_account_key = _stored_key(
            row[27], "stream account key", _ACCOUNT_KEY_MAX_LENGTH
        )
        stream_opening_sha256 = _stored_sha256(row[30], "stream opening payload digest")
        if (
            row[0] is not True
            or execution_scope != context.execution_scope
            or account_key != context.account_key
            or type(row[3]) is not int
            or row[3] != context.owner_generation
            or type(row[4]) is not int
            or row[4] != 1
            or opening_sha256 != context.opening_payload_sha256
            or candidate_sha256 == "0" * 64
            or pin_sha256 == "0" * 64
            or database_incarnation_sha256 == "0" * 64
            or type(row[10]) is not int
            or row[10] != terminal_migration.version
            or migration_name != terminal_migration.name
            or migration_checksum != terminal_migration.checksum
            or terminal_catalog_sha256 == "0" * 64
            or row[14] != "LEGACY"
            or type(row[15]) is not int
            or row[15] != 0
            or type(row[16]) is not int
            or row[16] != 0
            or type(row[17]) is not int
            or row[17] != 0
            or row[18] is not False
            or row[19] is not False
            or row[20] is not True
            or provisioning_receipt_sha256 == "0" * 64
            or row[22] is not True
            or admission_candidate_sha256 != candidate_sha256
            or admission_pin_sha256 != pin_sha256
            or admission_deployment_incarnation != deployment_incarnation
            or stream_execution_scope != execution_scope
            or stream_account_key != account_key
            or type(row[28]) is not int
            or row[28] != context.owner_generation
            or type(row[29]) is not int
            or row[29] != 1
            or stream_opening_sha256 != opening_sha256
            or row[31] is not True
        ):
            raise ValueError("paper fresh opening provenance is not exact")
    except PaperAccountReadinessStorageError, TypeError, ValueError:
        return _finding(
            PaperAccountReadinessFindingKind.OPENING_PROVENANCE_MISMATCH,
            "fresh_opening_provisioning",
            "np.paper_fresh_opening_provisionings",
        )
    return None


def _assess_exact_schema(
    cursor: object,
    *,
    context: PaperAccountReadinessContext,
    expected: tuple[MigrationIdentity, ...],
    applied: tuple[MigrationIdentity, ...],
    runtime_mode: str,
    runtime_generation_evidence_is_exact: bool,
    required_runtime_mode: str,
    lock_replayed_state: bool,
) -> PaperAccountReadinessAssessment:
    findings = []
    account_version = None

    provisioning_finding = _fresh_opening_provenance_finding(
        cursor,
        context=context,
        terminal_migration=expected[-1],
    )
    if provisioning_finding is not None:
        findings.append(provisioning_finding)

    if runtime_mode != required_runtime_mode:
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.RUNTIME_CONTROL_NOT_LEGACY,
                "runtime_control",
                _RUNTIME_CONTROL_RELATION,
            )
        )

    if not runtime_generation_evidence_is_exact:
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.RUNTIME_GENERATION_MISMATCH,
                "runtime_generation",
                _RUNTIME_GENERATION_RELATION,
            )
        )

    raw_order_references = None
    try:
        cursor.execute(_SELECT_ORDER_REFERENCES_SQL)
        raw_order_references = _decode_order_references(cursor.fetchall())
        if len(
            {client_order_id for _, _, client_order_id in raw_order_references}
        ) != len(raw_order_references):
            raise JournalRepositoryError(
                "stored order inventory repeats a client identity"
            )
    except JournalRepositoryError:
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
                "durable_relation",
                "np.orders",
            )
        )

    raw_manifest_references = None
    try:
        cursor.execute(_SELECT_MANIFEST_REFERENCES_SQL)
        raw_manifest_references = _decode_manifest_references(cursor.fetchall())
        if len(
            {client_order_id for _, _, _, client_order_id in raw_manifest_references}
        ) != len(raw_manifest_references):
            raise PaperAccountReplayError(
                "stored manifest inventory repeats an order identity"
            )
    except PaperAccountReplayError:
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                "durable_relation",
                "np.paper_account_batch_manifests",
            )
        )

    try:
        cursor.execute(_SELECT_ACCOUNT_IDENTITIES_SQL)
        account_identities = _decode_identities(
            cursor.fetchall(), field="account key", maximum=_ACCOUNT_KEY_MAX_LENGTH
        )
    except PaperAccountReplayError:
        account_identities = ()
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                "durable_relation",
                "np.paper_account_streams",
            )
        )

    if (context.account_key, context.execution_scope) not in account_identities:
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED,
                "paper_account",
                context.account_key,
            )
        )
    for account_key, execution_scope in account_identities:
        if (account_key, execution_scope) != (
            context.account_key,
            context.execution_scope,
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.UNEXPECTED_ACCOUNT,
                    "paper_account",
                    account_key,
                )
            )

    replayed_accounts = {}
    replayed_manifest_references = []
    for account_key, execution_scope in account_identities:
        try:
            replayed = _replay_account_locked(
                cursor,
                execution_scope=execution_scope,
                account_key=account_key,
                lock=lock_replayed_state,
            )
        except PaperAccountJournalError:
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                    "paper_account",
                    account_key,
                )
            )
            continue
        replayed_accounts[(account_key, execution_scope)] = replayed
        replayed_manifest_references.extend(
            (
                account_key,
                execution_scope,
                batch.position_key,
                batch.client_order_id,
            )
            for batch in replayed.batches
        )
        if replayed.account.state is PaperAccountState.INSOLVENT:
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.ACCOUNT_INSOLVENT,
                    "paper_account",
                    account_key,
                )
            )
        for reservation in replayed.account.reservations:
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.MARGIN_RESERVATION_PRESENT,
                    "position_stream",
                    reservation.position_key,
                )
            )

    expected_account = replayed_accounts.get(
        (context.account_key, context.execution_scope)
    )
    if expected_account is not None:
        account_version = len(expected_account.account.records)
        if (
            expected_account.owner_generation != context.owner_generation
            or expected_account.opening_payload_sha256 != context.opening_payload_sha256
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.ACCOUNT_PROVENANCE_MISMATCH,
                    "paper_account",
                    context.account_key,
                )
            )

    try:
        cursor.execute(_SELECT_POSITION_IDENTITIES_SQL)
        position_identities = _decode_identities(
            cursor.fetchall(), field="position key", maximum=_POSITION_KEY_MAX_LENGTH
        )
    except PaperAccountReplayError:
        position_identities = ()
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
                "durable_relation",
                "np.position_streams",
            )
        )

    replayed_order_references = []
    for position_key, execution_scope in position_identities:
        try:
            projection = _replay_stream(
                cursor,
                execution_scope=execution_scope,
                position_key=position_key,
                lock=lock_replayed_state,
            ).projection
        except JournalRepositoryError:
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
                    "position_stream",
                    position_key,
                )
            )
            continue
        if projection.position is not None and (
            projection.position.state is PositionState.OPEN
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.DURABLE_OPEN_POSITION,
                    "position_stream",
                    position_key,
                )
            )
        for order in projection.orders:
            client_order_id = order.instruction.order_intent.client_order_id
            replayed_order_references.append(
                (position_key, execution_scope, client_order_id)
            )
            if order.lifecycle.state not in _TERMINAL_LIFECYCLE_STATES:
                findings.append(
                    _finding(
                        PaperAccountReadinessFindingKind.UNRESOLVED_SUBMISSION,
                        "client_order",
                        client_order_id,
                    )
                )

    replayed_orders = tuple(sorted(replayed_order_references))
    replayed_manifests = tuple(sorted(replayed_manifest_references))
    if raw_order_references is not None and Counter(raw_order_references) != Counter(
        replayed_orders
    ):
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
                "durable_relation",
                "np.orders",
            )
        )
    if raw_manifest_references is not None and Counter(
        raw_manifest_references
    ) != Counter(replayed_manifests):
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                "durable_relation",
                "np.paper_account_batch_manifests",
            )
        )

    if raw_order_references is not None and raw_manifest_references is not None:
        order_claims = Counter(raw_order_references)
        manifest_claims = Counter(
            (execution_scope, position_key, client_order_id)
            for _, execution_scope, position_key, client_order_id in (
                raw_manifest_references
            )
        )
        normalized_orders = Counter(
            (execution_scope, position_key, client_order_id)
            for position_key, execution_scope, client_order_id in (
                order_claims.elements()
            )
        )
        for _, _, client_order_id in sorted(
            (normalized_orders - manifest_claims).elements()
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
                    "client_order",
                    client_order_id,
                )
            )
        for _, _, client_order_id in sorted(
            (manifest_claims - normalized_orders).elements()
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                    "client_order",
                    client_order_id,
                )
            )

    watermarks = _read_legacy_watermarks(cursor)
    return PaperAccountReadinessAssessment(
        context=context,
        expected_migrations=expected,
        applied_migrations=applied,
        account_version=account_version,
        legacy_watermarks=watermarks,
        findings=tuple(findings),
    )


def _collect_paper_account_readiness(
    cursor: object,
    *,
    context: PaperAccountReadinessContext,
    required_runtime_mode: str,
    lock_replayed_state: bool,
) -> PaperAccountReadinessAssessment:
    """Collect complete evidence on a caller-owned cursor and transaction."""
    if type(context) is not PaperAccountReadinessContext:
        raise PaperAccountReadinessInputError(
            "context must be a PaperAccountReadinessContext"
        )
    if required_runtime_mode not in {"LEGACY", "PAUSED"}:
        raise PaperAccountReadinessInputError(
            "required_runtime_mode must be LEGACY or PAUSED"
        )
    if type(lock_replayed_state) is not bool:
        raise PaperAccountReadinessInputError("lock_replayed_state must be a boolean")

    expected = _expected_migrations()
    applied: tuple[MigrationIdentity, ...] = ()
    try:
        applied, migration_findings = _read_migration_evidence(cursor)
        if migration_findings or applied != expected:
            return _migration_only_assessment(
                context=context,
                expected=expected,
                applied=applied,
                findings=migration_findings,
            )
        if not _durable_business_relations_are_authoritative(cursor):
            return _schema_drift_assessment(context, expected, applied)

        runtime_control = (
            _read_runtime_control(cursor)
            if _runtime_control_catalog_is_exact(cursor)
            and _runtime_generation_catalog_is_exact(cursor)
            and _runtime_activation_capabilities_catalog_is_exact(cursor)
            else None
        )
        if runtime_control is None:
            return _schema_drift_assessment(context, expected, applied)
        return _assess_exact_schema(
            cursor,
            context=context,
            expected=expected,
            applied=applied,
            runtime_mode=runtime_control[0],
            runtime_generation_evidence_is_exact=(
                _runtime_generation_evidence_is_exact(
                    cursor,
                    context=context,
                    runtime_mode=runtime_control[0],
                    runtime_generation=runtime_control[1],
                )
            ),
            required_runtime_mode=required_runtime_mode,
            lock_replayed_state=lock_replayed_state,
        )
    except psycopg2.Error as exc:
        if getattr(exc, "pgcode", None) in _SCHEMA_DRIFT_SQLSTATES:
            return _schema_drift_assessment(context, expected, applied)
        raise


def _activation_catalog_is_authoritative(cursor: object) -> bool:
    """Prove the catalog authority required before replaying an activation ID."""
    try:
        expected = _expected_migrations()
        applied, migration_findings = _read_migration_evidence(cursor)
        return (
            not migration_findings
            and applied == expected
            and _durable_business_relations_are_authoritative(cursor)
            and _runtime_control_catalog_is_exact(cursor)
            and _runtime_generation_catalog_is_exact(cursor)
            and _runtime_activation_capabilities_catalog_is_exact(cursor)
        )
    except psycopg2.Error as exc:
        if getattr(exc, "pgcode", None) in _SCHEMA_DRIFT_SQLSTATES:
            return False
        raise


class PostgresPaperAccountReadiness:
    """Collect one stale-on-return assessment from a single read-only snapshot."""

    def __init__(self, connection_factory: Callable[[], object]) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._journal_boundary = PostgresOrderPositionJournal(connection_factory)

    def assess(
        self,
        context: PaperAccountReadinessContext,
        /,
    ) -> PaperAccountReadinessAssessment:
        """Assess durable evidence without committing or granting authority."""
        if type(context) is not PaperAccountReadinessContext:
            raise PaperAccountReadinessInputError(
                "context must be a PaperAccountReadinessContext"
            )
        try:
            connection = self._journal_boundary._connection()
        except JournalRepositoryError as exc:
            raise PaperAccountReadinessStorageError(
                "could not open a readiness assessment connection"
            ) from exc

        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_TRANSACTION_SQL)
                    result = _collect_paper_account_readiness(
                        cursor,
                        context=context,
                        required_runtime_mode="LEGACY",
                        lock_replayed_state=False,
                    )
            except psycopg2.Error as exc:
                raise PaperAccountReadinessStorageError(
                    "paper readiness assessment query failed"
                ) from exc
            except PaperAccountReadinessError:
                raise
            except (PaperAccountJournalError, JournalRepositoryError) as exc:
                raise PaperAccountReadinessStorageError(
                    "paper readiness assessment could not replay its snapshot"
                ) from exc
            except Exception as exc:
                raise PaperAccountReadinessStorageError(
                    "paper readiness assessment failed"
                ) from exc

            try:
                connection.rollback()
            except Exception as exc:
                raise PaperAccountReadinessStorageError(
                    "paper readiness snapshot could not finish"
                ) from exc
            return result
        except Exception:
            self._journal_boundary._rollback(connection)
            raise
        finally:
            self._journal_boundary._close(connection)


__all__ = [
    "PaperAccountReadinessError",
    "PaperAccountReadinessInputError",
    "PaperAccountReadinessStorageError",
    "PostgresPaperAccountReadiness",
]
