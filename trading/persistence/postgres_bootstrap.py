"""Dormant, operator-driven PostgreSQL role bootstrap for paper trading."""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

from trading.persistence.migration_runner import (
    MigrationDriftError,
    apply_migrations,
    load_migrations,
)
from trading.persistence.paper_account_readiness import (
    _activation_catalog_is_authoritative,
    _migration_metadata_is_exact,
)

_ROLE_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,62}")
_BOOTSTRAP_ADVISORY_LOCK_ID = 4_544_865_376_849_464
_ROLE_MARKER_PREFIX = "elvis-postgres-bootstrap:v1:"
_SCHEMA_MARKER_PREFIX = "elvis-postgres-bootstrap-schema:v1:"
_READ_COMMITTED_SQL = "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
_REPEATABLE_READ_ONLY_SQL = "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
_UTC_SQL = "SET LOCAL TIME ZONE 'UTC'"
_SAFE_SEARCH_PATH_SQL = "SET LOCAL search_path = pg_catalog"

_AUTHORITY_TABLES = (
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
_TERMINAL_DATA_TABLES = tuple(
    table
    for table in _AUTHORITY_TABLES
    if table not in ("paper_runtime_control", "schema_migrations")
)
_LEGACY_TABLES = (
    "account_balances",
    "liquidations",
    "margin_history",
    "model_predictions",
    "open_positions",
    "trades",
    "trading_session_resets",
)
_ATOMIC_TABLES = (
    "order_events",
    "orders",
    "paper_account_balances",
    "paper_account_batch_manifests",
    "paper_account_postings",
    "paper_account_settlements",
    "paper_account_streams",
    "paper_margin_reservations",
    "position_streams",
)
_LEGACY_SEQUENCES = tuple(f"{table}_id_seq" for table in _LEGACY_TABLES)
_READ_ONLY_TABLES = _AUTHORITY_TABLES
_TRAINER_TABLES = ("trades",)
_ACTIVATION_LOCK_TABLES = _AUTHORITY_TABLES
_ACTIVATION_READ_TABLES = _AUTHORITY_TABLES
_ACTIVATION_INSERT_TABLES = ("paper_runtime_generations",)
_NON_ACTIVATION_FUNCTIONS = (
    "np.enforce_legacy_paper_runtime_fence()",
    "np.reject_paper_runtime_generation_mutation()",
)
_ACTIVATION_FUNCTIONS = (
    "np.acquire_paper_runtime_activation_fence()",
    (
        "np.activate_paper_runtime_generation("
        "text,bigint,bigint,text,text,text,bigint,text)"
    ),
)
_EXPECTED_FUNCTION_IDENTITIES = (
    ("acquire_paper_runtime_activation_fence", "", "f"),
    (
        "activate_paper_runtime_generation",
        "expected_mode text, expected_generation bigint, target_generation "
        "bigint, requested_activation_id text, requested_execution_scope text, "
        "requested_account_key text, requested_owner_generation bigint, "
        "requested_opening_payload_sha256 text",
        "f",
    ),
    ("enforce_legacy_paper_runtime_fence", "", "f"),
    ("reject_paper_runtime_generation_mutation", "", "f"),
)
_EXPECTED_ACCESS_METHOD_EVIDENCE = (
    ("brin", "i", "brinhandler(internal)"),
    ("btree", "i", "bthandler(internal)"),
    ("gin", "i", "ginhandler(internal)"),
    ("gist", "i", "gisthandler(internal)"),
    ("hash", "i", "hashhandler(internal)"),
    ("heap", "t", "heap_tableam_handler(internal)"),
    ("spgist", "i", "spghandler(internal)"),
)
_EXPECTED_LANGUAGE_EVIDENCE_WITHOUT_OWNER = (
    ("c", False, False, "", "", "pg_catalog.fmgr_c_validator(oid)", True),
    (
        "internal",
        False,
        False,
        "",
        "",
        "pg_catalog.fmgr_internal_validator(oid)",
        True,
    ),
    (
        "plpgsql",
        True,
        True,
        "pg_catalog.plpgsql_call_handler()",
        "pg_catalog.plpgsql_inline_handler(internal)",
        "pg_catalog.plpgsql_validator(oid)",
        True,
    ),
    ("sql", False, True, "", "", "pg_catalog.fmgr_sql_validator(oid)", True),
)
_EXPECTED_HANDLER_PROCEDURE_EVIDENCE_WITHOUT_OWNER = (
    (
        "pg_catalog",
        "brinhandler",
        "internal",
        "index_am_handler",
        "internal",
        "f",
        "v",
        "s",
        True,
        False,
        False,
        False,
        1,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "brinhandler",
        "",
        True,
        True,
        True,
    ),
    (
        "pg_catalog",
        "bthandler",
        "internal",
        "index_am_handler",
        "internal",
        "f",
        "v",
        "s",
        True,
        False,
        False,
        False,
        1,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "bthandler",
        "",
        True,
        True,
        True,
    ),
    (
        "pg_catalog",
        "ginhandler",
        "internal",
        "index_am_handler",
        "internal",
        "f",
        "v",
        "s",
        True,
        False,
        False,
        False,
        1,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "ginhandler",
        "",
        True,
        True,
        True,
    ),
    (
        "pg_catalog",
        "gisthandler",
        "internal",
        "index_am_handler",
        "internal",
        "f",
        "v",
        "s",
        True,
        False,
        False,
        False,
        1,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "gisthandler",
        "",
        True,
        True,
        True,
    ),
    (
        "pg_catalog",
        "hashhandler",
        "internal",
        "index_am_handler",
        "internal",
        "f",
        "v",
        "s",
        True,
        False,
        False,
        False,
        1,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "hashhandler",
        "",
        True,
        True,
        True,
    ),
    (
        "pg_catalog",
        "heap_tableam_handler",
        "internal",
        "table_am_handler",
        "internal",
        "f",
        "v",
        "s",
        True,
        False,
        False,
        False,
        1,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "heap_tableam_handler",
        "",
        True,
        True,
        True,
    ),
    (
        "pg_catalog",
        "plpgsql_call_handler",
        "",
        "language_handler",
        "c",
        "f",
        "v",
        "u",
        False,
        False,
        False,
        False,
        0,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "plpgsql_call_handler",
        "$libdir/plpgsql",
        True,
        True,
        True,
    ),
    (
        "pg_catalog",
        "plpgsql_inline_handler",
        "internal",
        "void",
        "c",
        "f",
        "v",
        "u",
        True,
        False,
        False,
        False,
        1,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "plpgsql_inline_handler",
        "$libdir/plpgsql",
        True,
        True,
        True,
    ),
    (
        "pg_catalog",
        "plpgsql_validator",
        "oid",
        "void",
        "c",
        "f",
        "v",
        "u",
        True,
        False,
        False,
        False,
        1,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "plpgsql_validator",
        "$libdir/plpgsql",
        True,
        True,
        True,
    ),
    (
        "pg_catalog",
        "spghandler",
        "internal",
        "index_am_handler",
        "internal",
        "f",
        "v",
        "s",
        True,
        False,
        False,
        False,
        1,
        0,
        True,
        True,
        1.0,
        0.0,
        True,
        True,
        True,
        True,
        True,
        "spghandler",
        "",
        True,
        True,
        True,
    ),
)
_EXPECTED_PLPGSQL_DEPENDENCY_EVIDENCE = (
    ("extension_member", "pg_language", "", "plpgsql", "e"),
    (
        "extension_member",
        "pg_proc",
        "pg_catalog",
        "plpgsql_call_handler()",
        "e",
    ),
    (
        "extension_member",
        "pg_proc",
        "pg_catalog",
        "plpgsql_inline_handler(internal)",
        "e",
    ),
    (
        "extension_member",
        "pg_proc",
        "pg_catalog",
        "plpgsql_validator(oid)",
        "e",
    ),
    ("language_dependency", "pg_extension", "", "plpgsql", "e"),
    (
        "language_dependency",
        "pg_proc",
        "pg_catalog",
        "plpgsql_call_handler()",
        "n",
    ),
    (
        "language_dependency",
        "pg_proc",
        "pg_catalog",
        "plpgsql_inline_handler(internal)",
        "n",
    ),
    (
        "language_dependency",
        "pg_proc",
        "pg_catalog",
        "plpgsql_validator(oid)",
        "n",
    ),
)
_REQUIRED_STANDALONE_INDEXES = frozenset(
    {
        "idx_model_predictions_scored",
        "idx_trades_symbol_ts",
        "order_events_fill_identity_uq",
        "order_events_order_replay_idx",
        "order_events_paper_account_fill_ref_uq",
        "order_events_paper_account_submission_ref_uq",
        "orders_paper_account_batch_ref_uq",
        "orders_paper_account_symbol_ref_uq",
        "orders_venue_identity_uq",
    }
)
_EXPECTED_INDEX_NAMES = frozenset(
    {
        "account_balances_asset_key",
        "account_balances_pkey",
        "idx_model_predictions_scored",
        "idx_trades_symbol_ts",
        "liquidations_pkey",
        "margin_history_pkey",
        "model_predictions_pkey",
        "open_positions_pkey",
        "order_events_event_identity_uq",
        "order_events_fill_identity_uq",
        "order_events_order_replay_idx",
        "order_events_paper_account_fill_ref_uq",
        "order_events_paper_account_submission_ref_uq",
        "order_events_position_version_pk",
        "orders_paper_account_batch_ref_uq",
        "orders_paper_account_symbol_ref_uq",
        "orders_pkey",
        "orders_position_client_uq",
        "orders_scope_decision_uq",
        "orders_venue_identity_uq",
        "paper_account_balances_pk",
        "paper_account_batch_manifests_membership_uq",
        "paper_account_batch_manifests_order_owner_uq",
        "paper_account_batch_manifests_pk",
        "paper_account_postings_bucket_identity_uq",
        "paper_account_postings_pk",
        "paper_account_settlements_batch_ordinal_uq",
        "paper_account_settlements_event_identity_uq",
        "paper_account_settlements_fill_identity_uq",
        "paper_account_settlements_pk",
        "paper_account_settlements_position_version_uq",
        "paper_account_streams_collateral_identity_uq",
        "paper_account_streams_opening_envelope_uq",
        "paper_account_streams_opening_identity_uq",
        "paper_account_streams_pkey",
        "paper_account_streams_scope_identity_uq",
        "paper_margin_reservations_pk",
        "paper_runtime_control_pkey",
        "paper_runtime_generations_activation_id_uq",
        "paper_runtime_generations_manifest_ref_uq",
        "paper_runtime_generations_pkey",
        "position_streams_pkey",
        "position_streams_scope_identity_uq",
        "schema_migrations_pkey",
        "trades_pkey",
        "trading_session_resets_pkey",
    }
)
_EXPECTED_STANDALONE_INDEX_EVIDENCE = (
    (
        "idx_model_predictions_scored",
        "model_predictions",
        "btree",
        False,
        False,
        True,
        True,
        2,
        2,
        "[0:1]={7,2}",
        "",
        "",
    ),
    (
        "idx_trades_symbol_ts",
        "trades",
        "btree",
        False,
        False,
        True,
        True,
        2,
        2,
        "[0:1]={3,2}",
        "",
        "",
    ),
    (
        "order_events_fill_identity_uq",
        "order_events",
        "btree",
        True,
        False,
        True,
        True,
        2,
        2,
        "[0:1]={3,9}",
        "((event_type)::text = 'CONFIRMED_FILL'::text)",
        "",
    ),
    (
        "order_events_order_replay_idx",
        "order_events",
        "btree",
        False,
        False,
        True,
        True,
        2,
        2,
        "[0:1]={3,2}",
        "",
        "",
    ),
    (
        "order_events_paper_account_fill_ref_uq",
        "order_events",
        "btree",
        True,
        False,
        True,
        True,
        7,
        7,
        "[0:6]={1,2,3,4,9,5,8}",
        "",
        "",
    ),
    (
        "order_events_paper_account_submission_ref_uq",
        "order_events",
        "btree",
        True,
        False,
        True,
        True,
        7,
        7,
        "[0:6]={1,2,3,4,5,10,8}",
        "",
        "",
    ),
    (
        "orders_paper_account_batch_ref_uq",
        "orders",
        "btree",
        True,
        False,
        True,
        True,
        4,
        4,
        "[0:3]={3,1,4,9}",
        "",
        "",
    ),
    (
        "orders_paper_account_symbol_ref_uq",
        "orders",
        "btree",
        True,
        False,
        True,
        True,
        3,
        3,
        "[0:2]={3,1,5}",
        "",
        "",
    ),
    (
        "orders_venue_identity_uq",
        "orders",
        "btree",
        True,
        False,
        True,
        True,
        3,
        3,
        "[0:2]={4,5,10}",
        "(venue_order_id IS NOT NULL)",
        "",
    ),
)
_EXPECTED_COLUMN_EVIDENCE_SHA256 = (
    "3f5ec5bcb2aa0e37e1ab6e540743a79d8ecb8f161819eaf1305541ed3ed87464"
)
_EXPECTED_CONSTRAINT_EVIDENCE_SHA256 = (
    "d948983f3fdae119ead76a43c7a5d9adafcf3d5f592bb1b71809efbb6dd6a8c2"
)
_EXPECTED_INDEX_SECURITY_EVIDENCE_SHA256 = (
    "0c825eb6e5136237ad2db9d79eb74a8bc5f0b2ced97cc961e44f2b48c4592c4a"
)
_EXPECTED_SEQUENCE_EVIDENCE = tuple(
    (
        sequence,
        "integer",
        1,
        1,
        2_147_483_647,
        1,
        1,
        False,
        sequence.removesuffix("_id_seq"),
        "id",
    )
    for sequence in sorted(_LEGACY_SEQUENCES)
)
_TABLE_PRIVILEGES = (
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "TRUNCATE",
    "REFERENCES",
    "TRIGGER",
)
_SEQUENCE_PRIVILEGES = ("SELECT", "USAGE", "UPDATE")
_FUNCTION_PRIVILEGES = ("EXECUTE",)
_EXPECTED_ROLE_ATTRIBUTES = {
    "schema_owner": (False, False, False, False, False, False, False, -1),
    "migrator": (True, False, False, False, False, False, False, -1),
    "legacy_runtime": (True, False, False, False, False, False, False, -1),
    "atomic_runtime": (True, False, False, False, False, False, False, -1),
    "activation": (True, False, False, False, False, False, False, -1),
    "readiness": (True, False, False, False, False, False, False, -1),
    "trainer": (True, False, False, False, False, False, False, -1),
}
_ROLE_PURPOSES = (
    "schema_owner",
    "migrator",
    "legacy_runtime",
    "atomic_runtime",
    "activation",
    "readiness",
    "trainer",
)
_LEGACY_PRIVILEGES = {
    "account_balances": ("SELECT", "INSERT", "UPDATE"),
    "liquidations": ("SELECT", "INSERT"),
    "margin_history": ("SELECT", "INSERT"),
    "model_predictions": ("SELECT", "INSERT", "UPDATE"),
    "open_positions": ("SELECT", "INSERT", "DELETE"),
    "trades": ("SELECT", "INSERT", "DELETE"),
    "trading_session_resets": ("SELECT", "INSERT"),
}
_ATOMIC_PRIVILEGES = {
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

_SELECT_ADMIN_IDENTITY_SQL = """
SELECT
    current_database(),
    session_user,
    current_user,
    activity_row.usename,
    role_row.rolsuper,
    role_row.rolcanlogin
FROM pg_roles role_row
JOIN pg_stat_activity activity_row ON activity_row.pid = pg_backend_pid()
WHERE role_row.rolname = current_user
"""
_SELECT_MANAGED_ROLES_SQL = """
SELECT
    role_row.rolname,
    role_row.rolcanlogin,
    role_row.rolsuper,
    role_row.rolinherit,
    role_row.rolcreaterole,
    role_row.rolcreatedb,
    role_row.rolreplication,
    role_row.rolbypassrls,
    role_row.rolconnlimit,
    role_row.rolconfig,
    shobj_description(role_row.oid, 'pg_authid')
FROM pg_roles role_row
WHERE role_row.rolname = ANY(%s)
ORDER BY role_row.rolname
"""
_SELECT_MANAGED_MEMBERSHIPS_SQL = """
SELECT parent_role.rolname, member_role.rolname, membership.admin_option
FROM pg_auth_members membership
JOIN pg_roles parent_role ON parent_role.oid = membership.roleid
JOIN pg_roles member_role ON member_role.oid = membership.member
WHERE parent_role.rolname = ANY(%s)
   OR member_role.rolname = ANY(%s)
ORDER BY parent_role.rolname, member_role.rolname
"""
_SELECT_CREDENTIAL_IDENTITY_SQL = """
SELECT
    current_database(),
    session_user,
    current_user,
    activity_row.usename,
    role_row.rolcanlogin,
    role_row.rolsuper,
    role_row.rolinherit,
    role_row.rolcreaterole,
    role_row.rolcreatedb,
    role_row.rolreplication,
    role_row.rolbypassrls,
    role_row.rolconnlimit,
    role_row.rolconfig,
    shobj_description(role_row.oid, 'pg_authid')
FROM pg_roles role_row
JOIN pg_stat_activity activity_row ON activity_row.pid = pg_backend_pid()
WHERE role_row.rolname = current_user
"""
_SELECT_MANAGED_DATABASE_SETTINGS_SQL = """
SELECT
    COALESCE(role_row.rolname, ''),
    COALESCE(database_row.datname, ''),
    setting_row.setconfig
FROM pg_db_role_setting setting_row
LEFT JOIN pg_roles role_row ON role_row.oid = setting_row.setrole
LEFT JOIN pg_database database_row ON database_row.oid = setting_row.setdatabase
WHERE (role_row.rolname = ANY(%s) OR setting_row.setrole = 0)
  AND (database_row.datname = %s OR setting_row.setdatabase = 0)
ORDER BY 1, 2
"""
_SELECT_MANAGED_PASSWORD_STATES_SQL = """
SELECT
    role_row.rolname,
    role_row.rolpassword IS NULL,
    role_row.rolvaliduntil IS NULL OR role_row.rolvaliduntil > clock_timestamp()
FROM pg_authid role_row
WHERE role_row.rolname = ANY(%s)
ORDER BY role_row.rolname
"""
_SELECT_DATABASE_AUTHORITY_SQL = """
SELECT
    pg_get_userbyid(database_row.datdba),
    has_database_privilege(%s, database_row.oid, 'CREATE'),
    COALESCE(
        (
            SELECT BOOL_OR(
                database_acl.grantee = 0
                AND database_acl.privilege_type = 'CREATE'
            )
            FROM aclexplode(database_row.datacl) database_acl
        ),
        FALSE
    )
FROM pg_database database_row
WHERE database_row.datname = current_database()
"""
_SELECT_DATABASE_OWNER_SQL = """
SELECT pg_get_userbyid(database_row.datdba)
FROM pg_database database_row
WHERE database_row.datname = current_database()
"""
_SELECT_CLUSTER_SYSTEM_IDENTIFIER_SQL = """
SELECT system_identifier
FROM pg_control_system()
"""
_SELECT_TERMINAL_RUNTIME_CONTROL_SQL = """
SELECT mode, runtime_generation
FROM np.paper_runtime_control
ORDER BY control_key
"""
_SELECT_PUBLIC_SCHEMA_ACL_SQL = """
SELECT
    pg_get_userbyid(namespace_row.nspowner),
    COALESCE(grantee_role.rolname, 'PUBLIC'),
    schema_acl.privilege_type,
    schema_acl.is_grantable
FROM pg_namespace namespace_row
CROSS JOIN LATERAL aclexplode(
    COALESCE(namespace_row.nspacl, acldefault('n', namespace_row.nspowner))
) schema_acl
LEFT JOIN pg_roles grantee_role ON grantee_role.oid = schema_acl.grantee
WHERE namespace_row.nspname = 'public'
ORDER BY 2, 3
"""
_SELECT_UNEXPECTED_USER_SCHEMAS_SQL = """
SELECT namespace_row.nspname
FROM pg_namespace namespace_row
WHERE namespace_row.nspname NOT IN ('np', 'public', 'information_schema')
  AND namespace_row.nspname !~ '^pg_'
ORDER BY namespace_row.nspname
"""
_SELECT_UNEXPECTED_PUBLIC_OBJECTS_SQL = """
SELECT object_kind, object_name
FROM (
    SELECT 'relation'::text AS object_kind, table_row.relname::text AS object_name
    FROM pg_class table_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'routine'::text, function_row.proname::text
    FROM pg_proc function_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = function_row.pronamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'type'::text, type_row.typname::text
    FROM pg_type type_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = type_row.typnamespace
    WHERE namespace_row.nspname = 'public'
      AND type_row.typrelid = 0
      AND type_row.typelem = 0
    UNION ALL
    SELECT 'collation'::text, collation_row.collname::text
    FROM pg_collation collation_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = collation_row.collnamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'operator'::text, operator_row.oprname::text
    FROM pg_operator operator_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = operator_row.oprnamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'operator_class'::text, opclass_row.opcname::text
    FROM pg_opclass opclass_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = opclass_row.opcnamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'operator_family'::text, opfamily_row.opfname::text
    FROM pg_opfamily opfamily_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = opfamily_row.opfnamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'conversion'::text, conversion_row.conname::text
    FROM pg_conversion conversion_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = conversion_row.connamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'text_search_configuration'::text, config_row.cfgname::text
    FROM pg_ts_config config_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = config_row.cfgnamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'text_search_dictionary'::text, dictionary_row.dictname::text
    FROM pg_ts_dict dictionary_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = dictionary_row.dictnamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'text_search_parser'::text, parser_row.prsname::text
    FROM pg_ts_parser parser_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = parser_row.prsnamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'text_search_template'::text, template_row.tmplname::text
    FROM pg_ts_template template_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = template_row.tmplnamespace
    WHERE namespace_row.nspname = 'public'
    UNION ALL
    SELECT 'statistics'::text, statistics_row.stxname::text
    FROM pg_statistic_ext statistics_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = statistics_row.stxnamespace
    WHERE namespace_row.nspname = 'public'
) evidence
ORDER BY object_kind, object_name
"""
_SELECT_LARGE_OBJECT_COUNT_SQL = """
SELECT COUNT(*) FROM pg_largeobject_metadata
"""
_SELECT_EXTENSION_EVIDENCE_SQL = """
SELECT
    extension_row.extname,
    extension_row.extversion,
    extension_row.extrelocatable,
    namespace_row.nspname,
    pg_get_userbyid(extension_row.extowner),
    extension_row.extconfig IS NULL,
    extension_row.extcondition IS NULL
FROM pg_extension extension_row
JOIN pg_namespace namespace_row
  ON namespace_row.oid = extension_row.extnamespace
ORDER BY extension_row.extname
"""
_SELECT_LANGUAGE_EVIDENCE_SQL = """
SELECT
    language_row.lanname,
    pg_get_userbyid(language_row.lanowner),
    language_row.lanispl,
    language_row.lanpltrusted,
    CASE
        WHEN language_row.lanplcallfoid = 0 THEN ''
        ELSE call_namespace.nspname || '.' || call_function.proname || '('
             || pg_get_function_identity_arguments(call_function.oid) || ')'
    END,
    CASE
        WHEN language_row.laninline = 0 THEN ''
        ELSE inline_namespace.nspname || '.' || inline_function.proname || '('
             || pg_get_function_identity_arguments(inline_function.oid) || ')'
    END,
    CASE
        WHEN language_row.lanvalidator = 0 THEN ''
        ELSE validator_namespace.nspname || '.' || validator_function.proname || '('
             || pg_get_function_identity_arguments(validator_function.oid) || ')'
    END,
    language_row.lanacl IS NULL
FROM pg_language language_row
LEFT JOIN pg_proc call_function
  ON call_function.oid = language_row.lanplcallfoid
LEFT JOIN pg_namespace call_namespace
  ON call_namespace.oid = call_function.pronamespace
LEFT JOIN pg_proc inline_function
  ON inline_function.oid = language_row.laninline
LEFT JOIN pg_namespace inline_namespace
  ON inline_namespace.oid = inline_function.pronamespace
LEFT JOIN pg_proc validator_function
  ON validator_function.oid = language_row.lanvalidator
LEFT JOIN pg_namespace validator_namespace
  ON validator_namespace.oid = validator_function.pronamespace
ORDER BY language_row.lanname
"""
_SELECT_ACCESS_METHOD_EVIDENCE_SQL = """
SELECT
    access_method.amname,
    access_method.amtype,
    access_method.amhandler::regprocedure::text
FROM pg_am access_method
ORDER BY access_method.amname
"""
_SELECT_HANDLER_PROCEDURE_EVIDENCE_SQL = """
WITH handler_oids AS (
    SELECT language_row.lanplcallfoid AS oid
    FROM pg_language language_row
    WHERE language_row.lanname = 'plpgsql'
    UNION
    SELECT language_row.laninline
    FROM pg_language language_row
    WHERE language_row.lanname = 'plpgsql'
    UNION
    SELECT language_row.lanvalidator
    FROM pg_language language_row
    WHERE language_row.lanname = 'plpgsql'
    UNION
    SELECT access_method.amhandler FROM pg_am access_method
)
SELECT
    namespace_row.nspname,
    function_row.proname,
    pg_get_function_identity_arguments(function_row.oid),
    format_type(function_row.prorettype, NULL),
    pg_get_userbyid(function_row.proowner),
    language_row.lanname,
    function_row.prokind,
    function_row.provolatile,
    function_row.proparallel,
    function_row.proisstrict,
    function_row.prosecdef,
    function_row.proleakproof,
    function_row.proretset,
    function_row.pronargs,
    function_row.pronargdefaults,
    function_row.provariadic = 0,
    function_row.prosupport = 0,
    function_row.procost::double precision,
    function_row.prorows::double precision,
    function_row.proallargtypes IS NULL,
    function_row.proargmodes IS NULL,
    function_row.proargnames IS NULL,
    function_row.proargdefaults IS NULL,
    function_row.protrftypes IS NULL,
    function_row.prosrc,
    COALESCE(function_row.probin, ''),
    function_row.prosqlbody IS NULL,
    function_row.proconfig IS NULL,
    function_row.proacl IS NULL
FROM pg_proc function_row
JOIN handler_oids ON handler_oids.oid = function_row.oid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = function_row.pronamespace
JOIN pg_language language_row ON language_row.oid = function_row.prolang
ORDER BY
    namespace_row.nspname,
    function_row.proname,
    pg_get_function_identity_arguments(function_row.oid)
"""
_SELECT_PLPGSQL_DEPENDENCY_EVIDENCE_SQL = """
WITH plpgsql_extension AS (
    SELECT extension_row.oid
    FROM pg_extension extension_row
    WHERE extension_row.extname = 'plpgsql'
), plpgsql_language AS (
    SELECT language_row.oid
    FROM pg_language language_row
    WHERE language_row.lanname = 'plpgsql'
), evidence AS (
    SELECT
        'extension_member'::text AS evidence_kind,
        dependency_row.classid::regclass::text AS object_class,
        CASE
            WHEN function_row.oid IS NULL THEN ''
            ELSE function_namespace.nspname
        END AS namespace_name,
        CASE
            WHEN function_row.oid IS NOT NULL
                THEN function_row.proname || '('
                     || pg_get_function_identity_arguments(function_row.oid) || ')'
            WHEN language_row.oid IS NOT NULL THEN language_row.lanname
            ELSE dependency_row.objid::text
        END AS object_identity,
        dependency_row.deptype
    FROM pg_depend dependency_row
    JOIN plpgsql_extension
      ON dependency_row.refclassid = 'pg_extension'::regclass
     AND dependency_row.refobjid = plpgsql_extension.oid
    LEFT JOIN pg_proc function_row
      ON dependency_row.classid = 'pg_proc'::regclass
     AND function_row.oid = dependency_row.objid
    LEFT JOIN pg_namespace function_namespace
      ON function_namespace.oid = function_row.pronamespace
    LEFT JOIN pg_language language_row
      ON dependency_row.classid = 'pg_language'::regclass
     AND language_row.oid = dependency_row.objid
    UNION ALL
    SELECT
        'language_dependency'::text,
        dependency_row.refclassid::regclass::text,
        CASE
            WHEN function_row.oid IS NULL THEN ''
            ELSE function_namespace.nspname
        END,
        CASE
            WHEN function_row.oid IS NOT NULL
                THEN function_row.proname || '('
                     || pg_get_function_identity_arguments(function_row.oid) || ')'
            WHEN extension_row.oid IS NOT NULL THEN extension_row.extname
            ELSE dependency_row.refobjid::text
        END,
        dependency_row.deptype
    FROM pg_depend dependency_row
    JOIN plpgsql_language
      ON dependency_row.classid = 'pg_language'::regclass
     AND dependency_row.objid = plpgsql_language.oid
    LEFT JOIN pg_proc function_row
      ON dependency_row.refclassid = 'pg_proc'::regclass
     AND function_row.oid = dependency_row.refobjid
    LEFT JOIN pg_namespace function_namespace
      ON function_namespace.oid = function_row.pronamespace
    LEFT JOIN pg_extension extension_row
      ON dependency_row.refclassid = 'pg_extension'::regclass
     AND extension_row.oid = dependency_row.refobjid
)
SELECT
    evidence_kind,
    object_class,
    namespace_name,
    object_identity,
    deptype
FROM evidence
ORDER BY 1, 2, 3, 4, 5
"""
_SELECT_UNEXPECTED_PG_CATALOG_OBJECTS_SQL = """
SELECT object_kind, object_name
FROM (
    SELECT 'relation'::text AS object_kind, table_row.relname::text AS object_name
    FROM pg_class table_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND table_row.oid >= 16384
    UNION ALL
    SELECT 'routine'::text, function_row.proname::text
    FROM pg_proc function_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = function_row.pronamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND function_row.oid >= 16384
    UNION ALL
    SELECT 'type'::text, type_row.typname::text
    FROM pg_type type_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = type_row.typnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND type_row.oid >= 16384
      AND type_row.typrelid = 0
      AND type_row.typelem = 0
    UNION ALL
    SELECT 'collation'::text, collation_row.collname::text
    FROM pg_collation collation_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = collation_row.collnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND collation_row.oid >= 16384
    UNION ALL
    SELECT 'operator'::text, operator_row.oprname::text
    FROM pg_operator operator_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = operator_row.oprnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND operator_row.oid >= 16384
    UNION ALL
    SELECT 'operator_class'::text, opclass_row.opcname::text
    FROM pg_opclass opclass_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = opclass_row.opcnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND opclass_row.oid >= 16384
    UNION ALL
    SELECT 'operator_family'::text, opfamily_row.opfname::text
    FROM pg_opfamily opfamily_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = opfamily_row.opfnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND opfamily_row.oid >= 16384
    UNION ALL
    SELECT 'conversion'::text, conversion_row.conname::text
    FROM pg_conversion conversion_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = conversion_row.connamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND conversion_row.oid >= 16384
    UNION ALL
    SELECT 'text_search_configuration'::text, config_row.cfgname::text
    FROM pg_ts_config config_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = config_row.cfgnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND config_row.oid >= 16384
    UNION ALL
    SELECT 'text_search_dictionary'::text, dictionary_row.dictname::text
    FROM pg_ts_dict dictionary_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = dictionary_row.dictnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND dictionary_row.oid >= 16384
    UNION ALL
    SELECT 'text_search_parser'::text, parser_row.prsname::text
    FROM pg_ts_parser parser_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = parser_row.prsnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND parser_row.oid >= 16384
    UNION ALL
    SELECT 'text_search_template'::text, template_row.tmplname::text
    FROM pg_ts_template template_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = template_row.tmplnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND template_row.oid >= 16384
    UNION ALL
    SELECT 'statistics'::text, statistics_row.stxname::text
    FROM pg_statistic_ext statistics_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = statistics_row.stxnamespace
    WHERE namespace_row.nspname = 'pg_catalog'
      AND statistics_row.oid >= 16384
) evidence
ORDER BY object_kind, object_name
"""
_SELECT_UNEXPECTED_DATABASE_OBJECTS_SQL = """
SELECT object_kind, object_name
FROM (
    SELECT 'event_trigger'::text AS object_kind, trigger_row.evtname::text AS object_name
    FROM pg_event_trigger trigger_row
    UNION ALL
    SELECT 'foreign_data_wrapper'::text, wrapper_row.fdwname::text
    FROM pg_foreign_data_wrapper wrapper_row
    UNION ALL
    SELECT 'foreign_server'::text, server_row.srvname::text
    FROM pg_foreign_server server_row
    UNION ALL
    SELECT 'user_mapping'::text, mapping_row.oid::text
    FROM pg_user_mapping mapping_row
    UNION ALL
    SELECT 'publication'::text, publication_row.pubname::text
    FROM pg_publication publication_row
    UNION ALL
    SELECT 'subscription'::text, subscription_row.subname::text
    FROM pg_subscription subscription_row
    WHERE subscription_row.subdbid = (
        SELECT database_row.oid
        FROM pg_database database_row
        WHERE database_row.datname = current_database()
    )
    UNION ALL
    SELECT 'default_acl'::text, default_acl.oid::text
    FROM pg_default_acl default_acl
    UNION ALL
    SELECT 'user_cast'::text, cast_row.oid::text
    FROM pg_cast cast_row
    WHERE cast_row.oid >= 16384
    UNION ALL
    SELECT 'transform'::text, transform_row.oid::text
    FROM pg_transform transform_row
    UNION ALL
    SELECT 'security_label'::text, label_row.objoid::text
    FROM pg_seclabel label_row
) evidence
ORDER BY object_kind, object_name
"""
_SELECT_RELEVANT_DATABASE_SETTINGS_SQL = """
SELECT
    COALESCE(database_row.datname, ''),
    COALESCE(role_row.rolname, ''),
    setting_row.setconfig
FROM pg_db_role_setting setting_row
LEFT JOIN pg_database database_row ON database_row.oid = setting_row.setdatabase
LEFT JOIN pg_roles role_row ON role_row.oid = setting_row.setrole
WHERE (setting_row.setdatabase = 0 OR database_row.datname = current_database())
  AND (setting_row.setrole = 0 OR role_row.rolname = ANY(%s))
ORDER BY 1, 2
"""
_SELECT_RELEVANT_SHARED_SECURITY_LABELS_SQL = """
SELECT label_row.classoid::regclass::text, label_row.objoid, label_row.provider
FROM pg_shseclabel label_row
WHERE (
    label_row.classoid = 'pg_database'::regclass
    AND label_row.objoid = (
        SELECT database_row.oid
        FROM pg_database database_row
        WHERE database_row.datname = current_database()
    )
) OR (
    label_row.classoid = 'pg_authid'::regclass
    AND label_row.objoid = ANY(
        SELECT role_row.oid FROM pg_roles role_row WHERE role_row.rolname = ANY(%s)
    )
)
ORDER BY 1, 2, 3
"""
_SELECT_RELEVANT_PARAMETER_ACLS_SQL = """
SELECT
    parameter_acl.parname,
    COALESCE(grantee_role.rolname, 'PUBLIC'),
    exploded_acl.privilege_type,
    exploded_acl.is_grantable
FROM pg_parameter_acl parameter_acl
CROSS JOIN LATERAL aclexplode(parameter_acl.paracl) exploded_acl
LEFT JOIN pg_roles grantee_role ON grantee_role.oid = exploded_acl.grantee
WHERE exploded_acl.grantee = 0 OR grantee_role.rolname = ANY(%s)
ORDER BY 1, 2, 3
"""
_SELECT_UNEXPECTED_NP_CATALOG_OBJECTS_SQL = """
SELECT object_kind, object_name
FROM (
    SELECT 'type'::text AS object_kind, type_row.typname::text AS object_name
    FROM pg_type type_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = type_row.typnamespace
    LEFT JOIN pg_class table_row ON table_row.oid = type_row.typrelid
    WHERE namespace_row.nspname = 'np'
      AND type_row.typelem = 0
      AND NOT (
          type_row.typtype = 'c'
          AND table_row.relkind = 'r'
          AND table_row.relname = ANY(%s)
      )
    UNION ALL
    SELECT 'collation'::text, collation_row.collname::text
    FROM pg_collation collation_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = collation_row.collnamespace
    WHERE namespace_row.nspname = 'np'
    UNION ALL
    SELECT 'operator'::text, operator_row.oprname::text
    FROM pg_operator operator_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = operator_row.oprnamespace
    WHERE namespace_row.nspname = 'np'
    UNION ALL
    SELECT 'operator_class'::text, opclass_row.opcname::text
    FROM pg_opclass opclass_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = opclass_row.opcnamespace
    WHERE namespace_row.nspname = 'np'
    UNION ALL
    SELECT 'operator_family'::text, opfamily_row.opfname::text
    FROM pg_opfamily opfamily_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = opfamily_row.opfnamespace
    WHERE namespace_row.nspname = 'np'
    UNION ALL
    SELECT 'conversion'::text, conversion_row.conname::text
    FROM pg_conversion conversion_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = conversion_row.connamespace
    WHERE namespace_row.nspname = 'np'
    UNION ALL
    SELECT 'text_search_configuration'::text, config_row.cfgname::text
    FROM pg_ts_config config_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = config_row.cfgnamespace
    WHERE namespace_row.nspname = 'np'
    UNION ALL
    SELECT 'text_search_dictionary'::text, dictionary_row.dictname::text
    FROM pg_ts_dict dictionary_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = dictionary_row.dictnamespace
    WHERE namespace_row.nspname = 'np'
    UNION ALL
    SELECT 'text_search_parser'::text, parser_row.prsname::text
    FROM pg_ts_parser parser_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = parser_row.prsnamespace
    WHERE namespace_row.nspname = 'np'
    UNION ALL
    SELECT 'text_search_template'::text, template_row.tmplname::text
    FROM pg_ts_template template_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = template_row.tmplnamespace
    WHERE namespace_row.nspname = 'np'
    UNION ALL
    SELECT 'statistics'::text, statistics_row.stxname::text
    FROM pg_statistic_ext statistics_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = statistics_row.stxnamespace
    WHERE namespace_row.nspname = 'np'
) evidence
ORDER BY object_kind, object_name
"""
_SELECT_SCHEMA_OBJECTS_SQL = """
SELECT table_row.relname, table_row.relkind, pg_get_userbyid(table_row.relowner)
FROM pg_class table_row
JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
ORDER BY table_row.relkind, table_row.relname
"""
_SELECT_MANAGED_OWNERSHIP_OUTSIDE_NP_SQL = """
WITH managed_roles AS (
    SELECT role_row.oid
    FROM pg_roles role_row
    WHERE role_row.rolname = ANY(%s)
), evidence AS (
    SELECT
        'schema'::text AS object_kind,
        namespace_row.nspname::text AS namespace_name,
        ''::text AS object_name,
        pg_get_userbyid(namespace_row.nspowner)::text AS owner_name
    FROM pg_namespace namespace_row
    WHERE namespace_row.nspowner = ANY(SELECT oid FROM managed_roles)
      AND namespace_row.nspname <> 'np'
      AND namespace_row.nspname NOT IN ('pg_catalog', 'information_schema')
      AND namespace_row.nspname !~ '^pg_toast'
    UNION ALL
    SELECT
        'relation'::text,
        namespace_row.nspname::text,
        table_row.relname::text,
        pg_get_userbyid(table_row.relowner)::text
    FROM pg_class table_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
    WHERE table_row.relowner = ANY(SELECT oid FROM managed_roles)
      AND namespace_row.nspname <> 'np'
      AND namespace_row.nspname NOT IN ('pg_catalog', 'information_schema')
      AND namespace_row.nspname !~ '^pg_toast'
    UNION ALL
    SELECT
        'function'::text,
        namespace_row.nspname::text,
        function_row.proname::text,
        pg_get_userbyid(function_row.proowner)::text
    FROM pg_proc function_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = function_row.pronamespace
    WHERE function_row.proowner = ANY(SELECT oid FROM managed_roles)
      AND namespace_row.nspname <> 'np'
      AND namespace_row.nspname NOT IN ('pg_catalog', 'information_schema')
      AND namespace_row.nspname !~ '^pg_toast'
)
SELECT object_kind, namespace_name, object_name, owner_name
FROM evidence
ORDER BY 1, 2, 3
"""
_SELECT_MANAGED_ACLS_OUTSIDE_NP_SQL = """
WITH managed_roles AS (
    SELECT role_row.oid
    FROM pg_roles role_row
    WHERE role_row.rolname = ANY(%s)
), evidence AS (
    SELECT 'schema'::text AS object_kind, namespace_row.nspname::text AS object_name
    FROM pg_namespace namespace_row
    CROSS JOIN LATERAL aclexplode(namespace_row.nspacl) schema_acl
    WHERE schema_acl.grantee = ANY(SELECT oid FROM managed_roles)
      AND namespace_row.nspname <> 'np'
      AND namespace_row.nspname NOT IN ('pg_catalog', 'information_schema')
      AND namespace_row.nspname !~ '^pg_toast'
    UNION ALL
    SELECT 'relation'::text, table_row.relname::text
    FROM pg_class table_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
    CROSS JOIN LATERAL aclexplode(table_row.relacl) relation_acl
    WHERE relation_acl.grantee = ANY(SELECT oid FROM managed_roles)
      AND namespace_row.nspname <> 'np'
      AND namespace_row.nspname NOT IN ('pg_catalog', 'information_schema')
      AND namespace_row.nspname !~ '^pg_toast'
    UNION ALL
    SELECT 'function'::text, function_row.proname::text
    FROM pg_proc function_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = function_row.pronamespace
    CROSS JOIN LATERAL aclexplode(function_row.proacl) function_acl
    WHERE function_acl.grantee = ANY(SELECT oid FROM managed_roles)
      AND namespace_row.nspname <> 'np'
      AND namespace_row.nspname NOT IN ('pg_catalog', 'information_schema')
      AND namespace_row.nspname !~ '^pg_toast'
)
SELECT object_kind, object_name FROM evidence ORDER BY 1, 2
"""
_SELECT_SCHEMA_FUNCTIONS_SQL = """
SELECT
    function_row.proname,
    pg_get_function_identity_arguments(function_row.oid),
    function_row.prokind,
    pg_get_userbyid(function_row.proowner)
FROM pg_proc function_row
JOIN pg_namespace namespace_row ON namespace_row.oid = function_row.pronamespace
WHERE namespace_row.nspname = 'np'
ORDER BY function_row.proname, function_row.oid
"""
_SELECT_SCHEMA_AUTHORITY_SQL = """
SELECT
    pg_get_userbyid(namespace_row.nspowner),
    obj_description(namespace_row.oid, 'pg_namespace')
FROM pg_namespace namespace_row
WHERE namespace_row.nspname = 'np'
"""
_SELECT_SCHEMA_ACL_GRANTEES_SQL = """
SELECT
    object_kind,
    object_name,
    object_identity,
    grantor_name,
    grantee_name,
    privilege_type,
    is_grantable
FROM (
    SELECT
        'schema'::text AS object_kind,
        namespace_row.nspname::text AS object_name,
        ''::text AS object_identity,
        pg_get_userbyid(schema_acl.grantor)::text AS grantor_name,
        COALESCE(grantee_role.rolname, 'PUBLIC')::text AS grantee_name,
        schema_acl.privilege_type::text AS privilege_type,
        schema_acl.is_grantable
    FROM pg_namespace namespace_row
    CROSS JOIN LATERAL aclexplode(namespace_row.nspacl) schema_acl
    LEFT JOIN pg_roles grantee_role ON grantee_role.oid = schema_acl.grantee
    WHERE namespace_row.nspname = 'np'
      AND schema_acl.grantee <> namespace_row.nspowner
    UNION ALL
    SELECT
        'relation'::text AS object_kind,
        table_row.relname::text AS object_name,
        ''::text AS object_identity,
        pg_get_userbyid(relation_acl.grantor)::text,
        COALESCE(grantee_role.rolname, 'PUBLIC')::text AS grantee_name,
        relation_acl.privilege_type::text AS privilege_type,
        relation_acl.is_grantable
    FROM pg_class table_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = table_row.relnamespace
    CROSS JOIN LATERAL aclexplode(table_row.relacl) relation_acl
    LEFT JOIN pg_roles grantee_role ON grantee_role.oid = relation_acl.grantee
    WHERE namespace_row.nspname = 'np'
      AND table_row.relkind IN ('r', 'S')
      AND relation_acl.grantee <> table_row.relowner
    UNION ALL
    SELECT
        'function'::text,
        function_row.proname::text,
        pg_get_function_identity_arguments(function_row.oid)::text,
        pg_get_userbyid(function_acl.grantor)::text,
        COALESCE(grantee_role.rolname, 'PUBLIC')::text,
        function_acl.privilege_type::text,
        function_acl.is_grantable
    FROM pg_proc function_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = function_row.pronamespace
    CROSS JOIN LATERAL aclexplode(
        COALESCE(function_row.proacl, acldefault('f', function_row.proowner))
    ) function_acl
    LEFT JOIN pg_roles grantee_role ON grantee_role.oid = function_acl.grantee
    WHERE namespace_row.nspname = 'np'
      AND function_acl.grantee <> function_row.proowner
) acl_rows
ORDER BY object_kind, object_name, object_identity, grantee_name, privilege_type
"""
_SELECT_DATABASE_ACL_GRANTEES_SQL = """
SELECT
    pg_get_userbyid(database_acl.grantor),
    COALESCE(grantee_role.rolname, 'PUBLIC'),
    database_acl.privilege_type,
    database_acl.is_grantable
FROM pg_database database_row
CROSS JOIN LATERAL aclexplode(
    COALESCE(database_row.datacl, acldefault('d', database_row.datdba))
) database_acl
LEFT JOIN pg_roles grantee_role ON grantee_role.oid = database_acl.grantee
WHERE database_row.datname = current_database()
  AND database_acl.grantee <> database_row.datdba
ORDER BY 1, 2, 3, 4
"""
_SELECT_DEFAULT_ACLS_SQL = """
SELECT
    owner_role.rolname,
    COALESCE(namespace_row.nspname, ''),
    default_acl.defaclobjtype,
    COALESCE(grantee_role.rolname, 'PUBLIC'),
    exploded_acl.privilege_type,
    exploded_acl.is_grantable
FROM pg_default_acl default_acl
JOIN pg_roles owner_role ON owner_role.oid = default_acl.defaclrole
LEFT JOIN pg_namespace namespace_row
  ON namespace_row.oid = default_acl.defaclnamespace
CROSS JOIN LATERAL aclexplode(default_acl.defaclacl) exploded_acl
LEFT JOIN pg_roles grantee_role ON grantee_role.oid = exploded_acl.grantee
WHERE default_acl.defaclrole = ANY(
    SELECT role_row.oid FROM pg_roles role_row WHERE role_row.rolname = ANY(%s)
)
   OR default_acl.defaclnamespace = 'np'::regnamespace
ORDER BY 1, 2, 3, 4, 5
"""
_SELECT_COLUMN_ACLS_SQL = """
SELECT
    table_row.relname,
    column_row.attname,
    pg_get_userbyid(column_acl.grantor),
    COALESCE(grantee_role.rolname, 'PUBLIC'),
    column_acl.privilege_type,
    column_acl.is_grantable
FROM pg_attribute column_row
JOIN pg_class table_row ON table_row.oid = column_row.attrelid
JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
CROSS JOIN LATERAL aclexplode(column_row.attacl) column_acl
LEFT JOIN pg_roles grantee_role ON grantee_role.oid = column_acl.grantee
WHERE namespace_row.nspname = 'np'
  AND column_row.attnum > 0
  AND NOT column_row.attisdropped
ORDER BY 1, 2, 4, 5
"""
_SELECT_INDEX_EVIDENCE_SQL = """
SELECT
    index_row.relname,
    table_row.relname,
    access_method.amname,
    index_catalog.indisunique,
    index_catalog.indisprimary,
    index_catalog.indisvalid,
    index_catalog.indisready,
    index_catalog.indnkeyatts,
    index_catalog.indnatts,
    index_catalog.indkey::int2[]::text,
    COALESCE(pg_get_expr(index_catalog.indpred, index_catalog.indrelid), ''),
    COALESCE(pg_get_expr(index_catalog.indexprs, index_catalog.indrelid), ''),
    index_row.relpersistence,
    index_catalog.indislive,
    index_catalog.indisclustered,
    index_catalog.indisreplident,
    index_catalog.indnullsnotdistinct,
    ARRAY(
        SELECT FORMAT('%I.%I', opclass_namespace.nspname, opclass_row.opcname)
        FROM UNNEST(index_catalog.indclass::oid[]) WITH ORDINALITY
            AS opclass_item(opclass_oid, ordinal)
        JOIN pg_opclass opclass_row ON opclass_row.oid = opclass_item.opclass_oid
        JOIN pg_namespace opclass_namespace
          ON opclass_namespace.oid = opclass_row.opcnamespace
        ORDER BY opclass_item.ordinal
    )::text,
    ARRAY(
        SELECT CASE
            WHEN collation_item.collation_oid = 0 THEN ''
            ELSE FORMAT(
                '%I.%I',
                collation_namespace.nspname,
                collation_row.collname
            )
        END
        FROM UNNEST(index_catalog.indcollation::oid[]) WITH ORDINALITY
            AS collation_item(collation_oid, ordinal)
        LEFT JOIN pg_collation collation_row
          ON collation_row.oid = collation_item.collation_oid
        LEFT JOIN pg_namespace collation_namespace
          ON collation_namespace.oid = collation_row.collnamespace
        ORDER BY collation_item.ordinal
    )::text,
    index_catalog.indoption::int2[]::text,
    index_row.relowner = table_row.relowner,
    NOT EXISTS (
        SELECT 1
        FROM UNNEST(index_catalog.indclass::oid[]) AS opclass_item(opclass_oid)
        JOIN pg_opclass opclass_row ON opclass_row.oid = opclass_item.opclass_oid
        JOIN pg_namespace opclass_namespace
          ON opclass_namespace.oid = opclass_row.opcnamespace
        WHERE opclass_namespace.nspname <> 'pg_catalog'
           OR NOT opclass_row.opcdefault
    ),
    NOT EXISTS (
        SELECT 1
        FROM UNNEST(index_catalog.indcollation::oid[])
            AS collation_item(collation_oid)
        LEFT JOIN pg_collation collation_row
          ON collation_row.oid = collation_item.collation_oid
        LEFT JOIN pg_namespace collation_namespace
          ON collation_namespace.oid = collation_row.collnamespace
        WHERE collation_item.collation_oid <> 0
          AND collation_namespace.nspname <> 'pg_catalog'
    ),
    NOT EXISTS (
        SELECT 1
        FROM UNNEST(index_catalog.indoption::smallint[]) AS option_item(option_value)
        WHERE option_item.option_value <> 0
    )
FROM pg_index index_catalog
JOIN pg_class index_row ON index_row.oid = index_catalog.indexrelid
JOIN pg_class table_row ON table_row.oid = index_catalog.indrelid
JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
JOIN pg_am access_method ON access_method.oid = index_row.relam
WHERE namespace_row.nspname = 'np'
ORDER BY index_row.relname
"""
_SELECT_COLUMN_EVIDENCE_SQL = """
SELECT
    table_row.relname,
    column_row.attnum,
    column_row.attname,
    format_type(column_row.atttypid, column_row.atttypmod),
    column_row.attnotnull,
    column_row.attidentity,
    column_row.attgenerated,
    column_row.attcollation::regcollation::text,
    COALESCE(pg_get_expr(default_row.adbin, default_row.adrelid), '')
FROM pg_class table_row
JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
JOIN pg_attribute column_row ON column_row.attrelid = table_row.oid
LEFT JOIN pg_attrdef default_row
  ON default_row.adrelid = table_row.oid
 AND default_row.adnum = column_row.attnum
WHERE namespace_row.nspname = 'np'
  AND table_row.relkind = 'r'
  AND column_row.attnum > 0
  AND NOT column_row.attisdropped
ORDER BY table_row.relname, column_row.attnum
"""
_SELECT_SEQUENCE_EVIDENCE_SQL = """
SELECT
    sequence_row.relname,
    format_type(sequence_catalog.seqtypid, NULL),
    sequence_catalog.seqstart,
    sequence_catalog.seqincrement,
    sequence_catalog.seqmax,
    sequence_catalog.seqmin,
    sequence_catalog.seqcache,
    sequence_catalog.seqcycle,
    COALESCE(owner_table.relname, ''),
    COALESCE(owner_column.attname, '')
FROM pg_class sequence_row
JOIN pg_namespace namespace_row ON namespace_row.oid = sequence_row.relnamespace
JOIN pg_sequence sequence_catalog ON sequence_catalog.seqrelid = sequence_row.oid
LEFT JOIN pg_depend ownership
  ON ownership.classid = 'pg_class'::regclass
 AND ownership.objid = sequence_row.oid
 AND ownership.deptype = 'a'
LEFT JOIN pg_class owner_table ON owner_table.oid = ownership.refobjid
LEFT JOIN pg_attribute owner_column
  ON owner_column.attrelid = ownership.refobjid
 AND owner_column.attnum = ownership.refobjsubid
WHERE namespace_row.nspname = 'np'
ORDER BY sequence_row.relname
"""
_SELECT_CONSTRAINT_EVIDENCE_SQL = """
SELECT
    table_row.relname,
    constraint_row.conname,
    constraint_row.contype,
    constraint_row.condeferrable,
    constraint_row.condeferred,
    constraint_row.convalidated,
    pg_get_constraintdef(constraint_row.oid, TRUE)
FROM pg_constraint constraint_row
JOIN pg_class table_row ON table_row.oid = constraint_row.conrelid
JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
ORDER BY table_row.relname, constraint_row.conname
"""
_SELECT_APPLIED_MIGRATIONS_SQL = """
SELECT version, name, checksum
FROM np.schema_migrations
ORDER BY version
"""
_LOCK_AUTHORITY_TABLES_SQL = (
    "LOCK TABLE "
    + ", ".join(f"ONLY np.{table}" for table in _AUTHORITY_TABLES)
    + " IN ACCESS EXCLUSIVE MODE NOWAIT"
)
_SELECT_OLD_ROLE_SQL = """
SELECT
    role_row.rolcanlogin,
    role_row.rolsuper,
    role_row.rolinherit,
    role_row.rolcreaterole,
    role_row.rolcreatedb,
    role_row.rolreplication,
    role_row.rolbypassrls,
    role_row.rolconnlimit,
    role_row.rolpassword IS NULL
FROM pg_authid role_row
WHERE role_row.rolname = %s
"""
_SELECT_OLD_BACKENDS_SQL = """
SELECT COUNT(*)
FROM pg_stat_activity activity_row
WHERE activity_row.usename = %s
  AND activity_row.pid <> pg_backend_pid()
"""
_SELECT_OLD_MEMBERSHIPS_SQL = """
SELECT parent_role.rolname, member_role.rolname
FROM pg_auth_members membership
JOIN pg_roles parent_role ON parent_role.oid = membership.roleid
JOIN pg_roles member_role ON member_role.oid = membership.member
WHERE parent_role.rolname = %s OR member_role.rolname = %s
ORDER BY parent_role.rolname, member_role.rolname
"""


class PostgresBootstrapStatus(str, Enum):
    """Durable state reached by one reconciliation attempt."""

    CREDENTIALS_REQUIRED = "CREDENTIALS_REQUIRED"
    DEMOTION_REQUIRED = "DEMOTION_REQUIRED"
    COMPLETE = "COMPLETE"


class PostgresBootstrapPhase(str, Enum):
    """Commit boundaries used for deterministic recovery reporting."""

    ROLES = "ROLES"
    MIGRATIONS = "MIGRATIONS"
    CATALOG = "CATALOG"
    DEMOTION = "DEMOTION"


class PostgresBootstrapError(RuntimeError):
    """Base class for fail-closed bootstrap failures."""


class PostgresBootstrapInputError(PostgresBootstrapError, ValueError):
    """Raised before database access for an unsafe operator input."""


class PostgresBootstrapStorageError(PostgresBootstrapError):
    """Raised when PostgreSQL cannot be reached or queried safely."""


class PostgresBootstrapDriftError(PostgresBootstrapError):
    """Raised for an unexpected object, owner, role, membership, or ACL."""


class PostgresBootstrapMigrationError(PostgresBootstrapError):
    """Raised when the packaged migration history cannot be reconciled."""


class PostgresBootstrapCommitUnknownError(PostgresBootstrapError):
    """Raised when a failed commit cannot be resolved by catalog reread."""

    def __init__(self, phase: PostgresBootstrapPhase) -> None:
        self.phase = phase
        super().__init__(f"bootstrap {phase.value.lower()} commit outcome is unknown")


@dataclass(frozen=True, slots=True)
class PostgresBootstrapRoles:
    """Exact seven-role authority manifest."""

    schema_owner: str
    migrator: str
    legacy_runtime: str
    atomic_runtime: str
    activation: str
    readiness: str
    trainer: str

    def __post_init__(self) -> None:
        values = self.all
        if any(
            not isinstance(role, str) or _ROLE_IDENTIFIER.fullmatch(role) is None
            for role in values
        ):
            raise PostgresBootstrapInputError(
                "bootstrap role names must be lowercase PostgreSQL identifiers"
            )
        if len(set(values)) != len(values):
            raise PostgresBootstrapInputError(
                "bootstrap role names must be pairwise distinct"
            )

    @property
    def login_roles(self) -> tuple[str, ...]:
        return (
            self.migrator,
            self.legacy_runtime,
            self.atomic_runtime,
            self.activation,
            self.readiness,
            self.trainer,
        )

    @property
    def all(self) -> tuple[str, ...]:
        return (self.schema_owner,) + self.login_roles


@dataclass(frozen=True, slots=True)
class PostgresBootstrapAdoption:
    """Explicit authority inputs for an existing pre-bootstrap volume."""

    migration_authority_role: str
    allowed_historical_owner_roles: tuple[str, ...]
    old_shared_runtime_role: str | None = None
    demote_old_shared_runtime: bool = False

    def __post_init__(self) -> None:
        names = self.allowed_historical_owner_roles
        candidates = names + (self.migration_authority_role,)
        if self.old_shared_runtime_role is not None:
            candidates += (self.old_shared_runtime_role,)
        if any(
            not isinstance(role, str) or _ROLE_IDENTIFIER.fullmatch(role) is None
            for role in candidates
        ):
            raise PostgresBootstrapInputError(
                "adoption role names must be lowercase PostgreSQL identifiers"
            )
        if names != (self.migration_authority_role,):
            raise PostgresBootstrapInputError(
                "historical ownership must be bound to the migration authority"
            )
        if type(self.demote_old_shared_runtime) is not bool:
            raise PostgresBootstrapInputError(
                "demote_old_shared_runtime must be a boolean"
            )
        if self.demote_old_shared_runtime and self.old_shared_runtime_role is None:
            raise PostgresBootstrapInputError(
                "old shared runtime role is required before demotion"
            )
        if (
            self.old_shared_runtime_role is not None
            and self.old_shared_runtime_role != self.migration_authority_role
        ):
            raise PostgresBootstrapInputError(
                "old shared runtime role must be the migration authority"
            )


@dataclass(frozen=True, slots=True)
class PostgresBootstrapContext:
    """Non-secret operator intent for one database."""

    expected_database: str
    admin_role: str
    roles: PostgresBootstrapRoles
    adoption: PostgresBootstrapAdoption | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.expected_database, str)
            or not self.expected_database
            or "\x00" in self.expected_database
        ):
            raise PostgresBootstrapInputError(
                "expected_database must be non-empty text"
            )
        if (
            not isinstance(self.admin_role, str)
            or _ROLE_IDENTIFIER.fullmatch(self.admin_role) is None
        ):
            raise PostgresBootstrapInputError(
                "admin_role must be a lowercase PostgreSQL identifier"
            )
        if type(self.roles) is not PostgresBootstrapRoles:
            raise PostgresBootstrapInputError(
                "roles must be a PostgresBootstrapRoles value"
            )
        if self.admin_role in self.roles.all:
            raise PostgresBootstrapInputError(
                "admin role must be distinct from managed bootstrap roles"
            )
        if self.adoption is not None:
            if type(self.adoption) is not PostgresBootstrapAdoption:
                raise PostgresBootstrapInputError(
                    "adoption must be a PostgresBootstrapAdoption value"
                )
            conflicting = set(self.roles.all).intersection(
                self.adoption.allowed_historical_owner_roles
            )
            if conflicting:
                raise PostgresBootstrapInputError(
                    "managed roles cannot be declared as historical owners"
                )
            if self.adoption.old_shared_runtime_role in self.roles.all:
                raise PostgresBootstrapInputError(
                    "old shared runtime role must be distinct from managed roles"
                )
            if self.adoption.old_shared_runtime_role == self.admin_role:
                raise PostgresBootstrapInputError(
                    "old shared runtime must differ from the bootstrap admin identity"
                )


@dataclass(frozen=True, slots=True)
class PostgresBootstrapReceipt:
    """Typed, secret-free result of a durable reconciliation."""

    status: PostgresBootstrapStatus
    migration_versions: tuple[int, ...]
    verified_role_probes: tuple[str, ...]
    pending_role_credentials: tuple[str, ...]
    old_shared_runtime_demoted: bool


@dataclass(frozen=True, slots=True)
class PostgresBootstrapTerminalInspection:
    """Read-only evidence from one exact terminal-catalog transaction."""

    system_identifier: int
    exact: bool
    migration_versions: tuple[int, ...]
    runtime_mode: str | None
    runtime_generation: int | None
    nonempty_relations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CredentialFactories:
    migrator: Callable[[], object] | None = field(repr=False, compare=False)
    legacy_runtime: Callable[[], object] | None = field(repr=False, compare=False)
    atomic_runtime: Callable[[], object] | None = field(repr=False, compare=False)
    activation: Callable[[], object] | None = field(repr=False, compare=False)
    readiness: Callable[[], object] | None = field(repr=False, compare=False)
    trainer: Callable[[], object] | None = field(repr=False, compare=False)


def _close_quietly(connection: object) -> None:
    try:
        close = getattr(connection, "close", None)
    except Exception:
        return
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _rollback_quietly(connection: object) -> None:
    try:
        rollback = getattr(connection, "rollback", None)
    except Exception:
        return
    if callable(rollback):
        try:
            rollback()
        except Exception:
            pass


def _fresh_connection(
    factory: Callable[[], object],
    *,
    label: str,
) -> object:
    failed = False
    try:
        connection = factory()
    except Exception:
        failed = True
        connection = None
    if failed:
        raise PostgresBootstrapStorageError(
            f"could not open the {label} bootstrap connection"
        )
    required = ("cursor", "commit", "rollback", "close")
    interface_failed = False
    try:
        valid_interface = all(
            callable(getattr(connection, name, None)) for name in required
        )
        transaction_status = getattr(connection, "get_transaction_status", None)
        autocommit = getattr(connection, "autocommit", None)
    except Exception:
        interface_failed = True
        valid_interface = False
        transaction_status = None
        autocommit = None
    if interface_failed or not valid_interface:
        _close_quietly(connection)
        raise PostgresBootstrapStorageError(
            f"the {label} bootstrap connection has an invalid interface"
        )
    if not callable(transaction_status):
        _close_quietly(connection)
        raise PostgresBootstrapStorageError(
            f"the {label} bootstrap connection has no transaction status"
        )
    if autocommit is not False:
        _close_quietly(connection)
        raise PostgresBootstrapStorageError(
            f"the {label} bootstrap connection must disable autocommit"
        )
    status_failed = False
    try:
        is_idle = transaction_status() == TRANSACTION_STATUS_IDLE
    except Exception:
        status_failed = True
        is_idle = False
    if status_failed:
        _close_quietly(connection)
        raise PostgresBootstrapStorageError(
            f"the {label} bootstrap connection status could not be inspected"
        )
    status_failed = False
    try:
        connection_status = getattr(connection, "status", None)
    except Exception:
        status_failed = True
        connection_status = None
    if status_failed or not is_idle or connection_status != STATUS_READY:
        _close_quietly(connection)
        raise PostgresBootstrapStorageError(
            f"the {label} bootstrap connection must be fresh and idle"
        )
    return connection


def _one_row(value: object, length: int, label: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise PostgresBootstrapDriftError(
            f"PostgreSQL returned invalid {label} evidence"
        )
    return tuple(value)


class PostgresBootstrap:
    """Reconcile the dormant PostgreSQL authority boundary in durable phases."""

    def __init__(
        self,
        admin_connection_factory: Callable[[], object],
        *,
        migrator_connection_factory: Callable[[], object] | None = None,
        legacy_runtime_connection_factory: Callable[[], object] | None = None,
        atomic_runtime_connection_factory: Callable[[], object] | None = None,
        activation_connection_factory: Callable[[], object] | None = None,
        readiness_connection_factory: Callable[[], object] | None = None,
        trainer_connection_factory: Callable[[], object] | None = None,
    ) -> None:
        factories = (
            admin_connection_factory,
            migrator_connection_factory,
            legacy_runtime_connection_factory,
            atomic_runtime_connection_factory,
            activation_connection_factory,
            readiness_connection_factory,
            trainer_connection_factory,
        )
        if not callable(admin_connection_factory):
            raise TypeError("admin_connection_factory must be callable")
        if any(factory is not None and not callable(factory) for factory in factories):
            raise TypeError("role connection factories must be callable or None")
        self._admin_connection_factory = admin_connection_factory
        self._credential_factories = _CredentialFactories(
            migrator=migrator_connection_factory,
            legacy_runtime=legacy_runtime_connection_factory,
            atomic_runtime=atomic_runtime_connection_factory,
            activation=activation_connection_factory,
            readiness=readiness_connection_factory,
            trainer=trainer_connection_factory,
        )

    def inspect_terminal(
        self, context: PostgresBootstrapContext, /
    ) -> PostgresBootstrapTerminalInspection:
        """Inspect terminal catalog and emptiness in one read-only transaction."""
        if type(context) is not PostgresBootstrapContext:
            raise TypeError("context must be a PostgresBootstrapContext")
        connection = _fresh_connection(
            self._admin_connection_factory,
            label="terminal catalog inspection",
        )
        exact = False
        inspection_failed = False
        system_identifier = None
        migration_versions: tuple[int, ...] = ()
        runtime_mode = None
        runtime_generation = None
        nonempty_relations: tuple[str, ...] = ()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_REPEATABLE_READ_ONLY_SQL)
                    cursor.execute(_UTC_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    cursor.execute(_SELECT_CLUSTER_SYSTEM_IDENTIFIER_SQL)
                    system_identifier = _one_row(
                        cursor.fetchone(), 1, "cluster system identifier"
                    )[0]
                    if type(system_identifier) is not int or system_identifier <= 0:
                        raise PostgresBootstrapStorageError(
                            "PostgreSQL returned invalid cluster evidence"
                        )
                    cursor.execute("SELECT to_regclass('np.schema_migrations')")
                    migration_relation = _one_row(
                        cursor.fetchone(), 1, "migration relation"
                    )[0]
                    cursor.execute("SELECT to_regclass('np.paper_runtime_control')")
                    control_relation = _one_row(
                        cursor.fetchone(), 1, "runtime control relation"
                    )[0]
                    ledger_layout_exact = (
                        migration_relation is not None
                        and _migration_metadata_is_exact(cursor)
                    )
                    if ledger_layout_exact:
                        cursor.execute(
                            "SELECT version FROM np.schema_migrations "
                            "ORDER BY version"
                        )
                        migration_versions = tuple(row[0] for row in cursor.fetchall())
                    control_layout_exact = False
                    if control_relation is not None:
                        cursor.execute(
                            "SELECT column_name, udt_name FROM information_schema.columns "
                            "WHERE table_schema = 'np' "
                            "AND table_name = 'paper_runtime_control' "
                            "ORDER BY ordinal_position"
                        )
                        control_layout_exact = tuple(
                            tuple(row) for row in cursor.fetchall()
                        ) == (
                            ("control_key", "bool"),
                            ("mode", "text"),
                            ("runtime_generation", "int8"),
                            ("updated_at", "timestamptz"),
                        )
                    if control_layout_exact:
                        cursor.execute(_SELECT_TERMINAL_RUNTIME_CONTROL_SQL)
                        control_rows = tuple(tuple(row) for row in cursor.fetchall())
                        if len(control_rows) == 1 and len(control_rows[0]) == 2:
                            runtime_mode, runtime_generation = control_rows[0]

                    cursor.execute(
                        "SELECT relname FROM pg_class table_row "
                        "JOIN pg_namespace namespace_row "
                        "ON namespace_row.oid = table_row.relnamespace "
                        "WHERE namespace_row.nspname = 'np' "
                        "AND table_row.relkind = 'r' "
                        "AND table_row.relname = ANY(%s) ORDER BY relname",
                        (list(_TERMINAL_DATA_TABLES),),
                    )
                    present = {row[0] for row in cursor.fetchall()}
                    nonempty = []
                    for table in _TERMINAL_DATA_TABLES:
                        if table not in present:
                            continue
                        cursor.execute(
                            sql.SQL("SELECT EXISTS (SELECT 1 FROM np.{})").format(
                                sql.Identifier(table)
                            )
                        )
                        if (
                            _one_row(
                                cursor.fetchone(), 1, "terminal relation emptiness"
                            )[0]
                            is True
                        ):
                            nonempty.append(f"np.{table}")
                    nonempty_relations = tuple(sorted(nonempty))

                    try:
                        self._require_admin_identity(cursor, context)
                        managed_roles_exact = self._managed_roles_are_exact(
                            cursor,
                            context,
                            allow_absent=False,
                        )
                    except PostgresBootstrapDriftError:
                        exact = False
                    else:
                        if (
                            not managed_roles_exact
                            or not ledger_layout_exact
                            or not control_layout_exact
                        ):
                            exact = False
                        elif not self._migration_history_is_exact(cursor):
                            exact = False
                        elif not self._catalog_shape_is_expected(
                            cursor,
                            context,
                            allow_historical_owners=False,
                        ):
                            exact = False
                        elif not _activation_catalog_is_authoritative(cursor):
                            exact = False
                        else:
                            cursor.execute(
                                _SELECT_DATABASE_AUTHORITY_SQL,
                                (context.roles.schema_owner,),
                            )
                            database = _one_row(
                                cursor.fetchone(), 3, "database authority"
                            )
                            exact = database == (context.admin_role, True, False)
            except PostgresBootstrapDriftError, PostgresBootstrapMigrationError:
                exact = False
            except Exception:
                inspection_failed = True
        finally:
            _rollback_quietly(connection)
            _close_quietly(connection)
        if inspection_failed:
            raise PostgresBootstrapStorageError(
                "terminal PostgreSQL catalog inspection failed"
            )
        return PostgresBootstrapTerminalInspection(
            system_identifier=system_identifier,
            exact=exact,
            migration_versions=migration_versions,
            runtime_mode=runtime_mode,
            runtime_generation=runtime_generation,
            nonempty_relations=nonempty_relations,
        )

    def reconcile(
        self,
        context: PostgresBootstrapContext,
        /,
    ) -> PostgresBootstrapReceipt:
        """Reconcile roles, migrations, ownership, ACLs, probes, and demotion."""
        if type(context) is not PostgresBootstrapContext:
            raise TypeError("context must be a PostgresBootstrapContext")
        self._preflight_database(context)
        self._reconcile_roles(context)
        verified, pending = self._probe_credentials(context)
        if pending:
            return PostgresBootstrapReceipt(
                status=PostgresBootstrapStatus.CREDENTIALS_REQUIRED,
                migration_versions=(),
                verified_role_probes=verified,
                pending_role_credentials=pending,
                old_shared_runtime_demoted=False,
            )
        adoption = context.adoption
        if self._catalog_readback_is_exact(context):
            return PostgresBootstrapReceipt(
                status=PostgresBootstrapStatus.COMPLETE,
                migration_versions=tuple(
                    row[0] for row in self._expected_migration_rows()
                ),
                verified_role_probes=verified,
                pending_role_credentials=(),
                old_shared_runtime_demoted=bool(
                    adoption is not None
                    and adoption.old_shared_runtime_role is not None
                ),
            )
        self._require_managed_roles_exact(context)
        expected_versions = self._reconcile_migrations(context)
        if (
            adoption is not None
            and adoption.old_shared_runtime_role is not None
            and not adoption.demote_old_shared_runtime
        ):
            self._preflight_old_login_demotion(context)
            return PostgresBootstrapReceipt(
                status=PostgresBootstrapStatus.DEMOTION_REQUIRED,
                migration_versions=expected_versions,
                verified_role_probes=verified,
                pending_role_credentials=(),
                old_shared_runtime_demoted=False,
            )
        if adoption is not None and adoption.old_shared_runtime_role is not None:
            old_login_disabled = self._old_login_is_disabled(context)
            if old_login_disabled is None:
                raise PostgresBootstrapStorageError(
                    "old shared runtime login state could not be read"
                )
            if not old_login_disabled:
                self._preflight_old_login_demotion(context)
            if not old_login_disabled and self._disable_old_login(context):
                return PostgresBootstrapReceipt(
                    status=PostgresBootstrapStatus.DEMOTION_REQUIRED,
                    migration_versions=expected_versions,
                    verified_role_probes=verified,
                    pending_role_credentials=(),
                    old_shared_runtime_demoted=False,
                )
            if not self._old_backends_are_drained(context):
                return PostgresBootstrapReceipt(
                    status=PostgresBootstrapStatus.DEMOTION_REQUIRED,
                    migration_versions=expected_versions,
                    verified_role_probes=verified,
                    pending_role_credentials=(),
                    old_shared_runtime_demoted=False,
                )
            self._preflight_old_login_demotion(context)
        demoted = self._reconcile_catalog(context)
        return PostgresBootstrapReceipt(
            status=PostgresBootstrapStatus.COMPLETE,
            migration_versions=expected_versions,
            verified_role_probes=verified,
            pending_role_credentials=(),
            old_shared_runtime_demoted=demoted,
        )

    def _preflight_database(self, context: PostgresBootstrapContext) -> None:
        connection = _fresh_connection(self._admin_connection_factory, label="admin")
        preflight_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_COMMITTED_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    self._require_admin_identity(cursor, context)
                    cursor.execute(_SELECT_DATABASE_OWNER_SQL)
                    database_owner = _one_row(cursor.fetchone(), 1, "database owner")[0]
                    expected_owners = {context.admin_role}
                    if context.adoption is not None:
                        expected_owners.add(context.adoption.migration_authority_role)
                    if database_owner not in expected_owners:
                        raise PostgresBootstrapDriftError(
                            "target database has an unexpected owner"
                        )
                connection.rollback()
            except PostgresBootstrapDriftError:
                _rollback_quietly(connection)
                raise
            except Exception:
                _rollback_quietly(connection)
                preflight_failed = True
        finally:
            _close_quietly(connection)
        if preflight_failed:
            raise PostgresBootstrapStorageError(
                "target database authority preflight failed"
            )

    def _disable_old_login(self, context: PostgresBootstrapContext) -> bool:
        adoption = context.adoption
        if adoption is None or adoption.old_shared_runtime_role is None:
            return False
        old_role = adoption.old_shared_runtime_role
        connection = _fresh_connection(self._admin_connection_factory, label="admin")
        mutated = False
        commit_failed = False
        disable_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_COMMITTED_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_BOOTSTRAP_ADVISORY_LOCK_ID,),
                    )
                    self._require_admin_identity(cursor, context)
                    cursor.execute(
                        _SELECT_OLD_MEMBERSHIPS_SQL,
                        (old_role, old_role),
                    )
                    if cursor.fetchall():
                        raise PostgresBootstrapDriftError(
                            "old shared runtime role must have no memberships"
                        )
                    cursor.execute(_SELECT_OLD_ROLE_SQL, (old_role,))
                    old_state = _one_row(cursor.fetchone(), 9, "old shared runtime")
                    exact_disabled_state = (
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        -1,
                        True,
                    )
                    if old_state == exact_disabled_state:
                        connection.rollback()
                        return False
                    cursor.execute(
                        sql.SQL(
                            "ALTER ROLE {} NOLOGIN NOSUPERUSER NOINHERIT "
                            "NOCREATEDB NOCREATEROLE NOREPLICATION "
                            "NOBYPASSRLS CONNECTION LIMIT -1 PASSWORD NULL"
                        ).format(sql.Identifier(old_role))
                    )
                    mutated = True
            except PostgresBootstrapDriftError:
                _rollback_quietly(connection)
                raise
            except Exception:
                _rollback_quietly(connection)
                disable_failed = True
            if disable_failed:
                raise PostgresBootstrapStorageError(
                    "old shared runtime login could not be disabled"
                )
            try:
                connection.commit()
            except Exception:
                commit_failed = True
            if commit_failed:
                _rollback_quietly(connection)
                readback = self._old_login_is_disabled(context)
                if readback is True:
                    return True
                raise PostgresBootstrapCommitUnknownError(
                    PostgresBootstrapPhase.DEMOTION
                )
            return mutated
        finally:
            _close_quietly(connection)

    def _preflight_old_login_demotion(self, context: PostgresBootstrapContext) -> None:
        adoption = context.adoption
        if adoption is None or adoption.old_shared_runtime_role is None:
            return
        connection = _fresh_connection(self._admin_connection_factory, label="admin")
        preflight_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_COMMITTED_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    cursor.execute("SET LOCAL lock_timeout = '1s'")
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_BOOTSTRAP_ADVISORY_LOCK_ID,),
                    )
                    self._require_admin_identity(cursor, context)
                    self._managed_roles_are_exact(
                        cursor,
                        context,
                        allow_absent=False,
                    )
                    cursor.execute(_LOCK_AUTHORITY_TABLES_SQL)
                    if not self._migration_history_is_exact(cursor):
                        raise PostgresBootstrapMigrationError(
                            "migration history changed before login demotion"
                        )
                    if not self._catalog_shape_is_expected(
                        cursor,
                        context,
                        allow_historical_owners=True,
                    ):
                        raise PostgresBootstrapDriftError(
                            "PostgreSQL catalog is not eligible for demotion"
                        )
                connection.rollback()
            except PostgresBootstrapDriftError, PostgresBootstrapMigrationError:
                _rollback_quietly(connection)
                raise
            except Exception:
                _rollback_quietly(connection)
                preflight_failed = True
        finally:
            _close_quietly(connection)
        if preflight_failed:
            raise PostgresBootstrapStorageError(
                "old shared runtime demotion preflight failed"
            )

    def _old_login_is_disabled(self, context: PostgresBootstrapContext) -> bool | None:
        adoption = context.adoption
        if adoption is None or adoption.old_shared_runtime_role is None:
            return True
        try:
            connection = _fresh_connection(
                self._admin_connection_factory, label="demotion readback"
            )
        except PostgresBootstrapStorageError:
            return None
        try:
            with connection.cursor() as cursor:
                cursor.execute(_READ_COMMITTED_SQL)
                cursor.execute(_SAFE_SEARCH_PATH_SQL)
                self._require_admin_identity(cursor, context)
                cursor.execute(
                    _SELECT_OLD_ROLE_SQL, (adoption.old_shared_runtime_role,)
                )
                row = _one_row(cursor.fetchone(), 9, "old shared runtime")
                cursor.execute(
                    _SELECT_OLD_MEMBERSHIPS_SQL,
                    (
                        adoption.old_shared_runtime_role,
                        adoption.old_shared_runtime_role,
                    ),
                )
                memberships = tuple(cursor.fetchall())
            connection.rollback()
            return (
                row
                == (
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    -1,
                    True,
                )
                and not memberships
            )
        except Exception:
            _rollback_quietly(connection)
            return None
        finally:
            _close_quietly(connection)

    def _old_backends_are_drained(self, context: PostgresBootstrapContext) -> bool:
        adoption = context.adoption
        if adoption is None or adoption.old_shared_runtime_role is None:
            return True
        connection = _fresh_connection(
            self._admin_connection_factory, label="admin drain check"
        )
        inspection_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_COMMITTED_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    self._require_admin_identity(cursor, context)
                    cursor.execute(
                        _SELECT_OLD_BACKENDS_SQL,
                        (adoption.old_shared_runtime_role,),
                    )
                    count = _one_row(cursor.fetchone(), 1, "old runtime backend count")[
                        0
                    ]
                connection.rollback()
            except PostgresBootstrapDriftError:
                _rollback_quietly(connection)
                raise
            except Exception:
                _rollback_quietly(connection)
                inspection_failed = True
            if inspection_failed:
                raise PostgresBootstrapStorageError(
                    "old shared runtime backends could not be inspected"
                )
            return type(count) is int and count == 0
        finally:
            _close_quietly(connection)

    @staticmethod
    def _role_marker(context: PostgresBootstrapContext, purpose: str) -> str:
        return f"{_ROLE_MARKER_PREFIX}{context.expected_database}:" f"{purpose}"

    @staticmethod
    def _schema_marker(context: PostgresBootstrapContext) -> str:
        return f"{_SCHEMA_MARKER_PREFIX}{context.expected_database}"

    @staticmethod
    def _role_manifest(
        context: PostgresBootstrapContext,
    ) -> tuple[tuple[str, str], ...]:
        return tuple(zip(_ROLE_PURPOSES, context.roles.all))

    @staticmethod
    def _require_admin_identity(
        cursor: object,
        context: PostgresBootstrapContext,
    ) -> None:
        cursor.execute(_SELECT_ADMIN_IDENTITY_SQL)
        row = _one_row(cursor.fetchone(), 6, "admin identity")
        if row[:4] != (
            context.expected_database,
            context.admin_role,
            context.admin_role,
            context.admin_role,
        ):
            raise PostgresBootstrapDriftError(
                "bootstrap admin identity or database does not match the context"
            )
        if row[4:] != (True, True):
            raise PostgresBootstrapDriftError(
                "bootstrap admin must be an independently authenticated login superuser"
            )

    @classmethod
    def _managed_roles_are_exact(
        cls,
        cursor: object,
        context: PostgresBootstrapContext,
        *,
        allow_absent: bool,
        allow_staged_no_login: bool = False,
    ) -> bool:
        manifest = cls._role_manifest(context)
        role_names = [role for _, role in manifest]
        cursor.execute(_SELECT_MANAGED_ROLES_SQL, (role_names,))
        rows = tuple(tuple(row) for row in cursor.fetchall())
        if not rows and allow_absent:
            return False
        if len(rows) != len(manifest):
            raise PostgresBootstrapDriftError(
                "managed bootstrap roles are only partially present"
            )
        expected_by_role = {role: purpose for purpose, role in manifest}
        for row in rows:
            evidence = _one_row(row, 11, "managed role")
            role = evidence[0]
            purpose = expected_by_role.get(role)
            if purpose is None:
                raise PostgresBootstrapDriftError(
                    "PostgreSQL returned an unexpected managed role"
                )
            expected_attributes = _EXPECTED_ROLE_ATTRIBUTES[purpose]
            staged_attributes = (False,) + expected_attributes[1:]
            if evidence[1:9] == expected_attributes:
                pass
            elif (
                allow_staged_no_login
                and purpose != "schema_owner"
                and evidence[1:9] == staged_attributes
            ):
                pass
            else:
                raise PostgresBootstrapDriftError(
                    f"managed bootstrap role {role} has unsafe attributes"
                )
            if evidence[9] is not None:
                raise PostgresBootstrapDriftError(
                    f"managed bootstrap role {role} has role-level settings"
                )
            if evidence[10] != cls._role_marker(context, purpose):
                raise PostgresBootstrapDriftError(
                    f"managed bootstrap role {role} has an invalid marker"
                )

        cursor.execute(
            _SELECT_MANAGED_MEMBERSHIPS_SQL,
            (role_names, role_names),
        )
        memberships = tuple(tuple(row) for row in cursor.fetchall())
        expected_memberships = (
            (context.roles.schema_owner, context.roles.migrator, False),
        )
        if memberships != expected_memberships:
            raise PostgresBootstrapDriftError(
                "managed bootstrap role memberships have drifted"
            )
        cursor.execute(
            _SELECT_MANAGED_DATABASE_SETTINGS_SQL,
            (role_names, context.expected_database),
        )
        if cursor.fetchall():
            raise PostgresBootstrapDriftError(
                "managed bootstrap role or database settings have drifted"
            )
        login_states = {
            row[1] for row in rows if expected_by_role[row[0]] != "schema_owner"
        }
        if len(login_states) != 1:
            raise PostgresBootstrapDriftError(
                "managed bootstrap login roles are only partially provisioned"
            )
        login_enabled = next(iter(login_states))
        cursor.execute(_SELECT_MANAGED_PASSWORD_STATES_SQL, (role_names,))
        password_states = tuple(tuple(row) for row in cursor.fetchall())
        expected_password_states = tuple(
            sorted(
                (
                    role,
                    purpose == "schema_owner" or not login_enabled,
                    True,
                )
                for purpose, role in manifest
            )
        )
        if password_states != expected_password_states:
            raise PostgresBootstrapDriftError(
                "managed bootstrap role credentials are absent, expired, or unsafe"
            )
        return True

    @classmethod
    def _create_managed_roles(
        cls,
        cursor: object,
        context: PostgresBootstrapContext,
    ) -> None:
        for purpose, role in cls._role_manifest(context):
            cursor.execute(
                sql.SQL(
                    "CREATE ROLE {} NOLOGIN NOSUPERUSER NOINHERIT NOCREATEDB "
                    "NOCREATEROLE NOREPLICATION NOBYPASSRLS "
                    "CONNECTION LIMIT -1 PASSWORD NULL"
                ).format(sql.Identifier(role))
            )
            cursor.execute(
                sql.SQL("COMMENT ON ROLE {} IS %s").format(sql.Identifier(role)),
                (cls._role_marker(context, purpose),),
            )
        cursor.execute(
            sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(context.roles.schema_owner),
                sql.Identifier(context.roles.migrator),
            )
        )

    @classmethod
    def _database_catalog_is_admissible(
        cls,
        cursor: object,
        context: PostgresBootstrapContext,
    ) -> bool:
        relevant_roles = [context.admin_role, *context.roles.all]
        if context.adoption is not None:
            relevant_roles.extend(context.adoption.allowed_historical_owner_roles)

        cursor.execute(_SELECT_EXTENSION_EVIDENCE_SQL)
        extension_rows = tuple(tuple(row) for row in cursor.fetchall())
        if extension_rows != (
            (
                "plpgsql",
                "1.0",
                False,
                "pg_catalog",
                context.admin_role,
                True,
                True,
            ),
        ):
            return False

        cursor.execute(_SELECT_LANGUAGE_EVIDENCE_SQL)
        language_rows = tuple(tuple(row) for row in cursor.fetchall())
        if any(len(row) != 8 or row[1] != context.admin_role for row in language_rows):
            return False
        language_without_owner = tuple((row[0], *row[2:]) for row in language_rows)
        if language_without_owner != _EXPECTED_LANGUAGE_EVIDENCE_WITHOUT_OWNER:
            return False

        cursor.execute(_SELECT_HANDLER_PROCEDURE_EVIDENCE_SQL)
        handler_rows = tuple(tuple(row) for row in cursor.fetchall())
        if any(len(row) != 29 or row[4] != context.admin_role for row in handler_rows):
            return False
        handler_rows_without_owner = tuple((*row[:4], *row[5:]) for row in handler_rows)
        if handler_rows_without_owner != (
            _EXPECTED_HANDLER_PROCEDURE_EVIDENCE_WITHOUT_OWNER
        ):
            return False

        cursor.execute(_SELECT_ACCESS_METHOD_EVIDENCE_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            _EXPECTED_ACCESS_METHOD_EVIDENCE
        ):
            return False
        cursor.execute(_SELECT_PLPGSQL_DEPENDENCY_EVIDENCE_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            _EXPECTED_PLPGSQL_DEPENDENCY_EVIDENCE
        ):
            return False
        cursor.execute(_SELECT_UNEXPECTED_PG_CATALOG_OBJECTS_SQL)
        if cursor.fetchall():
            return False
        cursor.execute(_SELECT_UNEXPECTED_DATABASE_OBJECTS_SQL)
        if cursor.fetchall():
            return False
        cursor.execute(
            _SELECT_RELEVANT_DATABASE_SETTINGS_SQL,
            (relevant_roles,),
        )
        if cursor.fetchall():
            return False
        cursor.execute(
            _SELECT_RELEVANT_SHARED_SECURITY_LABELS_SQL,
            (relevant_roles,),
        )
        if cursor.fetchall():
            return False
        cursor.execute(
            _SELECT_RELEVANT_PARAMETER_ACLS_SQL,
            (relevant_roles,),
        )
        return not cursor.fetchall()

    @classmethod
    def _require_pre_role_catalog_admissible(
        cls,
        cursor: object,
        context: PostgresBootstrapContext,
    ) -> None:
        """Reject an invalid target before creating cluster-global roles."""
        inspection_failed = False
        try:
            cls._inspect_pre_role_catalog(cursor, context)
        except PostgresBootstrapDriftError, PostgresBootstrapMigrationError:
            raise
        except Exception:
            inspection_failed = True
        if inspection_failed:
            raise PostgresBootstrapStorageError(
                "pre-role PostgreSQL catalog admission failed"
            )

    @classmethod
    def _inspect_pre_role_catalog(
        cls,
        cursor: object,
        context: PostgresBootstrapContext,
    ) -> None:
        if not cls._database_catalog_is_admissible(cursor, context):
            raise PostgresBootstrapDriftError(
                "target database catalog is not admissible before role "
                "reconciliation"
            )
        cursor.execute(_SELECT_PUBLIC_SCHEMA_ACL_SQL)
        if tuple(tuple(row) for row in cursor.fetchall()) != (
            ("pg_database_owner", "PUBLIC", "USAGE", False),
            ("pg_database_owner", "pg_database_owner", "CREATE", False),
            ("pg_database_owner", "pg_database_owner", "USAGE", False),
        ):
            raise PostgresBootstrapDriftError(
                "public schema authority is not admissible before role reconciliation"
            )
        cursor.execute(_SELECT_UNEXPECTED_USER_SCHEMAS_SQL)
        if cursor.fetchall():
            raise PostgresBootstrapDriftError(
                "target database has an unexpected user schema"
            )
        cursor.execute(_SELECT_UNEXPECTED_PUBLIC_OBJECTS_SQL)
        if cursor.fetchall():
            raise PostgresBootstrapDriftError(
                "target database has an unexpected public object"
            )
        cursor.execute(_SELECT_LARGE_OBJECT_COUNT_SQL)
        large_object_count = _one_row(cursor.fetchone(), 1, "large object inventory")[0]
        if type(large_object_count) is not int or large_object_count != 0:
            raise PostgresBootstrapDriftError(
                "target database has an unexpected large object"
            )
        cursor.execute(_SELECT_SCHEMA_AUTHORITY_SQL)
        schema_rows = tuple(tuple(row) for row in cursor.fetchall())
        adoption = context.adoption
        if adoption is None:
            if not schema_rows:
                cursor.execute(_SELECT_DATABASE_ACL_GRANTEES_SQL)
                database_acl = tuple(tuple(row) for row in cursor.fetchall())
                if database_acl != tuple(
                    sorted(
                        (
                            (context.admin_role, "PUBLIC", "CONNECT", False),
                            (
                                context.admin_role,
                                "PUBLIC",
                                "TEMPORARY",
                                False,
                            ),
                        )
                    )
                ):
                    raise PostgresBootstrapDriftError(
                        "fresh database authority is not admissible before role "
                        "reconciliation"
                    )
                return
            if schema_rows != (
                (context.roles.schema_owner, cls._schema_marker(context)),
            ):
                raise PostgresBootstrapDriftError(
                    "fresh bootstrap catalog has an unexpected np schema owner"
                )
            cursor.execute(_SELECT_SCHEMA_OBJECTS_SQL)
            existing_objects = tuple(cursor.fetchall())
            if not existing_objects:
                cursor.execute(_SELECT_SCHEMA_FUNCTIONS_SQL)
                if cursor.fetchall():
                    raise PostgresBootstrapDriftError(
                        "prepared fresh schema contains unexpected routines"
                    )
                cursor.execute(
                    _SELECT_UNEXPECTED_NP_CATALOG_OBJECTS_SQL,
                    (list(_AUTHORITY_TABLES),),
                )
                if cursor.fetchall():
                    raise PostgresBootstrapDriftError(
                        "prepared fresh schema contains unexpected catalog objects"
                    )
                cursor.execute(_SELECT_SCHEMA_ACL_GRANTEES_SQL)
                if cursor.fetchall():
                    raise PostgresBootstrapDriftError(
                        "prepared fresh schema contains unexpected authority"
                    )
                cursor.execute(_SELECT_DATABASE_ACL_GRANTEES_SQL)
                database_acl = tuple(tuple(row) for row in cursor.fetchall())
                expected_acl = tuple(
                    sorted(
                        (
                            (context.admin_role, "PUBLIC", "CONNECT", False),
                            (
                                context.admin_role,
                                "PUBLIC",
                                "TEMPORARY",
                                False,
                            ),
                            (
                                context.admin_role,
                                context.roles.schema_owner,
                                "CREATE",
                                False,
                            ),
                        )
                    )
                )
                if database_acl != expected_acl:
                    raise PostgresBootstrapDriftError(
                        "fresh migration authority is not admissible before role "
                        "reconciliation"
                    )
                return
            if not cls._pre_role_migration_history_is_exact(cursor):
                raise PostgresBootstrapMigrationError(
                    "fresh PostgreSQL migration history is not exact"
                )
        else:
            if not cls._pre_role_migration_history_is_exact(cursor):
                raise PostgresBootstrapMigrationError(
                    "existing PostgreSQL migration history is not exact"
                )
        if not cls._catalog_shape_is_expected(
            cursor,
            context,
            allow_historical_owners=True,
        ):
            raise PostgresBootstrapDriftError(
                "PostgreSQL catalog is not admissible before role reconciliation"
            )
        if not _activation_catalog_is_authoritative(cursor):
            raise PostgresBootstrapDriftError(
                "PostgreSQL catalog authority is not admissible before role "
                "reconciliation"
            )

    def _reconcile_roles(self, context: PostgresBootstrapContext) -> None:
        connection = _fresh_connection(
            self._admin_connection_factory,
            label="admin",
        )
        mutated = False
        reconciliation_failed = False
        rollback_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_COMMITTED_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_BOOTSTRAP_ADVISORY_LOCK_ID,),
                    )
                    self._require_admin_identity(cursor, context)
                    exact = self._managed_roles_are_exact(
                        cursor,
                        context,
                        allow_absent=True,
                        allow_staged_no_login=True,
                    )
                    self._require_pre_role_catalog_admissible(cursor, context)
                    if not exact:
                        self._create_managed_roles(cursor, context)
                        self._managed_roles_are_exact(
                            cursor,
                            context,
                            allow_absent=False,
                            allow_staged_no_login=True,
                        )
                        mutated = True
            except PostgresBootstrapDriftError, PostgresBootstrapMigrationError:
                _rollback_quietly(connection)
                raise
            except Exception:
                _rollback_quietly(connection)
                reconciliation_failed = True

            if reconciliation_failed:
                raise PostgresBootstrapStorageError(
                    "managed bootstrap role reconciliation failed"
                )

            if not mutated:
                try:
                    connection.rollback()
                except Exception:
                    rollback_failed = True
                if rollback_failed:
                    raise PostgresBootstrapStorageError(
                        "managed bootstrap role verification could not roll back"
                    )
                return
            commit_failed = False
            try:
                connection.commit()
            except Exception:
                commit_failed = True
            if commit_failed:
                _rollback_quietly(connection)
                if not self._roles_are_exact_after_commit(context):
                    raise PostgresBootstrapCommitUnknownError(
                        PostgresBootstrapPhase.ROLES
                    )
        finally:
            _close_quietly(connection)

    def _roles_are_exact_after_commit(
        self,
        context: PostgresBootstrapContext,
    ) -> bool:
        try:
            connection = _fresh_connection(
                self._admin_connection_factory,
                label="admin readback",
            )
        except PostgresBootstrapStorageError:
            return False
        try:
            with connection.cursor() as cursor:
                cursor.execute(_READ_COMMITTED_SQL)
                cursor.execute(_SAFE_SEARCH_PATH_SQL)
                self._require_admin_identity(cursor, context)
                exact = self._managed_roles_are_exact(
                    cursor,
                    context,
                    allow_absent=False,
                    allow_staged_no_login=True,
                )
            connection.rollback()
            return exact
        except Exception:
            _rollback_quietly(connection)
            return False
        finally:
            _close_quietly(connection)

    def _require_managed_roles_exact(
        self,
        context: PostgresBootstrapContext,
    ) -> None:
        connection = _fresh_connection(
            self._admin_connection_factory,
            label="admin role recheck",
        )
        recheck_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_COMMITTED_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    self._require_admin_identity(cursor, context)
                    self._managed_roles_are_exact(
                        cursor,
                        context,
                        allow_absent=False,
                    )
                connection.rollback()
            except PostgresBootstrapDriftError:
                _rollback_quietly(connection)
                raise
            except Exception:
                _rollback_quietly(connection)
                recheck_failed = True
        finally:
            _close_quietly(connection)
        if recheck_failed:
            raise PostgresBootstrapStorageError(
                "managed bootstrap roles could not be rechecked"
            )

    def _probe_credentials(
        self,
        context: PostgresBootstrapContext,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        verified = []
        pending = []
        factories = (
            self._credential_factories.migrator,
            self._credential_factories.legacy_runtime,
            self._credential_factories.atomic_runtime,
            self._credential_factories.activation,
            self._credential_factories.readiness,
            self._credential_factories.trainer,
        )
        manifest = self._role_manifest(context)[1:]
        for (purpose, role), factory in zip(manifest, factories):
            if factory is None:
                pending.append(role)
                continue
            self._probe_credential(context, purpose, role, factory)
            verified.append(role)
        return tuple(verified), tuple(pending)

    @classmethod
    def _probe_credential(
        cls,
        context: PostgresBootstrapContext,
        purpose: str,
        role: str,
        factory: Callable[[], object],
    ) -> None:
        connection = _fresh_connection(factory, label=purpose)
        probe_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_COMMITTED_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    cursor.execute(_SELECT_CREDENTIAL_IDENTITY_SQL)
                    row = _one_row(
                        cursor.fetchone(),
                        14,
                        f"{purpose} credential identity",
                    )
                    if row[:4] != (context.expected_database, role, role, role):
                        raise PostgresBootstrapDriftError(
                            f"{purpose} credential authenticates as another identity"
                        )
                    if row[4:12] != _EXPECTED_ROLE_ATTRIBUTES[purpose]:
                        raise PostgresBootstrapDriftError(
                            f"{purpose} credential role attributes have drifted"
                        )
                    if row[12] is not None:
                        raise PostgresBootstrapDriftError(
                            f"{purpose} credential has role-level settings"
                        )
                    if row[13] != cls._role_marker(context, purpose):
                        raise PostgresBootstrapDriftError(
                            f"{purpose} credential role marker has drifted"
                        )
                connection.rollback()
            except PostgresBootstrapDriftError:
                _rollback_quietly(connection)
                raise
            except Exception:
                _rollback_quietly(connection)
                probe_failed = True
        finally:
            _close_quietly(connection)
        if probe_failed:
            raise PostgresBootstrapStorageError(f"{purpose} credential probe failed")

    @staticmethod
    def _expected_migration_rows() -> tuple[tuple[int, str, str], ...]:
        return tuple(
            (migration.version, migration.name, migration.checksum)
            for migration in load_migrations()
        )

    @classmethod
    def _migration_history_is_exact(cls, cursor: object) -> bool:
        try:
            cursor.execute(_SELECT_APPLIED_MIGRATIONS_SQL)
            rows = tuple(tuple(row) for row in cursor.fetchall())
        except Exception:
            return False
        return rows == cls._expected_migration_rows()

    @classmethod
    def _pre_role_migration_history_is_exact(cls, cursor: object) -> bool:
        try:
            cursor.execute(_SELECT_APPLIED_MIGRATIONS_SQL)
            rows = tuple(tuple(row) for row in cursor.fetchall())
        except psycopg2.Error as exc:
            if getattr(exc, "pgcode", None) == "42P01":
                return False
            raise
        return rows == cls._expected_migration_rows()

    def _prepare_fresh_migration_authority(
        self,
        context: PostgresBootstrapContext,
    ) -> None:
        connection = _fresh_connection(
            self._admin_connection_factory,
            label="admin",
        )
        preparation_failed = False
        commit_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_COMMITTED_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_BOOTSTRAP_ADVISORY_LOCK_ID,),
                    )
                    self._require_admin_identity(cursor, context)
                    cursor.execute(_SELECT_SCHEMA_AUTHORITY_SQL)
                    schema_rows = tuple(tuple(row) for row in cursor.fetchall())
                    if schema_rows:
                        if schema_rows != (
                            (
                                context.roles.schema_owner,
                                self._schema_marker(context),
                            ),
                        ):
                            raise PostgresBootstrapDriftError(
                                "fresh bootstrap catalog has an unexpected np schema owner"
                            )
                        cursor.execute(_SELECT_SCHEMA_OBJECTS_SQL)
                        existing_objects = tuple(cursor.fetchall())
                        if existing_objects and not self._migration_history_is_exact(
                            cursor
                        ):
                            raise PostgresBootstrapDriftError(
                                "fresh bootstrap schema is only partially migrated"
                            )
                    cursor.execute(
                        sql.SQL("REVOKE CREATE ON DATABASE {} FROM PUBLIC").format(
                            sql.Identifier(context.expected_database)
                        )
                    )
                    cursor.execute(
                        sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                            sql.Identifier(context.expected_database),
                            sql.Identifier(context.roles.schema_owner),
                        )
                    )
                    if not schema_rows:
                        cursor.execute(
                            sql.SQL("CREATE SCHEMA np AUTHORIZATION {}").format(
                                sql.Identifier(context.roles.schema_owner)
                            )
                        )
                        cursor.execute(
                            "COMMENT ON SCHEMA np IS %s",
                            (self._schema_marker(context),),
                        )
            except PostgresBootstrapDriftError:
                _rollback_quietly(connection)
                raise
            except Exception:
                _rollback_quietly(connection)
                preparation_failed = True
            if preparation_failed:
                raise PostgresBootstrapMigrationError(
                    "fresh migration authority preparation failed"
                )
            try:
                connection.commit()
            except Exception:
                commit_failed = True
            if commit_failed:
                _rollback_quietly(connection)
                if self._fresh_migration_authority_is_exact(context):
                    return
                raise PostgresBootstrapCommitUnknownError(
                    PostgresBootstrapPhase.MIGRATIONS
                )
        finally:
            _close_quietly(connection)

    def _fresh_migration_authority_is_exact(
        self,
        context: PostgresBootstrapContext,
    ) -> bool:
        try:
            connection = _fresh_connection(
                self._admin_connection_factory,
                label="admin readback",
            )
        except PostgresBootstrapStorageError:
            return False
        try:
            with connection.cursor() as cursor:
                cursor.execute(_READ_COMMITTED_SQL)
                cursor.execute(_SAFE_SEARCH_PATH_SQL)
                self._require_admin_identity(cursor, context)
                cursor.execute(
                    _SELECT_DATABASE_AUTHORITY_SQL, (context.roles.schema_owner,)
                )
                database = _one_row(cursor.fetchone(), 3, "database authority")
                cursor.execute(_SELECT_DATABASE_ACL_GRANTEES_SQL)
                database_acl = tuple(tuple(row) for row in cursor.fetchall())
                cursor.execute(_SELECT_SCHEMA_AUTHORITY_SQL)
                schema = _one_row(cursor.fetchone(), 2, "schema authority")
            connection.rollback()
            fresh_acl = tuple(
                sorted(
                    (
                        (context.admin_role, "PUBLIC", "CONNECT", False),
                        (context.admin_role, "PUBLIC", "TEMPORARY", False),
                        (
                            context.admin_role,
                            context.roles.schema_owner,
                            "CREATE",
                            False,
                        ),
                    )
                )
            )
            return (
                database == (context.admin_role, True, False)
                and database_acl
                in (fresh_acl, self._expected_database_acl_rows(context))
                and schema == (context.roles.schema_owner, self._schema_marker(context))
            )
        except Exception:
            _rollback_quietly(connection)
            return False
        finally:
            _close_quietly(connection)

    @staticmethod
    def _set_migration_role(connection: object, role: str) -> None:
        role_failed = False
        try:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute("SET search_path = pg_catalog")
                cursor.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
            connection.autocommit = False
        except Exception:
            role_failed = True
        if role_failed:
            _close_quietly(connection)
            raise PostgresBootstrapMigrationError(
                "migrator could not assume the schema owner role"
            )

    @classmethod
    def _require_migrator_connection_identity(
        cls,
        connection: object,
        context: PostgresBootstrapContext,
    ) -> None:
        identity_failed = False
        try:
            with connection.cursor() as cursor:
                cursor.execute(_READ_COMMITTED_SQL)
                cursor.execute(_SAFE_SEARCH_PATH_SQL)
                cursor.execute(_SELECT_CREDENTIAL_IDENTITY_SQL)
                row = _one_row(cursor.fetchone(), 14, "migrator connection identity")
                role = context.roles.migrator
                if row[:4] != (context.expected_database, role, role, role):
                    raise PostgresBootstrapDriftError(
                        "migrator factory returned another authenticated identity"
                    )
                if row[4:12] != _EXPECTED_ROLE_ATTRIBUTES["migrator"]:
                    raise PostgresBootstrapDriftError(
                        "migrator connection role attributes have drifted"
                    )
                if row[12] is not None or row[13] != cls._role_marker(
                    context, "migrator"
                ):
                    raise PostgresBootstrapDriftError(
                        "migrator connection authority has drifted"
                    )
            connection.rollback()
        except PostgresBootstrapDriftError:
            _rollback_quietly(connection)
            raise
        except Exception:
            _rollback_quietly(connection)
            identity_failed = True
        if identity_failed:
            raise PostgresBootstrapMigrationError(
                "migrator connection identity could not be verified"
            )

    def _migration_ledger_readback(
        self,
        context: PostgresBootstrapContext,
    ) -> bool | None:
        try:
            connection = _fresh_connection(
                self._admin_connection_factory,
                label="migration readback",
            )
        except PostgresBootstrapStorageError:
            return None
        try:
            with connection.cursor() as cursor:
                cursor.execute(_READ_COMMITTED_SQL)
                cursor.execute(_SAFE_SEARCH_PATH_SQL)
                self._require_admin_identity(cursor, context)
                try:
                    cursor.execute(_SELECT_APPLIED_MIGRATIONS_SQL)
                    rows = tuple(tuple(row) for row in cursor.fetchall())
                except psycopg2.Error as exc:
                    if getattr(exc, "pgcode", None) == "42P01":
                        _rollback_quietly(connection)
                        return False
                    raise
                exact = rows == self._expected_migration_rows()
            connection.rollback()
            return exact
        except Exception:
            _rollback_quietly(connection)
            return None
        finally:
            _close_quietly(connection)

    def _apply_packaged_migrations(
        self,
        context: PostgresBootstrapContext,
    ) -> None:
        factory = self._credential_factories.migrator
        if factory is None:
            raise PostgresBootstrapStorageError(
                "migrator credential disappeared during reconciliation"
            )
        connection = _fresh_connection(factory, label="migrator")
        try:
            self._require_migrator_connection_identity(connection, context)
            self._set_migration_role(connection, context.roles.schema_owner)
            outcome: str | None = None
            try:
                apply_migrations(connection, load_migrations())
                return
            except MigrationDriftError:
                outcome = "drift"
            except Exception:
                readback = self._migration_ledger_readback(context)
                if readback is True:
                    return
                if readback is False:
                    outcome = "not-committed"
                else:
                    outcome = "unknown"
            if outcome == "drift":
                raise PostgresBootstrapMigrationError(
                    "packaged PostgreSQL migrations have drifted"
                )
            if outcome == "not-committed":
                raise PostgresBootstrapMigrationError(
                    "packaged PostgreSQL migrations did not commit"
                )
            raise PostgresBootstrapCommitUnknownError(PostgresBootstrapPhase.MIGRATIONS)
        finally:
            _close_quietly(connection)

    def _reconcile_migrations(
        self,
        context: PostgresBootstrapContext,
    ) -> tuple[int, ...]:
        expected = self._expected_migration_rows()
        if context.adoption is None:
            self._prepare_fresh_migration_authority(context)
            self._apply_packaged_migrations(context)
        else:
            adoption_readback = self._migration_ledger_readback(context)
            if adoption_readback is None:
                raise PostgresBootstrapStorageError(
                    "existing PostgreSQL migration history could not be read"
                )
            if adoption_readback is False:
                raise PostgresBootstrapMigrationError(
                    "existing PostgreSQL migration history is not exact"
                )
        final_readback = self._migration_ledger_readback(context)
        if final_readback is None:
            raise PostgresBootstrapStorageError(
                "PostgreSQL migration history could not be read"
            )
        if final_readback is False:
            raise PostgresBootstrapMigrationError(
                "PostgreSQL migration history could not be verified"
            )
        return tuple(row[0] for row in expected)

    @staticmethod
    def _expected_relation_rows(
        owner: str,
    ) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            sorted(
                tuple((table, "r", owner) for table in _AUTHORITY_TABLES)
                + tuple((sequence, "S", owner) for sequence in _LEGACY_SEQUENCES),
                key=lambda row: (row[1], row[0]),
            )
        )

    @staticmethod
    def _expected_non_owner_acl_rows(
        context: PostgresBootstrapContext,
    ) -> tuple[tuple[str, str, str, str, str, str, bool], ...]:
        roles = context.roles
        rows: set[tuple[str, str, str, str, str, str, bool]] = set()
        for role in roles.login_roles:
            rows.add(
                (
                    "schema",
                    "np",
                    "",
                    roles.schema_owner,
                    role,
                    "USAGE",
                    False,
                )
            )
        for table, privileges in _LEGACY_PRIVILEGES.items():
            rows.update(
                (
                    "relation",
                    table,
                    "",
                    roles.schema_owner,
                    roles.legacy_runtime,
                    privilege,
                    False,
                )
                for privilege in privileges
            )
        for table, privileges in _ATOMIC_PRIVILEGES.items():
            rows.update(
                (
                    "relation",
                    table,
                    "",
                    roles.schema_owner,
                    roles.atomic_runtime,
                    privilege,
                    False,
                )
                for privilege in privileges
            )
        for table in ("paper_runtime_control", "paper_runtime_generations"):
            rows.add(
                (
                    "relation",
                    table,
                    "",
                    roles.schema_owner,
                    roles.atomic_runtime,
                    "SELECT",
                    False,
                )
            )
        rows.update(
            (
                "relation",
                table,
                "",
                roles.schema_owner,
                roles.readiness,
                "SELECT",
                False,
            )
            for table in _AUTHORITY_TABLES
        )
        rows.add(
            (
                "relation",
                "trades",
                "",
                roles.schema_owner,
                roles.trainer,
                "SELECT",
                False,
            )
        )
        for table in _AUTHORITY_TABLES:
            rows.update(
                {
                    (
                        "relation",
                        table,
                        "",
                        roles.schema_owner,
                        roles.activation,
                        privilege,
                        False,
                    )
                    for privilege in ("SELECT", "UPDATE")
                }
            )
        rows.add(
            (
                "relation",
                "paper_runtime_generations",
                "",
                roles.schema_owner,
                roles.activation,
                "INSERT",
                False,
            )
        )
        rows.update(
            (
                "relation",
                sequence,
                "",
                roles.schema_owner,
                roles.legacy_runtime,
                "USAGE",
                False,
            )
            for sequence in _LEGACY_SEQUENCES
        )
        return tuple(sorted(rows))

    @staticmethod
    def _expected_column_acl_rows(
        context: PostgresBootstrapContext,
    ) -> tuple[tuple[str, str, str, str, str, bool], ...]:
        roles = context.roles
        return tuple(
            sorted(
                (
                    (
                        "open_positions",
                        "id",
                        roles.schema_owner,
                        roles.legacy_runtime,
                        "UPDATE",
                        False,
                    ),
                    (
                        "paper_runtime_control",
                        "control_key",
                        roles.schema_owner,
                        roles.atomic_runtime,
                        "UPDATE",
                        False,
                    ),
                    (
                        "paper_runtime_generations",
                        "activation_id",
                        roles.schema_owner,
                        roles.atomic_runtime,
                        "UPDATE",
                        False,
                    ),
                )
            )
        )

    @staticmethod
    def _expected_database_acl_rows(
        context: PostgresBootstrapContext,
    ) -> tuple[tuple[str, str, str, bool], ...]:
        return tuple(
            sorted(
                ((context.admin_role, context.roles.schema_owner, "CREATE", False),)
                + tuple(
                    (context.admin_role, role, "CONNECT", False)
                    for role in context.roles.login_roles
                )
            )
        )

    @staticmethod
    def _index_evidence_is_exact(cursor: object) -> bool:
        cursor.execute(_SELECT_INDEX_EVIDENCE_SQL)
        rows = tuple(tuple(row) for row in cursor.fetchall())
        canonical_names = {row[0] for row in rows if row[0] in _EXPECTED_INDEX_NAMES}
        if canonical_names != _EXPECTED_INDEX_NAMES:
            return False
        standalone = tuple(
            row[:12] for row in rows if row[0] in _REQUIRED_STANDALONE_INDEXES
        )
        if standalone != _EXPECTED_STANDALONE_INDEX_EVIDENCE:
            return False
        canonical_security = tuple(
            row for row in rows if row[0] in _EXPECTED_INDEX_NAMES
        )
        encoded_security = json.dumps(
            canonical_security,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        if hashlib.sha256(encoded_security).hexdigest() != (
            _EXPECTED_INDEX_SECURITY_EVIDENCE_SHA256
        ):
            return False
        for row in rows:
            if row[0] in _EXPECTED_INDEX_NAMES:
                continue
            (
                _name,
                table,
                _method,
                unique,
                primary,
                valid,
                ready,
                key_count,
                attribute_count,
                _keys,
                predicate,
                expressions,
                persistence,
                live,
                clustered,
                replica_identity,
                nulls_not_distinct,
                opclasses,
                collations,
                options,
                owner_matches_table,
                default_builtin_opclasses,
                builtin_collations,
                default_options,
            ) = _one_row(row, 24, "index evidence")
            if (
                table not in _AUTHORITY_TABLES
                or _method != "btree"
                or unique
                or primary
                or not valid
                or not ready
                or key_count != attribute_count
                or predicate
                or expressions
                or persistence != "p"
                or not live
                or clustered
                or replica_identity
                or nulls_not_distinct
                or not owner_matches_table
                or not default_builtin_opclasses
                or not builtin_collations
                or not default_options
                or not isinstance(opclasses, str)
                or not isinstance(collations, str)
                or not isinstance(options, str)
            ):
                return False
        return True

    @staticmethod
    def _relation_shape_evidence_is_exact(cursor: object) -> bool:
        cursor.execute(_SELECT_COLUMN_EVIDENCE_SQL)
        columns = tuple(tuple(row) for row in cursor.fetchall())
        encoded_columns = json.dumps(
            columns,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        if hashlib.sha256(encoded_columns).hexdigest() != (
            _EXPECTED_COLUMN_EVIDENCE_SHA256
        ):
            return False
        cursor.execute(_SELECT_SEQUENCE_EVIDENCE_SQL)
        sequences = tuple(tuple(row) for row in cursor.fetchall())
        if sequences != _EXPECTED_SEQUENCE_EVIDENCE:
            return False
        cursor.execute(_SELECT_CONSTRAINT_EVIDENCE_SQL)
        constraints = tuple(tuple(row) for row in cursor.fetchall())
        encoded_constraints = json.dumps(
            constraints,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded_constraints).hexdigest() == (
            _EXPECTED_CONSTRAINT_EVIDENCE_SHA256
        )

    @classmethod
    def _catalog_shape_is_expected(
        cls,
        cursor: object,
        context: PostgresBootstrapContext,
        *,
        allow_historical_owners: bool,
    ) -> bool:
        if not cls._database_catalog_is_admissible(cursor, context):
            return False
        managed_or_historical = list(context.roles.all)
        if context.adoption is not None:
            managed_or_historical.extend(
                context.adoption.allowed_historical_owner_roles
            )
        cursor.execute(
            _SELECT_MANAGED_OWNERSHIP_OUTSIDE_NP_SQL,
            (managed_or_historical,),
        )
        if cursor.fetchall():
            return False
        cursor.execute(
            _SELECT_MANAGED_ACLS_OUTSIDE_NP_SQL,
            (managed_or_historical,),
        )
        if cursor.fetchall():
            return False
        cursor.execute(_SELECT_PUBLIC_SCHEMA_ACL_SQL)
        public_acl_rows = tuple(tuple(row) for row in cursor.fetchall())
        if public_acl_rows != (
            ("pg_database_owner", "PUBLIC", "USAGE", False),
            ("pg_database_owner", "pg_database_owner", "CREATE", False),
            ("pg_database_owner", "pg_database_owner", "USAGE", False),
        ):
            return False
        cursor.execute(_SELECT_UNEXPECTED_USER_SCHEMAS_SQL)
        if cursor.fetchall():
            return False
        cursor.execute(_SELECT_UNEXPECTED_PUBLIC_OBJECTS_SQL)
        if cursor.fetchall():
            return False
        cursor.execute(_SELECT_LARGE_OBJECT_COUNT_SQL)
        large_object_count = _one_row(cursor.fetchone(), 1, "large object inventory")[0]
        if type(large_object_count) is not int or large_object_count != 0:
            return False
        cursor.execute(
            _SELECT_UNEXPECTED_NP_CATALOG_OBJECTS_SQL,
            (list(_AUTHORITY_TABLES),),
        )
        if cursor.fetchall():
            return False
        cursor.execute(_SELECT_SCHEMA_OBJECTS_SQL)
        objects = tuple(tuple(row) for row in cursor.fetchall())
        expected_names = {
            *((table, "r") for table in _AUTHORITY_TABLES),
            *((sequence, "S") for sequence in _LEGACY_SEQUENCES),
        }
        if {(row[0], row[1]) for row in objects} != expected_names:
            return False
        cursor.execute(_SELECT_SCHEMA_AUTHORITY_SQL)
        schema_rows = tuple(tuple(row) for row in cursor.fetchall())
        if len(schema_rows) != 1:
            return False
        schema_owner = schema_rows[0][0]
        schema_marker = schema_rows[0][1]
        expected_marker = cls._schema_marker(context)
        final_authority = (
            schema_owner == context.roles.schema_owner
            and schema_marker == expected_marker
            and all(row[2] == context.roles.schema_owner for row in objects)
        )
        historical_authority = False
        if allow_historical_owners and context.adoption is not None:
            migration_authority = context.adoption.migration_authority_role
            historical_authority = (
                schema_owner == migration_authority
                and schema_marker is None
                and all(row[2] == migration_authority for row in objects)
            )
        if not final_authority and not historical_authority:
            return False

        cursor.execute(_SELECT_SCHEMA_FUNCTIONS_SQL)
        functions = tuple(tuple(row) for row in cursor.fetchall())
        if tuple(row[:3] for row in functions) != _EXPECTED_FUNCTION_IDENTITIES:
            return False
        activation_names = {
            "acquire_paper_runtime_activation_fence",
            "activate_paper_runtime_generation",
        }
        for name, _arguments, _kind, owner in functions:
            expected_owner = (
                context.roles.activation
                if name in activation_names
                else context.roles.schema_owner
            )
            if final_authority:
                if owner == expected_owner:
                    continue
                if (
                    allow_historical_owners
                    and name in activation_names
                    and owner == context.roles.schema_owner
                ):
                    continue
                return False
            if (
                historical_authority
                and context.adoption is not None
                and owner != context.adoption.migration_authority_role
            ):
                return False

        cursor.execute(_SELECT_SCHEMA_ACL_GRANTEES_SQL)
        acl_rows = tuple(tuple(row) for row in cursor.fetchall())
        if allow_historical_owners:
            historical_function_defaults = all(
                row[0] == "function"
                and row[5] == "EXECUTE"
                and row[4] == "PUBLIC"
                and row[6] is False
                for row in acl_rows
            )
            if (
                acl_rows != cls._expected_non_owner_acl_rows(context)
                and not historical_function_defaults
            ):
                return False
        elif acl_rows != cls._expected_non_owner_acl_rows(context):
            return False
        cursor.execute(_SELECT_DATABASE_ACL_GRANTEES_SQL)
        database_rows = tuple(tuple(row) for row in cursor.fetchall())
        expected_database_rows = cls._expected_database_acl_rows(context)
        if not allow_historical_owners and database_rows != expected_database_rows:
            return False
        if allow_historical_owners and database_rows != expected_database_rows:
            if context.adoption is None:
                permitted_initial = {
                    (context.admin_role, "PUBLIC", "CONNECT", False),
                    (context.admin_role, "PUBLIC", "TEMPORARY", False),
                    (
                        context.admin_role,
                        context.roles.schema_owner,
                        "CREATE",
                        False,
                    ),
                }
            else:
                adoption = context.adoption
                cursor.execute(_SELECT_DATABASE_OWNER_SQL)
                database_owner = _one_row(cursor.fetchone(), 1, "database owner")[0]
                if database_owner == context.admin_role:
                    permitted_initial = {
                        (context.admin_role, "PUBLIC", "CONNECT", False),
                        (context.admin_role, "PUBLIC", "TEMPORARY", False),
                    }
                elif database_owner == adoption.migration_authority_role:
                    permitted_initial = {
                        (
                            adoption.migration_authority_role,
                            "PUBLIC",
                            "CONNECT",
                            False,
                        ),
                        (
                            adoption.migration_authority_role,
                            "PUBLIC",
                            "TEMPORARY",
                            False,
                        ),
                    }
                else:
                    return False
                if (
                    database_owner == context.admin_role
                    and adoption.old_shared_runtime_role is not None
                ):
                    permitted_initial.add(
                        (
                            context.admin_role,
                            adoption.migration_authority_role,
                            "CREATE",
                            False,
                        )
                    )
            if set(database_rows) != permitted_initial:
                return False
        cursor.execute(_SELECT_COLUMN_ACLS_SQL)
        column_rows = tuple(tuple(row) for row in cursor.fetchall())
        if allow_historical_owners:
            if column_rows and column_rows != cls._expected_column_acl_rows(context):
                return False
        elif column_rows != cls._expected_column_acl_rows(context):
            return False
        cursor.execute(
            _SELECT_DEFAULT_ACLS_SQL,
            (list(context.roles.all),),
        )
        if cursor.fetchall():
            return False
        return cls._index_evidence_is_exact(
            cursor
        ) and cls._relation_shape_evidence_is_exact(cursor)

    @staticmethod
    def _grant_table_privileges(
        cursor: object,
        role: str,
        manifest: dict[str, tuple[str, ...]],
    ) -> None:
        for table, privileges in manifest.items():
            cursor.execute(
                sql.SQL("GRANT {} ON TABLE np.{} TO {}").format(
                    sql.SQL(", ").join(sql.SQL(item) for item in privileges),
                    sql.Identifier(table),
                    sql.Identifier(role),
                )
            )

    @classmethod
    def _apply_catalog_authority(
        cls,
        cursor: object,
        context: PostgresBootstrapContext,
    ) -> None:
        roles = context.roles
        cursor.execute(
            "COMMENT ON SCHEMA np IS %s",
            (cls._schema_marker(context),),
        )
        cursor.execute(
            sql.SQL("ALTER SCHEMA np OWNER TO {}").format(
                sql.Identifier(roles.schema_owner)
            )
        )
        for table in _AUTHORITY_TABLES:
            cursor.execute(
                sql.SQL("ALTER TABLE np.{} OWNER TO {}").format(
                    sql.Identifier(table),
                    sql.Identifier(roles.schema_owner),
                )
            )
        for sequence in _LEGACY_SEQUENCES:
            cursor.execute(
                sql.SQL("ALTER SEQUENCE np.{} OWNER TO {}").format(
                    sql.Identifier(sequence),
                    sql.Identifier(roles.schema_owner),
                )
            )
        for function in _NON_ACTIVATION_FUNCTIONS:
            cursor.execute(
                sql.SQL("ALTER FUNCTION {} OWNER TO {}").format(
                    sql.SQL(function),
                    sql.Identifier(roles.schema_owner),
                )
            )
        for function in _ACTIVATION_FUNCTIONS:
            cursor.execute(
                sql.SQL("ALTER FUNCTION {} OWNER TO {}").format(
                    sql.SQL(function),
                    sql.Identifier(roles.activation),
                )
            )

        cursor.execute("REVOKE ALL ON SCHEMA np FROM PUBLIC")
        cursor.execute(
            sql.SQL("GRANT USAGE ON SCHEMA np TO {}").format(
                sql.SQL(", ").join(sql.Identifier(role) for role in roles.login_roles)
            )
        )
        for role in roles.login_roles:
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

        cls._grant_table_privileges(cursor, roles.legacy_runtime, _LEGACY_PRIVILEGES)
        cls._grant_table_privileges(cursor, roles.atomic_runtime, _ATOMIC_PRIVILEGES)
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
                    for table in _AUTHORITY_TABLES
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
                    for sequence in _LEGACY_SEQUENCES
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
                "GRANT UPDATE(activation_id) ON np.paper_runtime_generations TO {}"
            ).format(sql.Identifier(roles.atomic_runtime))
        )
        cursor.execute(
            sql.SQL("GRANT SELECT ON TABLE {} TO {}").format(
                sql.SQL(", ").join(
                    sql.SQL("np.{}").format(sql.Identifier(table))
                    for table in _AUTHORITY_TABLES
                ),
                sql.Identifier(roles.activation),
            )
        )
        for table in _ACTIVATION_LOCK_TABLES:
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
        for function in _ACTIVATION_FUNCTIONS:
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
                sql.Identifier(context.expected_database)
            )
        )
        historical_roles = ()
        if context.adoption is not None:
            historical_roles = context.adoption.allowed_historical_owner_roles
        for historical_role in historical_roles:
            cursor.execute(
                sql.SQL("REVOKE ALL ON DATABASE {} FROM {}").format(
                    sql.Identifier(context.expected_database),
                    sql.Identifier(historical_role),
                )
            )
        cursor.execute(
            sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                sql.Identifier(context.expected_database),
                sql.Identifier(context.admin_role),
            )
        )
        cursor.execute(
            sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                sql.Identifier(context.expected_database),
                sql.Identifier(roles.schema_owner),
            )
        )
        cursor.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(context.expected_database),
                sql.SQL(", ").join(sql.Identifier(role) for role in roles.login_roles),
            )
        )

    @staticmethod
    def _require_no_old_backends(
        cursor: object,
        context: PostgresBootstrapContext,
    ) -> None:
        adoption = context.adoption
        if adoption is None or adoption.old_shared_runtime_role is None:
            return
        cursor.execute(_SELECT_OLD_BACKENDS_SQL, (adoption.old_shared_runtime_role,))
        count = _one_row(cursor.fetchone(), 1, "old runtime backend count")[0]
        if type(count) is not int or count != 0:
            raise PostgresBootstrapDriftError(
                "old shared runtime still has an active backend session"
            )

    @staticmethod
    def _demote_old_role(
        cursor: object,
        context: PostgresBootstrapContext,
    ) -> bool:
        adoption = context.adoption
        if adoption is None or adoption.old_shared_runtime_role is None:
            return False
        old_role = adoption.old_shared_runtime_role
        cursor.execute(
            sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                sql.Identifier(context.expected_database),
                sql.Identifier(context.admin_role),
            )
        )
        cursor.execute(
            _SELECT_OLD_MEMBERSHIPS_SQL,
            (old_role, old_role),
        )
        if cursor.fetchall():
            raise PostgresBootstrapDriftError(
                "old shared runtime role memberships changed before cutover"
            )
        cursor.execute(
            sql.SQL("REVOKE ALL ON DATABASE {} FROM {}").format(
                sql.Identifier(context.expected_database),
                sql.Identifier(old_role),
            )
        )
        cursor.execute(
            sql.SQL("REVOKE ALL ON SCHEMA np FROM {}").format(sql.Identifier(old_role))
        )
        cursor.execute(
            sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA np FROM {}").format(
                sql.Identifier(old_role)
            )
        )
        cursor.execute(
            sql.SQL("REVOKE ALL ON ALL SEQUENCES IN SCHEMA np FROM {}").format(
                sql.Identifier(old_role)
            )
        )
        cursor.execute(
            sql.SQL("REVOKE ALL ON ALL FUNCTIONS IN SCHEMA np FROM {}").format(
                sql.Identifier(old_role)
            )
        )
        cursor.execute(
            sql.SQL("REVOKE CONNECT ON DATABASE {} FROM {}").format(
                sql.Identifier(context.expected_database),
                sql.Identifier(old_role),
            )
        )
        cursor.execute(
            sql.SQL(
                "ALTER ROLE {} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS NOINHERIT CONNECTION LIMIT -1 "
                "PASSWORD NULL"
            ).format(sql.Identifier(old_role))
        )
        return True

    def _reconcile_catalog(self, context: PostgresBootstrapContext) -> bool:
        connection = _fresh_connection(self._admin_connection_factory, label="admin")
        phase = (
            PostgresBootstrapPhase.DEMOTION
            if context.adoption is not None
            and context.adoption.old_shared_runtime_role is not None
            else PostgresBootstrapPhase.CATALOG
        )
        cutover_failed = False
        commit_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_COMMITTED_SQL)
                    cursor.execute(_SAFE_SEARCH_PATH_SQL)
                    cursor.execute("SET LOCAL lock_timeout = '1s'")
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_BOOTSTRAP_ADVISORY_LOCK_ID,),
                    )
                    self._require_admin_identity(cursor, context)
                    cursor.execute(_LOCK_AUTHORITY_TABLES_SQL)
                    self._managed_roles_are_exact(
                        cursor,
                        context,
                        allow_absent=False,
                    )
                    self._require_no_old_backends(cursor, context)
                    if not self._migration_history_is_exact(cursor):
                        raise PostgresBootstrapDriftError(
                            "migration history changed before catalog cutover"
                        )
                    if not self._catalog_shape_is_expected(
                        cursor,
                        context,
                        allow_historical_owners=True,
                    ):
                        raise PostgresBootstrapDriftError(
                            "PostgreSQL catalog contains unexpected objects or owners"
                        )
                    self._apply_catalog_authority(cursor, context)
                    demoted = self._demote_old_role(cursor, context)
                    cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")
                    self._managed_roles_are_exact(
                        cursor,
                        context,
                        allow_absent=False,
                    )
                    if not self._catalog_shape_is_expected(
                        cursor,
                        context,
                        allow_historical_owners=False,
                    ):
                        raise PostgresBootstrapDriftError(
                            "PostgreSQL catalog cutover is not exact"
                        )
                    cursor.execute(
                        _SELECT_DATABASE_AUTHORITY_SQL,
                        (context.roles.schema_owner,),
                    )
                    database = _one_row(cursor.fetchone(), 3, "database authority")
                    if database != (context.admin_role, True, False):
                        raise PostgresBootstrapDriftError(
                            "database authority cutover is not exact"
                        )
                    if not _activation_catalog_is_authoritative(cursor):
                        raise PostgresBootstrapDriftError(
                            "PostgreSQL catalog cutover did not become authoritative"
                        )
            except PostgresBootstrapDriftError, PostgresBootstrapMigrationError:
                _rollback_quietly(connection)
                raise
            except Exception:
                _rollback_quietly(connection)
                cutover_failed = True
            if cutover_failed:
                raise PostgresBootstrapStorageError(
                    "PostgreSQL catalog cutover failed before commit"
                )
            try:
                connection.commit()
            except Exception:
                commit_failed = True
            if commit_failed:
                _rollback_quietly(connection)
                if self._catalog_readback_is_exact(context):
                    return bool(
                        context.adoption is not None
                        and context.adoption.old_shared_runtime_role is not None
                    )
                raise PostgresBootstrapCommitUnknownError(phase)
            return demoted
        finally:
            _close_quietly(connection)

    def _catalog_readback_is_exact(
        self,
        context: PostgresBootstrapContext,
    ) -> bool:
        try:
            connection = _fresh_connection(
                self._admin_connection_factory,
                label="catalog readback",
            )
        except PostgresBootstrapStorageError:
            return False
        try:
            with connection.cursor() as cursor:
                cursor.execute(_READ_COMMITTED_SQL)
                cursor.execute(_SAFE_SEARCH_PATH_SQL)
                self._require_admin_identity(cursor, context)
                self._managed_roles_are_exact(
                    cursor,
                    context,
                    allow_absent=False,
                )
                if not self._migration_history_is_exact(cursor):
                    return False
                if not self._catalog_shape_is_expected(
                    cursor,
                    context,
                    allow_historical_owners=False,
                ):
                    return False
                if not _activation_catalog_is_authoritative(cursor):
                    return False
                cursor.execute(
                    _SELECT_DATABASE_AUTHORITY_SQL,
                    (context.roles.schema_owner,),
                )
                database = _one_row(cursor.fetchone(), 3, "database authority")
                if database != (context.admin_role, True, False):
                    return False
                adoption = context.adoption
                if (
                    adoption is not None
                    and adoption.old_shared_runtime_role is not None
                ):
                    cursor.execute(
                        _SELECT_OLD_ROLE_SQL, (adoption.old_shared_runtime_role,)
                    )
                    old_role = _one_row(cursor.fetchone(), 9, "old shared runtime")
                    if old_role != (
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        -1,
                        True,
                    ):
                        return False
                    cursor.execute(
                        _SELECT_OLD_MEMBERSHIPS_SQL,
                        (
                            adoption.old_shared_runtime_role,
                            adoption.old_shared_runtime_role,
                        ),
                    )
                    if cursor.fetchall():
                        return False
                    cursor.execute(
                        _SELECT_OLD_BACKENDS_SQL,
                        (adoption.old_shared_runtime_role,),
                    )
                    old_backends = _one_row(
                        cursor.fetchone(), 1, "old runtime backend count"
                    )[0]
                    if type(old_backends) is not int or old_backends != 0:
                        return False
            connection.rollback()
            return True
        except Exception:
            _rollback_quietly(connection)
            return False
        finally:
            _close_quietly(connection)
