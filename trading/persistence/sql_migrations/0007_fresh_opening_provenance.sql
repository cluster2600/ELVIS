CREATE TABLE np.paper_fresh_opening_admissions (
    control_key BOOLEAN PRIMARY KEY DEFAULT TRUE,
    candidate_payload_sha256 CHAR(64) NOT NULL,
    pin_authority_record_sha256 CHAR(64) NOT NULL,
    deployment_incarnation_id VARCHAR(255) NOT NULL,
    admission_payload TEXT NOT NULL,
    admission_payload_sha256 CHAR(64) NOT NULL,
    admitted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT paper_fresh_opening_admissions_singleton CHECK (control_key),
    CONSTRAINT paper_fresh_opening_admissions_hashes_valid CHECK (
        candidate_payload_sha256 ~ '^[0-9a-f]{64}$'
        AND candidate_payload_sha256 <> repeat('0', 64)
        AND pin_authority_record_sha256 ~ '^[0-9a-f]{64}$'
        AND pin_authority_record_sha256 <> repeat('0', 64)
        AND admission_payload_sha256 ~ '^[0-9a-f]{64}$'
        AND admission_payload_sha256 <> repeat('0', 64)
    ),
    CONSTRAINT paper_fresh_opening_admissions_deployment_clean CHECK (
        deployment_incarnation_id = BTRIM(deployment_incarnation_id)
        AND deployment_incarnation_id <> ''
        AND deployment_incarnation_id ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_admissions_payload_present CHECK (
        admission_payload <> ''
    ),
    CONSTRAINT paper_fresh_opening_admissions_admitted_at_finite CHECK (
        isfinite(admitted_at)
    ),
    CONSTRAINT paper_fresh_opening_admissions_binding_uq UNIQUE (
        candidate_payload_sha256,
        pin_authority_record_sha256,
        deployment_incarnation_id
    )
);

CREATE TABLE np.paper_fresh_opening_nonces (
    trust_domain VARCHAR(128) NOT NULL,
    signer_key_id VARCHAR(255) NOT NULL,
    nonce CHAR(64) NOT NULL,
    candidate_payload_sha256 CHAR(64) NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT paper_fresh_opening_nonces_pk PRIMARY KEY (
        trust_domain,
        signer_key_id,
        nonce
    ),
    CONSTRAINT paper_fresh_opening_nonces_trust_domain_clean CHECK (
        trust_domain = BTRIM(trust_domain)
        AND trust_domain <> ''
        AND trust_domain ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_nonces_signer_key_id_clean CHECK (
        signer_key_id = BTRIM(signer_key_id)
        AND signer_key_id <> ''
        AND signer_key_id ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_nonces_nonce_valid CHECK (
        nonce ~ '^[0-9a-f]{64}$'
        AND nonce <> repeat('0', 64)
    ),
    CONSTRAINT paper_fresh_opening_nonces_candidate_sha256_valid CHECK (
        candidate_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_fresh_opening_nonces_registered_at_finite CHECK (
        isfinite(registered_at)
    ),
    CONSTRAINT paper_fresh_opening_nonces_candidate_ref_uq UNIQUE (
        trust_domain,
        signer_key_id,
        nonce,
        candidate_payload_sha256
    )
);

CREATE TABLE np.paper_fresh_opening_provisionings (
    control_key BOOLEAN PRIMARY KEY DEFAULT TRUE,
    trust_domain VARCHAR(128) NOT NULL,
    signer_key_id VARCHAR(255) NOT NULL,
    nonce CHAR(64) NOT NULL,
    logical_target VARCHAR(255) NOT NULL,
    execution_scope VARCHAR(128) NOT NULL,
    account_key VARCHAR(255) NOT NULL,
    owner_generation BIGINT NOT NULL,
    collateral_asset VARCHAR(64) NOT NULL,
    opening_version SMALLINT NOT NULL,
    intent_payload TEXT NOT NULL,
    intent_payload_sha256 CHAR(64) NOT NULL,
    approval_payload TEXT NOT NULL,
    approval_payload_sha256 CHAR(64) NOT NULL,
    trust_policy_payload TEXT NOT NULL,
    trust_policy_payload_sha256 CHAR(64) NOT NULL,
    candidate_payload TEXT NOT NULL,
    candidate_payload_sha256 CHAR(64) NOT NULL,
    opening_payload TEXT NOT NULL,
    opening_payload_sha256 CHAR(64) NOT NULL,
    opening_receipt_payload TEXT NOT NULL,
    opening_receipt_payload_sha256 CHAR(64) NOT NULL,
    provisioning_receipt_payload TEXT NOT NULL,
    provisioning_receipt_payload_sha256 CHAR(64) NOT NULL,
    database_name TEXT NOT NULL,
    system_identifier NUMERIC(20, 0) NOT NULL,
    control_plane_role VARCHAR(63) NOT NULL,
    opening_anchor_role VARCHAR(63) NOT NULL,
    migration_version INTEGER NOT NULL,
    migration_name TEXT NOT NULL,
    migration_checksum CHAR(64) NOT NULL,
    terminal_catalog_sha256 CHAR(64) NOT NULL,
    deployment_incarnation_id VARCHAR(255) NOT NULL,
    database_incarnation_id CHAR(64) NOT NULL,
    pin_authority_record_sha256 CHAR(64) NOT NULL,
    runtime_mode TEXT NOT NULL,
    runtime_generation BIGINT NOT NULL,
    authority_transition_sequence BIGINT NOT NULL,
    writer_fence BIGINT NOT NULL,
    runtime_activation_authorized BOOLEAN NOT NULL,
    trading_authorized BOOLEAN NOT NULL,
    stale_on_return BOOLEAN NOT NULL,
    authority_evaluated_at TIMESTAMPTZ NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT paper_fresh_opening_provisionings_singleton CHECK (control_key),
    CONSTRAINT paper_fresh_opening_provisionings_trust_domain_clean CHECK (
        trust_domain = BTRIM(trust_domain)
        AND trust_domain <> ''
        AND trust_domain ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_signer_key_id_clean CHECK (
        signer_key_id = BTRIM(signer_key_id)
        AND signer_key_id <> ''
        AND signer_key_id ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_nonce_valid CHECK (
        nonce ~ '^[0-9a-f]{64}$'
        AND nonce <> repeat('0', 64)
    ),
    CONSTRAINT paper_fresh_opening_provisionings_logical_target_clean CHECK (
        logical_target = BTRIM(logical_target)
        AND logical_target <> ''
        AND logical_target ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_execution_scope_clean CHECK (
        execution_scope = BTRIM(execution_scope)
        AND execution_scope <> ''
        AND execution_scope ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_account_key_clean CHECK (
        account_key = BTRIM(account_key)
        AND account_key <> ''
        AND account_key ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_owner_generation_positive CHECK (
        owner_generation > 0
    ),
    CONSTRAINT paper_fresh_opening_provisionings_collateral_asset_clean CHECK (
        collateral_asset = BTRIM(collateral_asset)
        AND collateral_asset <> ''
        AND collateral_asset ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_opening_version_known CHECK (
        opening_version = 1
    ),
    CONSTRAINT paper_fresh_opening_provisionings_payloads_present CHECK (
        intent_payload <> ''
        AND approval_payload <> ''
        AND trust_policy_payload <> ''
        AND candidate_payload <> ''
        AND opening_payload <> ''
        AND opening_receipt_payload <> ''
        AND provisioning_receipt_payload <> ''
    ),
    CONSTRAINT paper_fresh_opening_provisionings_sha256_values_valid CHECK (
        intent_payload_sha256 ~ '^[0-9a-f]{64}$'
        AND approval_payload_sha256 ~ '^[0-9a-f]{64}$'
        AND trust_policy_payload_sha256 ~ '^[0-9a-f]{64}$'
        AND candidate_payload_sha256 ~ '^[0-9a-f]{64}$'
        AND opening_payload_sha256 ~ '^[0-9a-f]{64}$'
        AND opening_receipt_payload_sha256 ~ '^[0-9a-f]{64}$'
        AND provisioning_receipt_payload_sha256 ~ '^[0-9a-f]{64}$'
        AND migration_checksum ~ '^[0-9a-f]{64}$'
        AND terminal_catalog_sha256 ~ '^[0-9a-f]{64}$'
        AND database_incarnation_id ~ '^[0-9a-f]{64}$'
        AND pin_authority_record_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_database_name_clean CHECK (
        database_name = BTRIM(database_name) AND database_name <> ''
    ),
    CONSTRAINT paper_fresh_opening_provisionings_system_identifier_unsigned CHECK (
        system_identifier BETWEEN 1 AND 18446744073709551615
    ),
    CONSTRAINT paper_fresh_opening_provisionings_control_plane_role_clean CHECK (
        control_plane_role ~ '^[a-z][a-z0-9_]{0,62}$'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_opening_anchor_role_clean CHECK (
        opening_anchor_role ~ '^[a-z][a-z0-9_]{0,62}$'
        AND opening_anchor_role <> control_plane_role
    ),
    CONSTRAINT paper_fresh_opening_provisionings_migration_terminal CHECK (
        migration_version = 7
        AND migration_name = 'fresh_opening_provenance'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_deployment_incarnation_clean CHECK (
        deployment_incarnation_id = BTRIM(deployment_incarnation_id)
        AND deployment_incarnation_id <> ''
        AND deployment_incarnation_id ~ '^[!-~]+$'
    ),
    CONSTRAINT paper_fresh_opening_provisionings_authority_dormant CHECK (
        runtime_mode = 'LEGACY'
        AND runtime_generation = 0
        AND authority_transition_sequence = 0
        AND writer_fence = 0
        AND runtime_activation_authorized IS FALSE
        AND trading_authorized IS FALSE
        AND stale_on_return IS TRUE
    ),
    CONSTRAINT paper_fresh_opening_provisionings_committed_at_finite CHECK (
        isfinite(authority_evaluated_at)
        AND isfinite(committed_at)
        AND authority_evaluated_at <= committed_at
    ),
    CONSTRAINT paper_fresh_opening_provisionings_nonce_fk FOREIGN KEY (
        trust_domain,
        signer_key_id,
        nonce,
        candidate_payload_sha256
    ) REFERENCES np.paper_fresh_opening_nonces (
        trust_domain,
        signer_key_id,
        nonce,
        candidate_payload_sha256
    ) ON DELETE RESTRICT,
    CONSTRAINT paper_fresh_opening_provisionings_opening_fk FOREIGN KEY (
        execution_scope,
        account_key,
        owner_generation,
        opening_version,
        opening_payload_sha256
    ) REFERENCES np.paper_account_streams (
        execution_scope,
        account_key,
        owner_generation,
        opening_version,
        opening_payload_sha256
    ) ON DELETE RESTRICT,
    CONSTRAINT paper_fresh_opening_provisionings_admission_fk FOREIGN KEY (
        candidate_payload_sha256,
        pin_authority_record_sha256,
        deployment_incarnation_id
    ) REFERENCES np.paper_fresh_opening_admissions (
        candidate_payload_sha256,
        pin_authority_record_sha256,
        deployment_incarnation_id
    ) ON DELETE RESTRICT,
    CONSTRAINT paper_fresh_opening_provisionings_receipt_uq UNIQUE (
        provisioning_receipt_payload_sha256
    ),
    CONSTRAINT paper_fresh_opening_provisionings_opening_ref_uq UNIQUE (
        execution_scope,
        account_key,
        owner_generation,
        opening_version,
        opening_payload_sha256
    )
);

ALTER TABLE np.paper_runtime_generations
ADD CONSTRAINT paper_runtime_generations_fresh_opening_provisioning_fk
FOREIGN KEY (
    execution_scope,
    account_key,
    owner_generation,
    opening_version,
    opening_payload_sha256
)
REFERENCES np.paper_fresh_opening_provisionings (
    execution_scope,
    account_key,
    owner_generation,
    opening_version,
    opening_payload_sha256
)
MATCH FULL
ON DELETE RESTRICT;

ALTER FUNCTION np.enforce_legacy_paper_runtime_fence()
SET search_path = pg_catalog, pg_temp;

ALTER FUNCTION np.reject_paper_runtime_generation_mutation()
SET search_path = pg_catalog, pg_temp;

ALTER FUNCTION np.acquire_paper_runtime_activation_fence()
SET search_path = pg_catalog, pg_temp;

ALTER FUNCTION np.activate_paper_runtime_generation(
    TEXT,
    BIGINT,
    BIGINT,
    TEXT,
    TEXT,
    TEXT,
    BIGINT,
    TEXT
)
SET search_path = pg_catalog, pg_temp;

CREATE FUNCTION np.reject_paper_fresh_opening_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
BEGIN
    RAISE EXCEPTION USING
        ERRCODE = '55000',
        MESSAGE = 'paper fresh opening provenance is append-only';
END
$function$;

REVOKE ALL ON FUNCTION np.reject_paper_fresh_opening_mutation() FROM PUBLIC;

CREATE TRIGGER paper_fresh_opening_nonces_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE
ON np.paper_fresh_opening_nonces
FOR EACH STATEMENT
EXECUTE FUNCTION np.reject_paper_fresh_opening_mutation();

ALTER TABLE np.paper_fresh_opening_nonces
ENABLE ALWAYS TRIGGER paper_fresh_opening_nonces_append_only;

CREATE TRIGGER paper_fresh_opening_admissions_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE
ON np.paper_fresh_opening_admissions
FOR EACH STATEMENT
EXECUTE FUNCTION np.reject_paper_fresh_opening_mutation();

ALTER TABLE np.paper_fresh_opening_admissions
ENABLE ALWAYS TRIGGER paper_fresh_opening_admissions_append_only;

CREATE TRIGGER paper_fresh_opening_provisionings_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE
ON np.paper_fresh_opening_provisionings
FOR EACH STATEMENT
EXECUTE FUNCTION np.reject_paper_fresh_opening_mutation();

ALTER TABLE np.paper_fresh_opening_provisionings
ENABLE ALWAYS TRIGGER paper_fresh_opening_provisionings_append_only;

CREATE FUNCTION np.protect_paper_account_opening_identity()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
BEGIN
    IF TG_OP = 'TRUNCATE' OR TG_OP = 'DELETE' THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper account opening identity is immutable';
    END IF;

    IF OLD.account_key IS DISTINCT FROM NEW.account_key
       OR OLD.execution_scope IS DISTINCT FROM NEW.execution_scope
       OR OLD.owner_generation IS DISTINCT FROM NEW.owner_generation
       OR OLD.collateral_asset IS DISTINCT FROM NEW.collateral_asset
       OR OLD.opening_version IS DISTINCT FROM NEW.opening_version
       OR OLD.opening_payload IS DISTINCT FROM NEW.opening_payload
       OR OLD.opening_payload_sha256 IS DISTINCT FROM NEW.opening_payload_sha256
       OR OLD.created_at IS DISTINCT FROM NEW.created_at THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper account opening identity is immutable';
    END IF;
    RETURN NEW;
END
$function$;

REVOKE ALL ON FUNCTION np.protect_paper_account_opening_identity() FROM PUBLIC;

CREATE TRIGGER paper_account_streams_opening_identity_immutable
BEFORE UPDATE OR DELETE
ON np.paper_account_streams
FOR EACH ROW
EXECUTE FUNCTION np.protect_paper_account_opening_identity();

ALTER TABLE np.paper_account_streams
ENABLE ALWAYS TRIGGER paper_account_streams_opening_identity_immutable;

CREATE TRIGGER paper_account_streams_opening_identity_truncate
BEFORE TRUNCATE
ON np.paper_account_streams
FOR EACH STATEMENT
EXECUTE FUNCTION np.protect_paper_account_opening_identity();

ALTER TABLE np.paper_account_streams
ENABLE ALWAYS TRIGGER paper_account_streams_opening_identity_truncate;

CREATE FUNCTION np.paper_terminal_catalog_fingerprint()
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    evidence JSONB;
BEGIN
    SELECT jsonb_build_object(
        'database_acl', COALESCE((
            SELECT jsonb_agg(to_jsonb(evidence_row) ORDER BY grantee, privilege_type)
            FROM (
                SELECT
                    COALESCE(grantee_role.rolname, 'PUBLIC') AS grantee,
                    database_acl.privilege_type,
                    database_acl.is_grantable
                FROM pg_catalog.pg_database database_row
                CROSS JOIN LATERAL pg_catalog.aclexplode(
                    COALESCE(
                        database_row.datacl,
                        pg_catalog.acldefault('d', database_row.datdba)
                    )
                ) database_acl
                LEFT JOIN pg_catalog.pg_roles grantee_role
                  ON grantee_role.oid = database_acl.grantee
                WHERE database_row.datname = pg_catalog.current_database()
            ) evidence_row
        ), '[]'::jsonb),
        'large_object_count', (
            SELECT COUNT(*)
            FROM pg_catalog.pg_largeobject_metadata
        ),
        'prepared_transaction_authority', pg_catalog.jsonb_build_object(
            'max_prepared_transactions',
            pg_catalog.current_setting('max_prepared_transactions')::INTEGER,
            'database_prepared_transaction_count', (
                SELECT COUNT(*)
                FROM pg_catalog.pg_prepared_xacts prepared_row
                WHERE prepared_row.database = pg_catalog.current_database()
            )
        ),
        'public_persistent_mutation_authority', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY function_name, identity_arguments
            )
            FROM (
                SELECT
                    function_row.proname AS function_name,
                    pg_catalog.pg_get_function_identity_arguments(
                        function_row.oid
                    ) AS identity_arguments,
                    EXISTS (
                        SELECT 1
                        FROM pg_catalog.aclexplode(
                            COALESCE(
                                function_row.proacl,
                                pg_catalog.acldefault(
                                    'f', function_row.proowner
                                )
                            )
                        ) function_acl
                        WHERE function_acl.grantee = 0
                          AND function_acl.privilege_type = 'EXECUTE'
                    ) AS public_execute,
                    COALESCE((
                        SELECT jsonb_agg(
                            role_row.rolname ORDER BY role_row.rolname
                        )
                        FROM pg_catalog.pg_roles role_row
                        WHERE pg_catalog.strpos(
                                pg_catalog.shobj_description(
                                    role_row.oid, 'pg_authid'
                                ),
                                'elvis-postgres-bootstrap:v2:'
                                || pg_catalog.current_database() || ':'
                              ) = 1
                          AND pg_catalog.has_function_privilege(
                                role_row.oid, function_row.oid, 'EXECUTE'
                              )
                    ), '[]'::jsonb) AS managed_executees
                FROM pg_catalog.pg_proc function_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = function_row.pronamespace
                WHERE namespace_row.nspname = 'pg_catalog'
                  AND function_row.oid IN (
                    'pg_catalog.lo_create(oid)'::pg_catalog.regprocedure,
                    'pg_catalog.lo_creat(integer)'::pg_catalog.regprocedure,
                    'pg_catalog.lo_from_bytea(oid,bytea)'::pg_catalog.regprocedure,
                    (
                        'pg_catalog.pg_logical_emit_message(boolean,text,text)'
                    )::pg_catalog.regprocedure,
                    (
                        'pg_catalog.pg_logical_emit_message(boolean,text,bytea)'
                    )::pg_catalog.regprocedure
                  )
            ) evidence_row
        ), '[]'::jsonb),
        'pg_catalog_explicit_execute_acls', COALESCE((
            WITH managed_roles AS (
                SELECT role_row.oid, role_row.rolname
                FROM pg_catalog.pg_roles role_row
                WHERE pg_catalog.strpos(
                    pg_catalog.shobj_description(role_row.oid, 'pg_authid'),
                    'elvis-postgres-bootstrap:v2:'
                    || pg_catalog.current_database() || ':'
                ) = 1
            )
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY function_name, identity_arguments, grantee
            )
            FROM (
                SELECT
                    function_row.proname AS function_name,
                    pg_catalog.pg_get_function_identity_arguments(
                        function_row.oid
                    ) AS identity_arguments,
                    COALESCE(grantee_role.rolname, 'PUBLIC') AS grantee,
                    function_acl.is_grantable
                FROM pg_catalog.pg_proc function_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = function_row.pronamespace
                CROSS JOIN LATERAL pg_catalog.aclexplode(
                    function_row.proacl
                ) function_acl
                LEFT JOIN pg_catalog.pg_roles grantee_role
                  ON grantee_role.oid = function_acl.grantee
                WHERE namespace_row.nspname = 'pg_catalog'
                  AND function_acl.privilege_type = 'EXECUTE'
                  AND (
                    function_acl.grantee = 0
                    OR function_acl.grantee = ANY(
                        SELECT managed_role.oid FROM managed_roles managed_role
                    )
                  )
            ) evidence_row
        ), '[]'::jsonb),
        'external_user_schemas', COALESCE((
            SELECT jsonb_agg(to_jsonb(evidence_row) ORDER BY schema_name)
            FROM (
                SELECT
                    namespace_row.nspname AS schema_name,
                    pg_catalog.pg_get_userbyid(namespace_row.nspowner) AS owner
                FROM pg_catalog.pg_namespace namespace_row
                WHERE namespace_row.nspname NOT IN (
                    'np', 'public', 'information_schema'
                )
                  AND namespace_row.nspname !~ '^pg_'
            ) evidence_row
        ), '[]'::jsonb),
        'public_objects', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY object_kind, object_name, identity_arguments
            )
            FROM (
                SELECT
                    'relation'::TEXT AS object_kind,
                    relation_row.relname::TEXT AS object_name,
                    ''::TEXT AS identity_arguments
                FROM pg_catalog.pg_class relation_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = relation_row.relnamespace
                WHERE namespace_row.nspname = 'public'
                UNION ALL
                SELECT
                    'routine'::TEXT,
                    function_row.proname::TEXT,
                    pg_catalog.pg_get_function_identity_arguments(function_row.oid)
                FROM pg_catalog.pg_proc function_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = function_row.pronamespace
                WHERE namespace_row.nspname = 'public'
                UNION ALL
                SELECT
                    'type'::TEXT,
                    type_row.typname::TEXT,
                    ''::TEXT
                FROM pg_catalog.pg_type type_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = type_row.typnamespace
                WHERE namespace_row.nspname = 'public'
                  AND type_row.typrelid = 0
                  AND type_row.typelem = 0
                UNION ALL
                SELECT
                    'collation'::TEXT,
                    collation_row.collname::TEXT,
                    ''::TEXT
                FROM pg_catalog.pg_collation collation_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = collation_row.collnamespace
                WHERE namespace_row.nspname = 'public'
            ) evidence_row
        ), '[]'::jsonb),
        'managed_external_acls', COALESCE((
            WITH managed_roles AS (
                SELECT role_row.oid, role_row.rolname
                FROM pg_catalog.pg_roles role_row
                WHERE pg_catalog.strpos(
                    pg_catalog.shobj_description(role_row.oid, 'pg_authid'),
                    'elvis-postgres-bootstrap:v2:'
                    || pg_catalog.current_database() || ':'
                ) = 1
            ), acl_evidence AS (
                SELECT
                    'schema'::TEXT AS object_kind,
                    namespace_row.nspname::TEXT AS schema_name,
                    namespace_row.nspname::TEXT AS object_name,
                    ''::TEXT AS identity_arguments,
                    managed_role.rolname AS grantee,
                    schema_acl.privilege_type,
                    schema_acl.is_grantable
                FROM pg_catalog.pg_namespace namespace_row
                CROSS JOIN LATERAL pg_catalog.aclexplode(namespace_row.nspacl) schema_acl
                JOIN managed_roles managed_role
                  ON managed_role.oid = schema_acl.grantee
                WHERE namespace_row.nspname <> 'np'
                UNION ALL
                SELECT
                    'relation'::TEXT,
                    namespace_row.nspname::TEXT,
                    relation_row.relname::TEXT,
                    ''::TEXT,
                    managed_role.rolname,
                    relation_acl.privilege_type,
                    relation_acl.is_grantable
                FROM pg_catalog.pg_class relation_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = relation_row.relnamespace
                CROSS JOIN LATERAL pg_catalog.aclexplode(relation_row.relacl) relation_acl
                JOIN managed_roles managed_role
                  ON managed_role.oid = relation_acl.grantee
                WHERE namespace_row.nspname <> 'np'
                UNION ALL
                SELECT
                    'function'::TEXT,
                    namespace_row.nspname::TEXT,
                    function_row.proname::TEXT,
                    pg_catalog.pg_get_function_identity_arguments(function_row.oid),
                    managed_role.rolname,
                    function_acl.privilege_type,
                    function_acl.is_grantable
                FROM pg_catalog.pg_proc function_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = function_row.pronamespace
                CROSS JOIN LATERAL pg_catalog.aclexplode(function_row.proacl) function_acl
                JOIN managed_roles managed_role
                  ON managed_role.oid = function_acl.grantee
                WHERE namespace_row.nspname <> 'np'
            )
            SELECT jsonb_agg(
                to_jsonb(acl_evidence)
                ORDER BY object_kind, schema_name, object_name,
                    identity_arguments, grantee, privilege_type
            )
            FROM acl_evidence
        ), '[]'::jsonb),
        'migrations', COALESCE((
            SELECT jsonb_agg(to_jsonb(evidence_row) ORDER BY version)
            FROM (
                SELECT version, name, checksum
                FROM np.schema_migrations
            ) evidence_row
        ), '[]'::jsonb),
        'relations', COALESCE((
            SELECT jsonb_agg(to_jsonb(evidence_row) ORDER BY relation_name)
            FROM (
                SELECT
                    relation_row.relname AS relation_name,
                    relation_row.relkind,
                    relation_row.relpersistence,
                    pg_catalog.pg_get_userbyid(relation_row.relowner) AS owner,
                    relation_row.relrowsecurity,
                    relation_row.relforcerowsecurity,
                    relation_row.relhassubclass,
                    relation_row.relispartition,
                    relation_row.relhasrules,
                    relation_row.relhastriggers,
                    COALESCE((
                        SELECT jsonb_agg(
                            jsonb_build_array(
                                COALESCE(grantee_role.rolname, 'PUBLIC'),
                                relation_acl.privilege_type,
                                relation_acl.is_grantable
                            ) ORDER BY
                                COALESCE(grantee_role.rolname, 'PUBLIC'),
                                relation_acl.privilege_type
                        )
                        FROM pg_catalog.aclexplode(
                            COALESCE(
                                relation_row.relacl,
                                pg_catalog.acldefault(
                                    CASE relation_row.relkind
                                        WHEN 'S' THEN 'S'::"char"
                                        ELSE 'r'::"char"
                                    END,
                                    relation_row.relowner
                                )
                            )) relation_acl
                        LEFT JOIN pg_catalog.pg_roles grantee_role
                          ON grantee_role.oid = relation_acl.grantee
                    ), '[]'::jsonb) AS acl
                FROM pg_catalog.pg_class relation_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = relation_row.relnamespace
                WHERE namespace_row.nspname = 'np'
                  AND relation_row.relkind IN ('r', 'S')
            ) evidence_row
        ), '[]'::jsonb),
        'inheritance', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY parent_schema, parent_relation,
                    child_schema, child_relation, inheritance_sequence
            )
            FROM (
                SELECT
                    parent_namespace.nspname AS parent_schema,
                    parent_relation.relname AS parent_relation,
                    child_namespace.nspname AS child_schema,
                    child_relation.relname AS child_relation,
                    inheritance_row.inhseqno AS inheritance_sequence
                FROM pg_catalog.pg_inherits inheritance_row
                JOIN pg_catalog.pg_class parent_relation
                  ON parent_relation.oid = inheritance_row.inhparent
                JOIN pg_catalog.pg_namespace parent_namespace
                  ON parent_namespace.oid = parent_relation.relnamespace
                JOIN pg_catalog.pg_class child_relation
                  ON child_relation.oid = inheritance_row.inhrelid
                JOIN pg_catalog.pg_namespace child_namespace
                  ON child_namespace.oid = child_relation.relnamespace
                WHERE parent_namespace.nspname = 'np'
                   OR child_namespace.nspname = 'np'
            ) evidence_row
        ), '[]'::jsonb),
        'columns', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY relation_name, ordinal_position
            )
            FROM (
                SELECT
                    relation_row.relname AS relation_name,
                    attribute_row.attnum AS ordinal_position,
                    attribute_row.attname AS column_name,
                    pg_catalog.format_type(
                        attribute_row.atttypid,
                        attribute_row.atttypmod
                    ) AS data_type,
                    attribute_row.attnotnull,
                    COALESCE(
                        pg_catalog.pg_get_expr(default_row.adbin, default_row.adrelid),
                        ''
                    ) AS column_default,
                    CASE
                        WHEN attribute_row.attcollation = 0 THEN ''
                        ELSE collation_namespace.nspname || '.' || collation_row.collname
                    END AS collation_name,
                    COALESCE((
                        SELECT jsonb_agg(
                            jsonb_build_array(
                                COALESCE(grantee_role.rolname, 'PUBLIC'),
                                column_acl.privilege_type,
                                column_acl.is_grantable
                            ) ORDER BY
                                COALESCE(grantee_role.rolname, 'PUBLIC'),
                                column_acl.privilege_type
                        )
                        FROM pg_catalog.aclexplode(attribute_row.attacl) column_acl
                        LEFT JOIN pg_catalog.pg_roles grantee_role
                          ON grantee_role.oid = column_acl.grantee
                    ), '[]'::jsonb) AS acl
                FROM pg_catalog.pg_attribute attribute_row
                JOIN pg_catalog.pg_class relation_row
                  ON relation_row.oid = attribute_row.attrelid
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = relation_row.relnamespace
                LEFT JOIN pg_catalog.pg_attrdef default_row
                  ON default_row.adrelid = attribute_row.attrelid
                 AND default_row.adnum = attribute_row.attnum
                LEFT JOIN pg_catalog.pg_collation collation_row
                  ON collation_row.oid = attribute_row.attcollation
                LEFT JOIN pg_catalog.pg_namespace collation_namespace
                  ON collation_namespace.oid = collation_row.collnamespace
                WHERE namespace_row.nspname = 'np'
                  AND relation_row.relkind = 'r'
                  AND attribute_row.attnum > 0
                  AND NOT attribute_row.attisdropped
            ) evidence_row
        ), '[]'::jsonb),
        'constraints', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY relation_name, constraint_name
            )
            FROM (
                SELECT
                    relation_row.relname AS relation_name,
                    constraint_row.conname AS constraint_name,
                    constraint_row.contype,
                    pg_catalog.pg_get_constraintdef(constraint_row.oid, true) AS definition,
                    constraint_row.condeferrable,
                    constraint_row.condeferred,
                    constraint_row.convalidated
                FROM pg_catalog.pg_constraint constraint_row
                JOIN pg_catalog.pg_class relation_row
                  ON relation_row.oid = constraint_row.conrelid
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = relation_row.relnamespace
                WHERE namespace_row.nspname = 'np'
            ) evidence_row
        ), '[]'::jsonb),
        'indexes', COALESCE((
            SELECT jsonb_agg(to_jsonb(evidence_row) ORDER BY index_name)
            FROM (
                SELECT
                    index_row.relname AS index_name,
                    table_row.relname AS relation_name,
                    pg_catalog.pg_get_indexdef(index_row.oid) AS definition,
                    index_metadata.indisunique,
                    index_metadata.indisprimary,
                    index_metadata.indisvalid,
                    index_metadata.indisready,
                    index_metadata.indislive,
                    index_metadata.indisclustered,
                    index_metadata.indisreplident,
                    index_metadata.indnullsnotdistinct
                FROM pg_catalog.pg_index index_metadata
                JOIN pg_catalog.pg_class index_row
                  ON index_row.oid = index_metadata.indexrelid
                JOIN pg_catalog.pg_class table_row
                  ON table_row.oid = index_metadata.indrelid
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = table_row.relnamespace
                WHERE namespace_row.nspname = 'np'
            ) evidence_row
        ), '[]'::jsonb),
        'rules', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY relation_name, rule_name
            )
            FROM (
                SELECT
                    relation_row.relname AS relation_name,
                    rewrite_row.rulename AS rule_name,
                    rewrite_row.ev_type,
                    rewrite_row.ev_enabled,
                    rewrite_row.is_instead,
                    pg_catalog.pg_get_ruledef(rewrite_row.oid, true) AS definition
                FROM pg_catalog.pg_rewrite rewrite_row
                JOIN pg_catalog.pg_class relation_row
                  ON relation_row.oid = rewrite_row.ev_class
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = relation_row.relnamespace
                WHERE namespace_row.nspname = 'np'
            ) evidence_row
        ), '[]'::jsonb),
        'policies', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY relation_name, policy_name
            )
            FROM (
                SELECT
                    relation_row.relname AS relation_name,
                    policy_row.polname AS policy_name,
                    policy_row.polcmd,
                    policy_row.polpermissive,
                    ARRAY(
                        SELECT role_row.rolname
                        FROM unnest(policy_row.polroles) role_oid
                        JOIN pg_catalog.pg_roles role_row ON role_row.oid = role_oid
                        ORDER BY role_row.rolname
                    ) AS roles,
                    COALESCE(
                        pg_catalog.pg_get_expr(policy_row.polqual, policy_row.polrelid),
                        ''
                    ) AS using_expression,
                    COALESCE(
                        pg_catalog.pg_get_expr(
                            policy_row.polwithcheck,
                            policy_row.polrelid
                        ),
                        ''
                    ) AS check_expression
                FROM pg_catalog.pg_policy policy_row
                JOIN pg_catalog.pg_class relation_row
                  ON relation_row.oid = policy_row.polrelid
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = relation_row.relnamespace
                WHERE namespace_row.nspname = 'np'
            ) evidence_row
        ), '[]'::jsonb),
        'triggers', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY relation_name, trigger_name
            )
            FROM (
                SELECT
                    relation_row.relname AS relation_name,
                    trigger_row.tgname AS trigger_name,
                    trigger_row.tgenabled,
                    pg_catalog.pg_get_triggerdef(trigger_row.oid, true) AS definition
                FROM pg_catalog.pg_trigger trigger_row
                JOIN pg_catalog.pg_class relation_row
                  ON relation_row.oid = trigger_row.tgrelid
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = relation_row.relnamespace
                WHERE namespace_row.nspname = 'np'
                  AND NOT trigger_row.tgisinternal
            ) evidence_row
        ), '[]'::jsonb),
        'functions', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY function_name, identity_arguments
            )
            FROM (
                SELECT
                    function_row.proname AS function_name,
                    pg_catalog.pg_get_function_identity_arguments(function_row.oid)
                        AS identity_arguments,
                    pg_catalog.pg_get_userbyid(function_row.proowner) AS owner,
                    function_row.prosecdef,
                    COALESCE(function_row.proconfig, ARRAY[]::TEXT[]) AS settings,
                    language_row.lanname AS language_name,
                    function_row.prokind,
                    function_row.provolatile,
                    function_row.proisstrict,
                    function_row.proparallel,
                    COALESCE((
                        SELECT jsonb_agg(
                            jsonb_build_array(
                                COALESCE(grantee_role.rolname, 'PUBLIC'),
                                function_acl.privilege_type,
                                function_acl.is_grantable
                            ) ORDER BY
                                COALESCE(grantee_role.rolname, 'PUBLIC'),
                                function_acl.privilege_type
                        )
                        FROM pg_catalog.aclexplode(
                            COALESCE(
                                function_row.proacl,
                                pg_catalog.acldefault('f', function_row.proowner)
                            )
                        ) function_acl
                        LEFT JOIN pg_catalog.pg_roles grantee_role
                          ON grantee_role.oid = function_acl.grantee
                    ), '[]'::jsonb) AS acl,
                    pg_catalog.pg_get_functiondef(function_row.oid) AS definition
                FROM pg_catalog.pg_proc function_row
                JOIN pg_catalog.pg_namespace namespace_row
                  ON namespace_row.oid = function_row.pronamespace
                JOIN pg_catalog.pg_language language_row
                  ON language_row.oid = function_row.prolang
                WHERE namespace_row.nspname = 'np'
            ) evidence_row
        ), '[]'::jsonb),
        'schema', (
            SELECT jsonb_build_object(
                'owner', pg_catalog.pg_get_userbyid(namespace_row.nspowner),
                'acl', COALESCE((
                    SELECT jsonb_agg(
                        jsonb_build_array(
                            COALESCE(grantee_role.rolname, 'PUBLIC'),
                            schema_acl.privilege_type,
                            schema_acl.is_grantable
                        ) ORDER BY
                            COALESCE(grantee_role.rolname, 'PUBLIC'),
                            schema_acl.privilege_type
                    )
                    FROM pg_catalog.aclexplode(
                        COALESCE(
                            namespace_row.nspacl,
                            pg_catalog.acldefault('n', namespace_row.nspowner)
                        )
                    ) schema_acl
                    LEFT JOIN pg_catalog.pg_roles grantee_role
                      ON grantee_role.oid = schema_acl.grantee
                ), '[]'::jsonb)
            )
            FROM pg_catalog.pg_namespace namespace_row
            WHERE namespace_row.nspname = 'np'
        ),
        'roles', COALESCE((
            SELECT jsonb_agg(to_jsonb(evidence_row) ORDER BY role_name)
            FROM (
                SELECT
                    role_row.rolname AS role_name,
                    role_row.rolcanlogin,
                    role_row.rolsuper,
                    role_row.rolinherit,
                    role_row.rolcreaterole,
                    role_row.rolcreatedb,
                    role_row.rolreplication,
                    role_row.rolbypassrls,
                    role_row.rolconnlimit,
                    COALESCE(role_row.rolconfig, ARRAY[]::TEXT[]) AS settings,
                    pg_catalog.shobj_description(role_row.oid, 'pg_authid') AS marker
                FROM pg_catalog.pg_roles role_row
                WHERE pg_catalog.strpos(
                    pg_catalog.shobj_description(role_row.oid, 'pg_authid'),
                    'elvis-postgres-bootstrap:v2:'
                    || pg_catalog.current_database() || ':'
                ) = 1
            ) evidence_row
        ), '[]'::jsonb),
        'memberships', COALESCE((
            SELECT jsonb_agg(
                to_jsonb(evidence_row)
                ORDER BY parent_role, member_role
            )
            FROM (
                SELECT
                    parent_role.rolname AS parent_role,
                    member_role.rolname AS member_role,
                    membership.admin_option
                FROM pg_catalog.pg_auth_members membership
                JOIN pg_catalog.pg_roles parent_role
                  ON parent_role.oid = membership.roleid
                JOIN pg_catalog.pg_roles member_role
                  ON member_role.oid = membership.member
                WHERE pg_catalog.strpos(
                        pg_catalog.shobj_description(parent_role.oid, 'pg_authid'),
                        'elvis-postgres-bootstrap:v2:'
                        || pg_catalog.current_database() || ':'
                      ) = 1
                   OR pg_catalog.strpos(
                        pg_catalog.shobj_description(member_role.oid, 'pg_authid'),
                        'elvis-postgres-bootstrap:v2:'
                        || pg_catalog.current_database() || ':'
                      ) = 1
            ) evidence_row
        ), '[]'::jsonb)
    ) INTO evidence;

    RETURN encode(sha256(convert_to(evidence::TEXT, 'UTF8')), 'hex');
END
$function$;

REVOKE ALL
ON FUNCTION np.paper_terminal_catalog_fingerprint()
FROM PUBLIC;

CREATE FUNCTION np.paper_canonical_json(payload JSONB)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
STRICT
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    result TEXT;
BEGIN
    CASE jsonb_typeof(payload)
        WHEN 'object' THEN
            SELECT '{' || COALESCE(
                string_agg(
                    to_jsonb(item.key)::TEXT || ':'
                    || np.paper_canonical_json(item.value),
                    ',' ORDER BY item.key COLLATE pg_catalog."C"
                ),
                ''
            ) || '}'
            INTO result
            FROM jsonb_each(payload) item;
        WHEN 'array' THEN
            SELECT '[' || COALESCE(
                string_agg(
                    np.paper_canonical_json(item.value),
                    ',' ORDER BY item.ordinality
                ),
                ''
            ) || ']'
            INTO result
            FROM jsonb_array_elements(payload) WITH ORDINALITY item(value, ordinality);
        ELSE
            result := payload::TEXT;
    END CASE;
    RETURN result;
END
$function$;

REVOKE ALL
ON FUNCTION np.paper_canonical_json(JSONB)
FROM PUBLIC;

CREATE FUNCTION np.paper_sha256_text(payload TEXT)
RETURNS TEXT
LANGUAGE SQL
IMMUTABLE
STRICT
SET search_path = pg_catalog, pg_temp
AS $function$
    SELECT encode(sha256(convert_to(payload, 'UTF8')), 'hex')
$function$;

REVOKE ALL
ON FUNCTION np.paper_sha256_text(TEXT)
FROM PUBLIC;

CREATE FUNCTION np.paper_sha256_fresh_opening_intent(payload TEXT)
RETURNS TEXT
LANGUAGE SQL
IMMUTABLE
STRICT
SET search_path = pg_catalog, pg_temp
AS $function$
    SELECT encode(
        sha256(
            convert_to('ELVIS', 'UTF8')
            || decode('00', 'hex')
            || convert_to('fresh-opening-intent', 'UTF8')
            || decode('00', 'hex')
            || convert_to('v1', 'UTF8')
            || decode('00', 'hex')
            || convert_to(payload, 'UTF8')
        ),
        'hex'
    )
$function$;

REVOKE ALL
ON FUNCTION np.paper_sha256_fresh_opening_intent(TEXT)
FROM PUBLIC;

CREATE FUNCTION np.paper_fresh_opening_database_incarnation(
    target_database_name TEXT,
    target_system_identifier NUMERIC,
    target_migration_version INTEGER,
    target_migration_name TEXT,
    target_migration_checksum TEXT,
    target_terminal_catalog_sha256 TEXT,
    target_control_plane_role TEXT,
    target_opening_anchor_role TEXT,
    target_deployment_incarnation_id TEXT
)
RETURNS TEXT
LANGUAGE SQL
IMMUTABLE
STRICT
SET search_path = pg_catalog, pg_temp
AS $function$
    SELECT np.paper_sha256_text(
        np.paper_canonical_json(
            jsonb_build_object(
                'database_name', target_database_name,
                'deployment_incarnation_id', target_deployment_incarnation_id,
                'migration_checksum', target_migration_checksum,
                'migration_head', target_migration_version,
                'migration_name', target_migration_name,
                'control_plane_role', target_control_plane_role,
                'opening_anchor_role', target_opening_anchor_role,
                'system_identifier', target_system_identifier::TEXT,
                'terminal_catalog_sha256', target_terminal_catalog_sha256
            )
        )
    )
$function$;

REVOKE ALL
ON FUNCTION np.paper_fresh_opening_database_incarnation(
    TEXT,
    NUMERIC,
    INTEGER,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT
)
FROM PUBLIC;

CREATE FUNCTION np.paper_fresh_opening_target_is_current()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    provisioning_row np.paper_fresh_opening_provisionings%ROWTYPE;
    admission_row np.paper_fresh_opening_admissions%ROWTYPE;
    stream_row np.paper_account_streams%ROWTYPE;
    admission_document JSONB;
    control_plane_row RECORD;
    opening_anchor_row RECORD;
    migration_row RECORD;
    migration_count BIGINT;
    observed_system_identifier NUMERIC;
    schema_marker TEXT;
    live_catalog_sha256 TEXT;
    derived_database_incarnation_id TEXT;
BEGIN
    SELECT *
    INTO STRICT provisioning_row
    FROM np.paper_fresh_opening_provisionings
    WHERE control_key IS TRUE;

    SELECT *
    INTO STRICT admission_row
    FROM np.paper_fresh_opening_admissions
    WHERE control_key IS TRUE;

    SELECT *
    INTO STRICT stream_row
    FROM np.paper_account_streams
    WHERE execution_scope = provisioning_row.execution_scope
      AND account_key = provisioning_row.account_key
      AND owner_generation = provisioning_row.owner_generation
      AND opening_version = provisioning_row.opening_version
      AND opening_payload_sha256 = provisioning_row.opening_payload_sha256;

    BEGIN
        admission_document := admission_row.admission_payload::JSONB;
    EXCEPTION
        WHEN invalid_text_representation THEN
            RETURN FALSE;
    END;

    SELECT
        role_record.rolname,
        role_record.rolcanlogin,
        role_record.rolsuper,
        pg_catalog.pg_get_userbyid(database_record.datdba) AS database_owner
    INTO STRICT control_plane_row
    FROM pg_catalog.pg_roles role_record
    JOIN pg_catalog.pg_database database_record
      ON database_record.datname = pg_catalog.current_database()
    WHERE role_record.rolname = provisioning_row.control_plane_role;

    SELECT
        role_record.rolname,
        role_record.rolcanlogin,
        role_record.rolsuper,
        role_record.rolinherit,
        role_record.rolcreaterole,
        role_record.rolcreatedb,
        role_record.rolreplication,
        role_record.rolbypassrls,
        role_record.rolconnlimit,
        role_record.rolconfig,
        pg_catalog.shobj_description(role_record.oid, 'pg_authid') AS marker
    INTO STRICT opening_anchor_row
    FROM pg_catalog.pg_roles role_record
    WHERE role_record.rolname = provisioning_row.opening_anchor_role;

    SELECT version, name, checksum
    INTO STRICT migration_row
    FROM np.schema_migrations
    ORDER BY version DESC
    LIMIT 1;

    SELECT COUNT(*)
    INTO STRICT migration_count
    FROM np.schema_migrations;

    SELECT control_system.system_identifier::NUMERIC
    INTO STRICT observed_system_identifier
    FROM pg_catalog.pg_control_system() control_system;

    SELECT pg_catalog.obj_description(namespace_row.oid, 'pg_namespace')
    INTO STRICT schema_marker
    FROM pg_catalog.pg_namespace namespace_row
    WHERE namespace_row.nspname = 'np';

    live_catalog_sha256 := np.paper_terminal_catalog_fingerprint();
    derived_database_incarnation_id := (
        np.paper_fresh_opening_database_incarnation(
            pg_catalog.current_database()::TEXT,
            observed_system_identifier,
            migration_row.version,
            migration_row.name::TEXT,
            migration_row.checksum::TEXT,
            live_catalog_sha256,
            control_plane_row.rolname::TEXT,
            opening_anchor_row.rolname::TEXT,
            admission_row.deployment_incarnation_id::TEXT
        )
    );

    IF admission_row.candidate_payload_sha256 IS DISTINCT FROM (
            provisioning_row.candidate_payload_sha256
       )
       OR stream_row.execution_scope IS DISTINCT FROM (
            provisioning_row.execution_scope
       )
       OR stream_row.account_key IS DISTINCT FROM provisioning_row.account_key
       OR stream_row.owner_generation IS DISTINCT FROM (
            provisioning_row.owner_generation
       )
       OR stream_row.collateral_asset IS DISTINCT FROM (
            provisioning_row.collateral_asset
       )
       OR stream_row.opening_version IS DISTINCT FROM (
            provisioning_row.opening_version
       )
       OR np.paper_canonical_json(stream_row.opening_payload) IS DISTINCT FROM (
            provisioning_row.opening_payload
       )
       OR stream_row.opening_payload_sha256 IS DISTINCT FROM (
            provisioning_row.opening_payload_sha256
       )
       OR admission_row.pin_authority_record_sha256 IS DISTINCT FROM (
            provisioning_row.pin_authority_record_sha256
       )
       OR admission_row.deployment_incarnation_id IS DISTINCT FROM (
            provisioning_row.deployment_incarnation_id
       )
       OR jsonb_typeof(admission_document) IS DISTINCT FROM 'object'
       OR ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(admission_document) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) IS DISTINCT FROM ARRAY[
            'candidate_sha256',
            'deployment_incarnation_id',
            'pin_authority_record_sha256',
            'schema_version'
       ]::TEXT[]
       OR admission_document->'schema_version' IS DISTINCT FROM '1'::JSONB
       OR admission_document->'candidate_sha256' IS DISTINCT FROM to_jsonb(
            admission_row.candidate_payload_sha256::TEXT
       )
       OR admission_document->'pin_authority_record_sha256' IS DISTINCT FROM (
            to_jsonb(admission_row.pin_authority_record_sha256::TEXT)
       )
       OR admission_document->'deployment_incarnation_id' IS DISTINCT FROM (
            to_jsonb(admission_row.deployment_incarnation_id::TEXT)
       )
       OR np.paper_canonical_json(admission_document) IS DISTINCT FROM (
            admission_row.admission_payload
       )
       OR np.paper_sha256_text(admission_row.admission_payload) IS DISTINCT FROM (
            admission_row.admission_payload_sha256
       )
       OR np.paper_sha256_text(provisioning_row.candidate_payload) IS DISTINCT FROM (
            provisioning_row.candidate_payload_sha256
       )
       OR np.paper_sha256_text(provisioning_row.opening_payload) IS DISTINCT FROM (
            provisioning_row.opening_payload_sha256
       )
       OR pg_catalog.current_database()::TEXT IS DISTINCT FROM provisioning_row.database_name
       OR observed_system_identifier IS DISTINCT FROM provisioning_row.system_identifier
       OR (observed_system_identifier BETWEEN 1 AND 18446744073709551615) IS NOT TRUE
       OR control_plane_row.rolname IS DISTINCT FROM provisioning_row.control_plane_role
       OR control_plane_row.rolcanlogin IS DISTINCT FROM TRUE
       OR control_plane_row.rolsuper IS DISTINCT FROM TRUE
       OR control_plane_row.database_owner IS DISTINCT FROM (
            provisioning_row.control_plane_role
       )
       OR opening_anchor_row.rolname IS DISTINCT FROM (
            provisioning_row.opening_anchor_role
       )
       OR opening_anchor_row.marker IS DISTINCT FROM (
            'elvis-postgres-bootstrap:v2:'
            || pg_catalog.current_database()
            || ':opening:'
            || admission_row.admission_payload_sha256
       )
       OR opening_anchor_row.rolcanlogin IS DISTINCT FROM FALSE
       OR opening_anchor_row.rolsuper IS DISTINCT FROM FALSE
       OR opening_anchor_row.rolinherit IS DISTINCT FROM FALSE
       OR opening_anchor_row.rolcreaterole IS DISTINCT FROM FALSE
       OR opening_anchor_row.rolcreatedb IS DISTINCT FROM FALSE
       OR opening_anchor_row.rolreplication IS DISTINCT FROM FALSE
       OR opening_anchor_row.rolbypassrls IS DISTINCT FROM FALSE
       OR opening_anchor_row.rolconnlimit IS DISTINCT FROM -1
       OR opening_anchor_row.rolconfig IS NOT NULL
       OR migration_count IS DISTINCT FROM 7::BIGINT
       OR migration_row.version IS DISTINCT FROM provisioning_row.migration_version
       OR migration_row.name IS DISTINCT FROM provisioning_row.migration_name
       OR migration_row.checksum IS DISTINCT FROM provisioning_row.migration_checksum
       OR live_catalog_sha256 IS DISTINCT FROM (
            provisioning_row.terminal_catalog_sha256
       )
       OR schema_marker IS DISTINCT FROM (
            'elvis-postgres-bootstrap-schema:v2:'
            || pg_catalog.current_database()
            || ':'
            || live_catalog_sha256
       )
       OR derived_database_incarnation_id IS DISTINCT FROM (
            provisioning_row.database_incarnation_id
       )
       OR provisioning_row.runtime_mode IS DISTINCT FROM 'LEGACY'
       OR provisioning_row.runtime_generation IS DISTINCT FROM 0::BIGINT
       OR provisioning_row.authority_transition_sequence IS DISTINCT FROM 0::BIGINT
       OR provisioning_row.writer_fence IS DISTINCT FROM 0::BIGINT
       OR provisioning_row.runtime_activation_authorized IS DISTINCT FROM FALSE
       OR provisioning_row.trading_authorized IS DISTINCT FROM FALSE
       OR provisioning_row.stale_on_return IS DISTINCT FROM TRUE
       OR np.paper_sha256_text(
            provisioning_row.provisioning_receipt_payload
       ) IS DISTINCT FROM (
            provisioning_row.provisioning_receipt_payload_sha256
       ) THEN
        RETURN FALSE;
    END IF;

    RETURN TRUE;
EXCEPTION
    WHEN NO_DATA_FOUND OR TOO_MANY_ROWS THEN
        RETURN FALSE;
END
$function$;

REVOKE ALL
ON FUNCTION np.paper_fresh_opening_target_is_current()
FROM PUBLIC;

CREATE FUNCTION np.require_current_paper_fresh_opening_provenance()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    provisioning_row np.paper_fresh_opening_provisionings%ROWTYPE;
BEGIN
    IF TG_OP IS DISTINCT FROM 'INSERT' THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper runtime generation provenance guard is invalid';
    END IF;

    SELECT *
    INTO STRICT provisioning_row
    FROM np.paper_fresh_opening_provisionings
    WHERE control_key IS TRUE
    FOR KEY SHARE;

    IF NEW.execution_scope IS DISTINCT FROM provisioning_row.execution_scope
       OR NEW.account_key IS DISTINCT FROM provisioning_row.account_key
       OR NEW.owner_generation IS DISTINCT FROM provisioning_row.owner_generation
       OR NEW.opening_version IS DISTINCT FROM provisioning_row.opening_version
       OR NEW.opening_payload_sha256 IS DISTINCT FROM (
            provisioning_row.opening_payload_sha256
       )
       OR np.paper_fresh_opening_target_is_current() IS NOT TRUE THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening current physical target is invalid';
    END IF;

    RETURN NEW;
EXCEPTION
    WHEN NO_DATA_FOUND OR TOO_MANY_ROWS THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening current physical target is unavailable';
END
$function$;

REVOKE ALL
ON FUNCTION np.require_current_paper_fresh_opening_provenance()
FROM PUBLIC;

CREATE TRIGGER paper_runtime_generations_require_fresh_opening_provenance
BEFORE INSERT
ON np.paper_runtime_generations
FOR EACH ROW
EXECUTE FUNCTION np.require_current_paper_fresh_opening_provenance();

ALTER TABLE np.paper_runtime_generations
ENABLE ALWAYS TRIGGER paper_runtime_generations_require_fresh_opening_provenance;

CREATE FUNCTION np.acquire_paper_fresh_opening_fence(
    requested_trust_domain TEXT,
    requested_signer_key_id TEXT,
    requested_nonce TEXT,
    requested_candidate_payload_sha256 TEXT
)
RETURNS TABLE(
    resolution TEXT,
    evaluated_at TIMESTAMPTZ,
    database_name TEXT,
    system_identifier NUMERIC,
    control_plane_role TEXT,
    opening_anchor_role TEXT,
    migration_version INTEGER,
    migration_name TEXT,
    migration_checksum TEXT,
    terminal_catalog_sha256 TEXT,
    pin_authority_record_sha256 TEXT,
    deployment_incarnation_id TEXT,
    database_incarnation_id TEXT,
    runtime_mode TEXT,
    runtime_generation BIGINT,
    authority_transition_sequence BIGINT,
    writer_fence BIGINT,
    v2_empty BOOLEAN,
    stored_authority_evaluated_at TIMESTAMPTZ,
    committed_at TIMESTAMPTZ,
    intent_payload TEXT,
    intent_sha256 TEXT,
    approval_payload TEXT,
    approval_sha256 TEXT,
    trust_policy_payload TEXT,
    trust_policy_sha256 TEXT,
    candidate_payload TEXT,
    candidate_sha256 TEXT,
    opening_payload TEXT,
    opening_sha256 TEXT,
    opening_receipt_payload TEXT,
    opening_receipt_sha256 TEXT,
    provisioning_receipt_payload TEXT,
    provisioning_receipt_sha256 TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    role_row RECORD;
    opening_anchor_row RECORD;
    migration_row RECORD;
    control_row RECORD;
    admission_row np.paper_fresh_opening_admissions%ROWTYPE;
    admission_document JSONB;
    stored_row np.paper_fresh_opening_provisionings%ROWTYPE;
    nonce_candidate_sha256 TEXT;
    schema_marker TEXT;
    live_catalog_sha256 TEXT;
    observed_system_identifier NUMERIC;
    target_is_empty BOOLEAN;
    resolved TEXT;
BEGIN
    IF requested_trust_domain IS NULL
       OR requested_trust_domain = ''
       OR requested_trust_domain <> BTRIM(requested_trust_domain)
       OR LENGTH(requested_trust_domain) > 128
       OR requested_trust_domain !~ '^[!-~]+$'
       OR requested_signer_key_id IS NULL
       OR requested_signer_key_id = ''
       OR requested_signer_key_id <> BTRIM(requested_signer_key_id)
       OR LENGTH(requested_signer_key_id) > 255
       OR requested_signer_key_id !~ '^[!-~]+$'
       OR requested_nonce IS NULL
       OR requested_nonce !~ '^[0-9a-f]{64}$'
       OR requested_nonce = repeat('0', 64)
       OR requested_candidate_payload_sha256 IS NULL
       OR requested_candidate_payload_sha256 !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening fence arguments are invalid';
    END IF;

    SELECT
        role_record.rolname,
        role_record.rolcanlogin,
        role_record.rolsuper,
        role_record.rolinherit,
        role_record.rolcreaterole,
        role_record.rolcreatedb,
        role_record.rolreplication,
        role_record.rolbypassrls,
        role_record.rolconnlimit,
        role_record.rolconfig,
        pg_catalog.pg_get_userbyid(database_record.datdba) AS database_owner
    INTO STRICT role_row
    FROM pg_catalog.pg_roles role_record
    JOIN pg_catalog.pg_database database_record
      ON database_record.datname = pg_catalog.current_database()
    WHERE role_record.rolname = session_user;

    IF role_row.rolcanlogin IS NOT TRUE
       OR role_row.rolsuper IS NOT TRUE
       OR role_row.database_owner IS DISTINCT FROM session_user THEN
        RAISE EXCEPTION USING
            ERRCODE = '42501',
            MESSAGE = 'paper fresh opening control-plane identity is not admitted';
    END IF;

    -- This admission row is the ACL-scoped opening mutex.  It preserves exact
    -- concurrent replay semantics without exposing a PUBLIC advisory-lock key.
    SELECT *
    INTO STRICT admission_row
    FROM np.paper_fresh_opening_admissions
    WHERE control_key IS TRUE
    FOR UPDATE;

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
        ONLY np.paper_fresh_opening_admissions,
        ONLY np.paper_fresh_opening_nonces,
        ONLY np.paper_fresh_opening_provisionings,
        ONLY np.paper_margin_reservations,
        ONLY np.paper_runtime_control,
        ONLY np.paper_runtime_generations,
        ONLY np.position_streams,
        ONLY np.schema_migrations,
        ONLY np.trades,
        ONLY np.trading_session_resets
    IN SHARE ROW EXCLUSIVE MODE NOWAIT;

    BEGIN
        admission_document := admission_row.admission_payload::JSONB;
    EXCEPTION
        WHEN invalid_text_representation THEN
            RAISE EXCEPTION USING
                ERRCODE = '55000',
                MESSAGE = 'paper fresh opening admission is invalid';
    END;
    IF jsonb_typeof(admission_document) IS DISTINCT FROM 'object'
       OR ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(admission_document) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY[
            'candidate_sha256',
            'deployment_incarnation_id',
            'pin_authority_record_sha256',
            'schema_version'
       ]::TEXT[]
       OR admission_document->'schema_version' IS DISTINCT FROM '1'::JSONB
       OR admission_document->'candidate_sha256' IS DISTINCT FROM to_jsonb(
            admission_row.candidate_payload_sha256::TEXT
       )
       OR admission_document->'pin_authority_record_sha256' IS DISTINCT FROM (
            to_jsonb(admission_row.pin_authority_record_sha256::TEXT)
       )
       OR admission_document->'deployment_incarnation_id' IS DISTINCT FROM (
            to_jsonb(admission_row.deployment_incarnation_id)
       )
       OR np.paper_canonical_json(admission_document) IS DISTINCT FROM (
            admission_row.admission_payload
       )
       OR np.paper_sha256_text(admission_row.admission_payload) IS DISTINCT FROM (
            admission_row.admission_payload_sha256
       )
       THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening admission is invalid';
    END IF;

    SELECT
        role_record.rolname,
        role_record.rolcanlogin,
        role_record.rolsuper,
        role_record.rolinherit,
        role_record.rolcreaterole,
        role_record.rolcreatedb,
        role_record.rolreplication,
        role_record.rolbypassrls,
        role_record.rolconnlimit,
        role_record.rolconfig
    INTO STRICT opening_anchor_row
    FROM pg_catalog.pg_roles role_record
    WHERE pg_catalog.shobj_description(role_record.oid, 'pg_authid') = (
        'elvis-postgres-bootstrap:v2:'
        || pg_catalog.current_database()
        || ':opening:'
        || admission_row.admission_payload_sha256
    );

    IF opening_anchor_row.rolcanlogin IS TRUE
       OR opening_anchor_row.rolsuper IS TRUE
       OR opening_anchor_row.rolinherit IS TRUE
       OR opening_anchor_row.rolcreaterole IS TRUE
       OR opening_anchor_row.rolcreatedb IS TRUE
       OR opening_anchor_row.rolreplication IS TRUE
       OR opening_anchor_row.rolbypassrls IS TRUE
       OR opening_anchor_row.rolconnlimit <> -1
       OR opening_anchor_row.rolconfig IS NOT NULL THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening inert anchor role is invalid';
    END IF;

    SELECT version, name, checksum
    INTO STRICT migration_row
    FROM np.schema_migrations
    ORDER BY version DESC
    LIMIT 1;

    IF migration_row.version <> 7
       OR migration_row.name <> 'fresh_opening_provenance'
       OR migration_row.checksum !~ '^[0-9a-f]{64}$'
       OR (SELECT COUNT(*) FROM np.schema_migrations) <> 7 THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening migration head is not admitted';
    END IF;

    SELECT pg_catalog.obj_description(namespace_row.oid, 'pg_namespace')
    INTO STRICT schema_marker
    FROM pg_catalog.pg_namespace namespace_row
    WHERE namespace_row.nspname = 'np';

    live_catalog_sha256 := np.paper_terminal_catalog_fingerprint();
    IF schema_marker IS DISTINCT FROM (
            'elvis-postgres-bootstrap-schema:v2:'
            || pg_catalog.current_database()
            || ':'
            || live_catalog_sha256
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening terminal catalog is not admitted';
    END IF;

    SELECT control_system.system_identifier::NUMERIC
    INTO STRICT observed_system_identifier
    FROM pg_catalog.pg_control_system() control_system;
    IF observed_system_identifier NOT BETWEEN 1 AND 18446744073709551615 THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening system identifier is invalid';
    END IF;

    SELECT
        runtime_control.mode,
        runtime_control.runtime_generation
    INTO STRICT control_row
    FROM np.paper_runtime_control runtime_control
    WHERE runtime_control.control_key IS TRUE
    FOR UPDATE NOWAIT;

    IF control_row.mode <> 'LEGACY' OR control_row.runtime_generation <> 0 THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening target authority is not LEGACY/0';
    END IF;

    SELECT candidate_payload_sha256
    INTO nonce_candidate_sha256
    FROM np.paper_fresh_opening_nonces
    WHERE trust_domain = requested_trust_domain
      AND signer_key_id = requested_signer_key_id
      AND nonce = requested_nonce
    FOR UPDATE;

    SELECT *
    INTO stored_row
    FROM np.paper_fresh_opening_provisionings
    WHERE control_key IS TRUE
    FOR UPDATE;

    IF nonce_candidate_sha256 IS NOT NULL THEN
        IF nonce_candidate_sha256 = requested_candidate_payload_sha256
           AND stored_row.control_key IS TRUE
           AND stored_row.trust_domain = requested_trust_domain
           AND stored_row.signer_key_id = requested_signer_key_id
           AND stored_row.nonce = requested_nonce
           AND stored_row.candidate_payload_sha256 = (
                requested_candidate_payload_sha256
           ) THEN
            resolved := 'EXACT_REPLAY';
        ELSE
            resolved := 'NONCE_CONFLICT';
        END IF;
    ELSIF stored_row.control_key IS TRUE THEN
        resolved := 'TARGET_CONFLICT';
    ELSIF requested_candidate_payload_sha256 <> (
            admission_row.candidate_payload_sha256
       ) THEN
        resolved := 'ADMISSION_CONFLICT';
    ELSE
        resolved := 'ABSENT';
    END IF;

    SELECT NOT EXISTS (
        SELECT 1 FROM ONLY np.account_balances
        UNION ALL SELECT 1 FROM ONLY np.liquidations
        UNION ALL SELECT 1 FROM ONLY np.margin_history
        UNION ALL SELECT 1 FROM ONLY np.model_predictions
        UNION ALL SELECT 1 FROM ONLY np.open_positions
        UNION ALL SELECT 1 FROM ONLY np.order_events
        UNION ALL SELECT 1 FROM ONLY np.orders
        UNION ALL SELECT 1 FROM ONLY np.paper_account_balances
        UNION ALL SELECT 1 FROM ONLY np.paper_account_batch_manifests
        UNION ALL SELECT 1 FROM ONLY np.paper_account_postings
        UNION ALL SELECT 1 FROM ONLY np.paper_account_settlements
        UNION ALL SELECT 1 FROM ONLY np.paper_account_streams
        UNION ALL SELECT 1 FROM ONLY np.paper_fresh_opening_nonces
        UNION ALL SELECT 1 FROM ONLY np.paper_fresh_opening_provisionings
        UNION ALL SELECT 1 FROM ONLY np.paper_margin_reservations
        UNION ALL SELECT 1 FROM ONLY np.paper_runtime_generations
        UNION ALL SELECT 1 FROM ONLY np.position_streams
        UNION ALL SELECT 1 FROM ONLY np.trades
        UNION ALL SELECT 1 FROM ONLY np.trading_session_resets
    ) INTO target_is_empty;

    RETURN QUERY
    SELECT
        resolved,
        transaction_timestamp(),
        pg_catalog.current_database()::TEXT,
        observed_system_identifier,
        session_user::TEXT,
        opening_anchor_row.rolname::TEXT,
        migration_row.version,
        migration_row.name::TEXT,
        migration_row.checksum::TEXT,
        live_catalog_sha256,
        admission_row.pin_authority_record_sha256::TEXT,
        admission_row.deployment_incarnation_id::TEXT,
        stored_row.database_incarnation_id::TEXT,
        control_row.mode::TEXT,
        control_row.runtime_generation,
        0::BIGINT,
        0::BIGINT,
        target_is_empty,
        stored_row.authority_evaluated_at,
        stored_row.committed_at,
        stored_row.intent_payload::TEXT,
        stored_row.intent_payload_sha256::TEXT,
        stored_row.approval_payload::TEXT,
        stored_row.approval_payload_sha256::TEXT,
        stored_row.trust_policy_payload::TEXT,
        stored_row.trust_policy_payload_sha256::TEXT,
        stored_row.candidate_payload::TEXT,
        stored_row.candidate_payload_sha256::TEXT,
        stored_row.opening_payload::TEXT,
        stored_row.opening_payload_sha256::TEXT,
        stored_row.opening_receipt_payload::TEXT,
        stored_row.opening_receipt_payload_sha256::TEXT,
        stored_row.provisioning_receipt_payload::TEXT,
        stored_row.provisioning_receipt_payload_sha256::TEXT;
EXCEPTION
    WHEN NO_DATA_FOUND OR TOO_MANY_ROWS THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening target identity is unavailable';
END
$function$;

REVOKE ALL
ON FUNCTION np.acquire_paper_fresh_opening_fence(TEXT, TEXT, TEXT, TEXT)
FROM PUBLIC;

CREATE FUNCTION np.commit_paper_fresh_opening(
    requested_intent_payload TEXT,
    requested_intent_payload_sha256 TEXT,
    requested_approval_payload TEXT,
    requested_approval_payload_sha256 TEXT,
    requested_trust_policy_payload TEXT,
    requested_trust_policy_payload_sha256 TEXT,
    requested_candidate_payload TEXT,
    requested_candidate_payload_sha256 TEXT,
    requested_opening_payload TEXT,
    requested_opening_payload_sha256 TEXT,
    requested_opening_receipt_payload TEXT,
    requested_opening_receipt_payload_sha256 TEXT,
    requested_provisioning_receipt_payload TEXT,
    requested_provisioning_receipt_payload_sha256 TEXT
)
RETURNS TABLE(
    disposition TEXT,
    committed_at TIMESTAMPTZ,
    opening_receipt_payload TEXT,
    opening_receipt_sha256 TEXT,
    provisioning_receipt_payload TEXT,
    provisioning_receipt_sha256 TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    intent_document JSONB;
    approval_document JSONB;
    trust_policy_document JSONB;
    candidate_document JSONB;
    opening_document JSONB;
    opening_receipt_document JSONB;
    provisioning_receipt_document JSONB;
    opening_balance JSONB;
    opening_policy JSONB;
    matching_anchor JSONB;
    fence_row RECORD;
    approval_issued_at TIMESTAMPTZ;
    approval_expires_at TIMESTAMPTZ;
    max_lifetime_seconds INTEGER;
    collateral_amount NUMERIC;
    margin_quantum NUMERIC;
    derived_database_incarnation_id TEXT;
    commit_timestamp TIMESTAMPTZ;
BEGIN
    IF requested_intent_payload IS NULL
       OR requested_approval_payload IS NULL
       OR requested_trust_policy_payload IS NULL
       OR requested_candidate_payload IS NULL
       OR requested_opening_payload IS NULL
       OR requested_opening_receipt_payload IS NULL
       OR requested_provisioning_receipt_payload IS NULL
       OR requested_intent_payload_sha256 IS NULL
       OR requested_approval_payload_sha256 IS NULL
       OR requested_trust_policy_payload_sha256 IS NULL
       OR requested_candidate_payload_sha256 IS NULL
       OR requested_opening_payload_sha256 IS NULL
       OR requested_opening_receipt_payload_sha256 IS NULL
       OR requested_provisioning_receipt_payload_sha256 IS NULL
       OR requested_intent_payload_sha256 !~ '^[0-9a-f]{64}$'
       OR requested_approval_payload_sha256 !~ '^[0-9a-f]{64}$'
       OR requested_trust_policy_payload_sha256 !~ '^[0-9a-f]{64}$'
       OR requested_candidate_payload_sha256 !~ '^[0-9a-f]{64}$'
       OR requested_opening_payload_sha256 !~ '^[0-9a-f]{64}$'
       OR requested_opening_receipt_payload_sha256 !~ '^[0-9a-f]{64}$'
       OR requested_provisioning_receipt_payload_sha256 !~ '^[0-9a-f]{64}$'
       OR OCTET_LENGTH(requested_intent_payload) > 65536
       OR OCTET_LENGTH(requested_approval_payload) > 65536
       OR OCTET_LENGTH(requested_trust_policy_payload) > 65536
       OR OCTET_LENGTH(requested_candidate_payload) > 65536
       OR OCTET_LENGTH(requested_opening_payload) > 65536
       OR OCTET_LENGTH(requested_opening_receipt_payload) > 65536
       OR OCTET_LENGTH(requested_provisioning_receipt_payload) > 65536 THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening documents are invalid';
    END IF;

    BEGIN
        intent_document := requested_intent_payload::JSONB;
        approval_document := requested_approval_payload::JSONB;
        trust_policy_document := requested_trust_policy_payload::JSONB;
        candidate_document := requested_candidate_payload::JSONB;
        opening_document := requested_opening_payload::JSONB;
        opening_receipt_document := requested_opening_receipt_payload::JSONB;
        provisioning_receipt_document := (
            requested_provisioning_receipt_payload::JSONB
        );
    EXCEPTION
        WHEN invalid_text_representation THEN
            RAISE EXCEPTION USING
                ERRCODE = '22023',
                MESSAGE = 'paper fresh opening documents are not JSON';
    END;

    IF jsonb_typeof(intent_document) IS DISTINCT FROM 'object'
       OR jsonb_typeof(approval_document) IS DISTINCT FROM 'object'
       OR jsonb_typeof(trust_policy_document) IS DISTINCT FROM 'object'
       OR jsonb_typeof(candidate_document) IS DISTINCT FROM 'object'
       OR jsonb_typeof(opening_document) IS DISTINCT FROM 'object'
       OR jsonb_typeof(opening_receipt_document) IS DISTINCT FROM 'object'
       OR jsonb_typeof(provisioning_receipt_document) IS DISTINCT FROM 'object'
       OR np.paper_canonical_json(intent_document) IS DISTINCT FROM requested_intent_payload
       OR np.paper_canonical_json(approval_document) IS DISTINCT FROM (
            requested_approval_payload
       )
       OR np.paper_canonical_json(trust_policy_document) IS DISTINCT FROM (
            requested_trust_policy_payload
       )
       OR np.paper_canonical_json(candidate_document) IS DISTINCT FROM (
            requested_candidate_payload
       )
       OR np.paper_canonical_json(opening_document) IS DISTINCT FROM (
            requested_opening_payload
       )
       OR np.paper_canonical_json(opening_receipt_document) IS DISTINCT FROM (
            requested_opening_receipt_payload
       )
       OR np.paper_canonical_json(provisioning_receipt_document) IS DISTINCT FROM (
            requested_provisioning_receipt_payload
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening documents are not canonical';
    END IF;

    IF np.paper_sha256_fresh_opening_intent(requested_intent_payload) IS DISTINCT FROM (
            requested_intent_payload_sha256
       )
       OR np.paper_sha256_text(requested_approval_payload) IS DISTINCT FROM (
            requested_approval_payload_sha256
       )
       OR np.paper_sha256_text(requested_trust_policy_payload) IS DISTINCT FROM (
            requested_trust_policy_payload_sha256
       )
       OR np.paper_sha256_text(requested_candidate_payload) IS DISTINCT FROM (
            requested_candidate_payload_sha256
       )
       OR np.paper_sha256_text(requested_opening_payload) IS DISTINCT FROM (
            requested_opening_payload_sha256
       )
       OR np.paper_sha256_text(requested_opening_receipt_payload) IS DISTINCT FROM (
            requested_opening_receipt_payload_sha256
       )
       OR np.paper_sha256_text(requested_provisioning_receipt_payload) IS DISTINCT FROM (
            requested_provisioning_receipt_payload_sha256
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening document digest mismatch';
    END IF;

    IF ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(intent_document) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY[
            'account_key',
            'approval_expires_at',
            'approval_id',
            'approval_issued_at',
            'approver_identity',
            'collateral_amount',
            'collateral_asset',
            'continuity',
            'execution_scope',
            'logical_target',
            'margin_quantum',
            'nonce',
            'opening_codec',
            'opening_policy',
            'opening_version',
            'operator_identity',
            'owner_generation',
            'purpose',
            'schema_version',
            'signer_key_id',
            'signer_public_key_sha256',
            'trajectory',
            'trust_domain',
            'trust_policy_sha256'
       ]::TEXT[]
       OR EXISTS (
            SELECT 1
            FROM jsonb_each(intent_document) intent_field(key, value)
            WHERE intent_field.key NOT IN (
                'opening_version', 'owner_generation', 'schema_version'
            )
              AND jsonb_typeof(intent_field.value) IS DISTINCT FROM 'string'
       )
       OR jsonb_typeof(intent_document->'schema_version') IS DISTINCT FROM 'number'
       OR jsonb_typeof(intent_document->'opening_version') IS DISTINCT FROM 'number'
       OR jsonb_typeof(intent_document->'owner_generation') IS DISTINCT FROM 'number'
       OR intent_document->'schema_version' IS DISTINCT FROM '1'::JSONB
       OR intent_document->'purpose' IS DISTINCT FROM (
            '"ELVIS_V2_FRESH_PAPER_OPENING"'::JSONB
       )
       OR intent_document->'trajectory' IS DISTINCT FROM '"B"'::JSONB
       OR intent_document->'continuity' IS DISTINCT FROM (
            '"NO_V1_CONTINUITY"'::JSONB
       )
       OR intent_document->'opening_codec' IS DISTINCT FROM (
            '"paper-account-opening"'::JSONB
       )
       OR intent_document->'opening_version' IS DISTINCT FROM '1'::JSONB
       OR intent_document->'opening_policy' IS DISTINCT FROM (
            '"EXPLICIT_FRESH_SINGLE_COLLATERAL"'::JSONB
       )
       OR intent_document->>'owner_generation' !~ '^[1-9][0-9]*$'
       OR (intent_document->>'owner_generation')::NUMERIC > 9223372036854775807
       OR intent_document->>'nonce' !~ '^[0-9a-f]{64}$'
       OR intent_document->>'nonce' = repeat('0', 64)
       OR intent_document->'trust_policy_sha256' IS DISTINCT FROM (
            to_jsonb(requested_trust_policy_payload_sha256)
       )
       OR intent_document->>'signer_public_key_sha256' !~ '^[0-9a-f]{64}$'
       OR intent_document->>'trust_domain' = ''
       OR intent_document->>'signer_key_id' = '' THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening intent is invalid';
    END IF;

    SELECT *
    INTO STRICT fence_row
    FROM np.acquire_paper_fresh_opening_fence(
        intent_document->>'trust_domain',
        intent_document->>'signer_key_id',
        intent_document->>'nonce',
        requested_candidate_payload_sha256
    );

    IF fence_row.resolution = 'EXACT_REPLAY' THEN
        IF fence_row.intent_payload = requested_intent_payload
           AND fence_row.intent_sha256 = requested_intent_payload_sha256
           AND fence_row.approval_payload = requested_approval_payload
           AND fence_row.approval_sha256 = requested_approval_payload_sha256
           AND fence_row.trust_policy_payload = requested_trust_policy_payload
           AND fence_row.trust_policy_sha256 = (
                requested_trust_policy_payload_sha256
           )
           AND fence_row.candidate_payload = requested_candidate_payload
           AND fence_row.candidate_sha256 = requested_candidate_payload_sha256
           AND fence_row.opening_payload = requested_opening_payload
           AND fence_row.opening_sha256 = requested_opening_payload_sha256
           AND fence_row.opening_receipt_payload = (
                requested_opening_receipt_payload
           )
           AND fence_row.opening_receipt_sha256 = (
                requested_opening_receipt_payload_sha256
           )
           AND fence_row.provisioning_receipt_payload = (
                requested_provisioning_receipt_payload
           )
           AND fence_row.provisioning_receipt_sha256 = (
                requested_provisioning_receipt_payload_sha256
           ) THEN
            RETURN QUERY SELECT
                'REPLAYED'::TEXT,
                fence_row.committed_at,
                fence_row.opening_receipt_payload,
                fence_row.opening_receipt_sha256,
                fence_row.provisioning_receipt_payload,
                fence_row.provisioning_receipt_sha256;
            RETURN;
        END IF;
        RAISE EXCEPTION USING
            ERRCODE = 'PT002',
            MESSAGE = 'paper fresh opening durable replay conflicts';
    ELSIF fence_row.resolution = 'NONCE_CONFLICT' THEN
        RAISE EXCEPTION USING
            ERRCODE = 'PT002',
            MESSAGE = 'paper fresh opening nonce conflicts';
    ELSIF fence_row.resolution = 'TARGET_CONFLICT' THEN
        RAISE EXCEPTION USING
            ERRCODE = 'PT003',
            MESSAGE = 'paper fresh opening target already has another opening';
    ELSIF fence_row.resolution = 'ADMISSION_CONFLICT' THEN
        RAISE EXCEPTION USING
            ERRCODE = 'PT003',
            MESSAGE = 'paper fresh opening candidate is not admitted';
    ELSIF fence_row.resolution <> 'ABSENT' THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening fence returned an unknown result';
    END IF;

    IF fence_row.v2_empty IS NOT TRUE THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper fresh opening target is not empty';
    END IF;

    IF ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(approval_document) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY['intent_sha256', 'schema_version', 'signature']::TEXT[]
       OR jsonb_typeof(approval_document->'schema_version') IS DISTINCT FROM 'number'
       OR jsonb_typeof(approval_document->'intent_sha256') IS DISTINCT FROM 'string'
       OR jsonb_typeof(approval_document->'signature') IS DISTINCT FROM 'string'
       OR approval_document->'schema_version' IS DISTINCT FROM '1'::JSONB
       OR approval_document->'intent_sha256' IS DISTINCT FROM (
            to_jsonb(requested_intent_payload_sha256)
       )
       OR approval_document->>'signature' !~ '^[0-9a-f]{128}$' THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening approval is invalid';
    END IF;

    IF ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(trust_policy_document) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY[
            'anchors',
            'max_approval_lifetime_seconds',
            'purpose',
            'schema_version',
            'trust_domain'
       ]::TEXT[]
       OR jsonb_typeof(trust_policy_document->'schema_version') IS DISTINCT FROM 'number'
       OR jsonb_typeof(trust_policy_document->'purpose') IS DISTINCT FROM 'string'
       OR jsonb_typeof(trust_policy_document->'trust_domain') IS DISTINCT FROM 'string'
       OR jsonb_typeof(
            trust_policy_document->'max_approval_lifetime_seconds'
       ) IS DISTINCT FROM 'number'
       OR trust_policy_document->'schema_version' IS DISTINCT FROM '1'::JSONB
       OR trust_policy_document->'purpose' IS DISTINCT FROM (
            '"ELVIS_V2_FRESH_PAPER_OPENING"'::JSONB
       )
       OR trust_policy_document->'trust_domain' IS DISTINCT FROM (
            intent_document->'trust_domain'
       )
       OR jsonb_typeof(trust_policy_document->'anchors') IS DISTINCT FROM 'array'
       OR trust_policy_document->>'max_approval_lifetime_seconds' !~ (
            '^[1-9][0-9]*$'
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening trust policy is invalid';
    END IF;

    SELECT anchor.value
    INTO STRICT matching_anchor
    FROM jsonb_array_elements(trust_policy_document->'anchors') anchor(value)
    WHERE anchor.value->'signer_key_id' = intent_document->'signer_key_id';

    IF ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(matching_anchor) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY[
            'approver_identity',
            'ed25519_public_key',
            'revoked',
            'signer_key_id'
       ]::TEXT[]
       OR jsonb_typeof(matching_anchor->'approver_identity') IS DISTINCT FROM 'string'
       OR jsonb_typeof(matching_anchor->'ed25519_public_key') IS DISTINCT FROM 'string'
       OR jsonb_typeof(matching_anchor->'signer_key_id') IS DISTINCT FROM 'string'
       OR matching_anchor->'approver_identity' IS DISTINCT FROM (
            intent_document->'approver_identity'
       )
       OR matching_anchor->'signer_key_id' IS DISTINCT FROM (
            intent_document->'signer_key_id'
       )
       OR matching_anchor->'revoked' IS DISTINCT FROM 'false'::JSONB
       OR matching_anchor->>'ed25519_public_key' !~ '^[0-9a-f]{64}$'
       OR encode(
            sha256(decode(matching_anchor->>'ed25519_public_key', 'hex')),
            'hex'
       ) IS DISTINCT FROM intent_document->>'signer_public_key_sha256' THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening trust anchor is invalid';
    END IF;

    BEGIN
        approval_issued_at := (
            intent_document->>'approval_issued_at'
        )::TIMESTAMPTZ;
        approval_expires_at := (
            intent_document->>'approval_expires_at'
        )::TIMESTAMPTZ;
        max_lifetime_seconds := (
            trust_policy_document->>'max_approval_lifetime_seconds'
        )::INTEGER;
    EXCEPTION
        WHEN datetime_field_overflow OR invalid_datetime_format
             OR numeric_value_out_of_range THEN
            RAISE EXCEPTION USING
                ERRCODE = '22023',
                MESSAGE = 'paper fresh opening approval window is invalid';
    END;

    IF intent_document->>'approval_issued_at' !~ (
            '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:'
            || '[0-9]{2}:[0-9]{2}\.[0-9]{6}\+00:00$'
       )
       OR intent_document->>'approval_expires_at' !~ (
            '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:'
            || '[0-9]{2}:[0-9]{2}\.[0-9]{6}\+00:00$'
       )
       OR approval_issued_at > fence_row.evaluated_at
       OR approval_expires_at <= fence_row.evaluated_at
       OR approval_expires_at <= approval_issued_at
       OR max_lifetime_seconds NOT BETWEEN 1 AND 2678400
       OR EXTRACT(EPOCH FROM approval_expires_at - approval_issued_at) > (
            max_lifetime_seconds
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = 'PT004',
            MESSAGE = 'paper fresh opening current approval is not valid';
    END IF;

    IF ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(candidate_document) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY[
            'approval_sha256',
            'intent_sha256',
            'opening_codec',
            'opening_payload_sha256',
            'opening_version',
            'schema_version',
            'trust_policy_sha256'
       ]::TEXT[]
       OR jsonb_typeof(candidate_document->'schema_version') IS DISTINCT FROM 'number'
       OR jsonb_typeof(candidate_document->'opening_version') IS DISTINCT FROM 'number'
       OR jsonb_typeof(candidate_document->'intent_sha256') IS DISTINCT FROM 'string'
       OR jsonb_typeof(candidate_document->'approval_sha256') IS DISTINCT FROM 'string'
       OR jsonb_typeof(candidate_document->'trust_policy_sha256') IS DISTINCT FROM 'string'
       OR jsonb_typeof(candidate_document->'opening_codec') IS DISTINCT FROM 'string'
       OR jsonb_typeof(candidate_document->'opening_payload_sha256') IS DISTINCT FROM 'string'
       OR candidate_document->'schema_version' IS DISTINCT FROM '1'::JSONB
       OR candidate_document->'intent_sha256' IS DISTINCT FROM (
            to_jsonb(requested_intent_payload_sha256)
       )
       OR candidate_document->'approval_sha256' IS DISTINCT FROM (
            to_jsonb(requested_approval_payload_sha256)
       )
       OR candidate_document->'trust_policy_sha256' IS DISTINCT FROM (
            to_jsonb(requested_trust_policy_payload_sha256)
       )
       OR candidate_document->'opening_codec' IS DISTINCT FROM (
            '"paper-account-opening"'::JSONB
       )
       OR candidate_document->'opening_version' IS DISTINCT FROM '1'::JSONB
       OR candidate_document->'opening_payload_sha256' IS DISTINCT FROM (
            to_jsonb(requested_opening_payload_sha256)
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening candidate is invalid';
    END IF;

    IF ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(opening_document) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY[
            'execution_scope',
            'opening_balances',
            'owner_generation',
            'policy'
       ]::TEXT[]
       OR jsonb_typeof(opening_document->'execution_scope') IS DISTINCT FROM 'string'
       OR jsonb_typeof(opening_document->'owner_generation') IS DISTINCT FROM 'number'
       OR jsonb_typeof(opening_document->'opening_balances') IS DISTINCT FROM 'array'
       OR jsonb_array_length(opening_document->'opening_balances') IS DISTINCT FROM 1
       OR jsonb_typeof(opening_document->'policy') IS DISTINCT FROM 'object' THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening payload is invalid';
    END IF;
    opening_balance := opening_document->'opening_balances'->0;
    opening_policy := opening_document->'policy';
    IF ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(opening_balance) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY['asset', 'available', 'reserved']::TEXT[]
       OR ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(opening_policy) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY['account_key', 'collateral_asset', 'margin_quantum']::TEXT[]
       OR EXISTS (
            SELECT 1
            FROM jsonb_each(opening_balance) opening_balance_field(key, value)
            WHERE jsonb_typeof(opening_balance_field.value) IS DISTINCT FROM 'string'
       )
       OR EXISTS (
            SELECT 1
            FROM jsonb_each(opening_policy) opening_policy_field(key, value)
            WHERE jsonb_typeof(opening_policy_field.value) IS DISTINCT FROM 'string'
       )
       OR opening_document->'execution_scope' IS DISTINCT FROM (
            intent_document->'execution_scope'
       )
       OR opening_document->'owner_generation' IS DISTINCT FROM (
            intent_document->'owner_generation'
       )
       OR opening_policy->'account_key' IS DISTINCT FROM intent_document->'account_key'
       OR opening_policy->'collateral_asset' IS DISTINCT FROM (
            intent_document->'collateral_asset'
       )
       OR opening_policy->'margin_quantum' IS DISTINCT FROM (
            intent_document->'margin_quantum'
       )
       OR opening_balance->'asset' IS DISTINCT FROM intent_document->'collateral_asset'
       OR opening_balance->'available' IS DISTINCT FROM (
            intent_document->'collateral_amount'
       )
       OR opening_balance->'reserved' IS DISTINCT FROM '"0"'::JSONB
       OR opening_balance->>'available' !~ (
            '^(0|[1-9][0-9]*)(\.[0-9]+)?$'
       )
       OR opening_policy->>'margin_quantum' !~ (
            '^(0|[1-9][0-9]*)(\.[0-9]+)?$'
       )
       OR LENGTH(SPLIT_PART(opening_balance->>'available', '.', 2)) > 128
       OR LENGTH(
            LTRIM(REPLACE(opening_balance->>'available', '.', ''), '0')
       ) > 128
       OR LENGTH(SPLIT_PART(opening_policy->>'margin_quantum', '.', 2)) > 128
       OR LENGTH(
            LTRIM(REPLACE(opening_policy->>'margin_quantum', '.', ''), '0')
       ) > 128 THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening balance is invalid';
    END IF;
    collateral_amount := (opening_balance->>'available')::NUMERIC;
    margin_quantum := (opening_policy->>'margin_quantum')::NUMERIC;
    IF collateral_amount <= 0
       OR margin_quantum <= 0
       OR MOD(collateral_amount, margin_quantum) <> 0 THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening amount is invalid';
    END IF;

    IF ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(opening_receipt_document) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY[
            'account_key',
            'collateral_asset',
            'execution_scope',
            'opening_payload_sha256',
            'opening_version',
            'owner_generation',
            'result',
            'schema_version'
       ]::TEXT[]
       OR opening_receipt_document->'schema_version' IS DISTINCT FROM '1'::JSONB
       OR opening_receipt_document->'result' IS DISTINCT FROM '"CREATED"'::JSONB
       OR opening_receipt_document->'execution_scope' IS DISTINCT FROM (
            intent_document->'execution_scope'
       )
       OR opening_receipt_document->'account_key' IS DISTINCT FROM (
            intent_document->'account_key'
       )
       OR opening_receipt_document->'owner_generation' IS DISTINCT FROM (
            intent_document->'owner_generation'
       )
       OR opening_receipt_document->'collateral_asset' IS DISTINCT FROM (
            intent_document->'collateral_asset'
       )
       OR opening_receipt_document->'opening_version' IS DISTINCT FROM (
            '1'::JSONB
       )
       OR opening_receipt_document->'opening_payload_sha256' IS DISTINCT FROM (
            to_jsonb(requested_opening_payload_sha256)
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening business receipt is invalid';
    END IF;

    IF ARRAY(
            SELECT object_key
            FROM jsonb_object_keys(provisioning_receipt_document) object_key
            ORDER BY object_key COLLATE pg_catalog."C"
       ) <> ARRAY[
            'approval_sha256',
            'authority_evaluated_at',
            'authority_transition_sequence',
            'candidate_sha256',
            'control_plane_role',
            'database_incarnation_id',
            'database_name',
            'deployment_incarnation_id',
            'intent_sha256',
            'migration_checksum',
            'migration_head',
            'migration_name',
            'opening_anchor_role',
            'opening_payload_sha256',
            'opening_receipt_sha256',
            'pin_authority_record_sha256',
            'runtime_activation_authorized',
            'runtime_generation',
            'runtime_mode',
            'schema_version',
            'stale_on_return',
            'system_identifier',
            'terminal_catalog_sha256',
            'trading_authorized',
            'trust_policy_sha256',
            'writer_fence'
       ]::TEXT[]
       OR provisioning_receipt_document->'schema_version' IS DISTINCT FROM (
            '1'::JSONB
       )
       OR provisioning_receipt_document->'intent_sha256' IS DISTINCT FROM (
            to_jsonb(requested_intent_payload_sha256)
       )
       OR provisioning_receipt_document->'approval_sha256' IS DISTINCT FROM (
            to_jsonb(requested_approval_payload_sha256)
       )
       OR provisioning_receipt_document->'trust_policy_sha256' IS DISTINCT FROM (
            to_jsonb(requested_trust_policy_payload_sha256)
       )
       OR provisioning_receipt_document->'candidate_sha256' IS DISTINCT FROM (
            to_jsonb(requested_candidate_payload_sha256)
       )
       OR provisioning_receipt_document->'opening_payload_sha256' IS DISTINCT FROM (
            to_jsonb(requested_opening_payload_sha256)
       )
       OR provisioning_receipt_document->'opening_receipt_sha256' IS DISTINCT FROM (
            to_jsonb(requested_opening_receipt_payload_sha256)
       )
       OR provisioning_receipt_document->'database_name' IS DISTINCT FROM (
            to_jsonb(fence_row.database_name)
       )
       OR provisioning_receipt_document->'system_identifier' IS DISTINCT FROM (
            to_jsonb(fence_row.system_identifier::TEXT)
       )
       OR provisioning_receipt_document->'control_plane_role' IS DISTINCT FROM (
            to_jsonb(fence_row.control_plane_role)
       )
       OR provisioning_receipt_document->'opening_anchor_role' IS DISTINCT FROM (
            to_jsonb(fence_row.opening_anchor_role)
       )
       OR provisioning_receipt_document->'authority_evaluated_at' IS DISTINCT FROM (
            to_jsonb(
                to_char(
                    fence_row.evaluated_at AT TIME ZONE 'UTC',
                    'YYYY-MM-DD"T"HH24:MI:SS.US'
                ) || '+00:00'
            )
       )
       OR provisioning_receipt_document->'migration_head' IS DISTINCT FROM (
            to_jsonb(fence_row.migration_version)
       )
       OR provisioning_receipt_document->'migration_name' IS DISTINCT FROM (
            to_jsonb(fence_row.migration_name)
       )
       OR provisioning_receipt_document->'migration_checksum' IS DISTINCT FROM (
            to_jsonb(fence_row.migration_checksum)
       )
       OR provisioning_receipt_document->'terminal_catalog_sha256' IS DISTINCT FROM (
            to_jsonb(fence_row.terminal_catalog_sha256)
       )
       OR provisioning_receipt_document->'pin_authority_record_sha256' IS DISTINCT FROM (
            to_jsonb(fence_row.pin_authority_record_sha256)
       )
       OR provisioning_receipt_document->'deployment_incarnation_id' IS DISTINCT FROM (
            to_jsonb(fence_row.deployment_incarnation_id)
       )
       OR provisioning_receipt_document->'runtime_mode' IS DISTINCT FROM (
            '"LEGACY"'::JSONB
       )
       OR provisioning_receipt_document->'runtime_generation' IS DISTINCT FROM (
            '0'::JSONB
       )
       OR provisioning_receipt_document->'authority_transition_sequence' IS DISTINCT FROM (
            '0'::JSONB
       )
       OR provisioning_receipt_document->'writer_fence' IS DISTINCT FROM (
            '0'::JSONB
       )
       OR provisioning_receipt_document->'runtime_activation_authorized' IS DISTINCT FROM (
            'false'::JSONB
       )
       OR provisioning_receipt_document->'trading_authorized' IS DISTINCT FROM (
            'false'::JSONB
       )
       OR provisioning_receipt_document->'stale_on_return' IS DISTINCT FROM (
            'true'::JSONB
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening provisioning receipt is invalid';
    END IF;

    derived_database_incarnation_id := (
        np.paper_fresh_opening_database_incarnation(
            fence_row.database_name,
            fence_row.system_identifier,
            fence_row.migration_version,
            fence_row.migration_name,
            fence_row.migration_checksum,
            fence_row.terminal_catalog_sha256,
            fence_row.control_plane_role,
            fence_row.opening_anchor_role,
            provisioning_receipt_document->>'deployment_incarnation_id'
        )
    );
    IF provisioning_receipt_document->'database_incarnation_id' IS DISTINCT FROM (
            to_jsonb(derived_database_incarnation_id)
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening database incarnation is invalid';
    END IF;

    commit_timestamp := clock_timestamp();
    IF approval_issued_at > commit_timestamp
       OR approval_expires_at <= commit_timestamp THEN
        RAISE EXCEPTION USING
            ERRCODE = 'PT004',
            MESSAGE = 'paper fresh opening current approval is not valid';
    END IF;

    INSERT INTO np.paper_fresh_opening_nonces (
        trust_domain,
        signer_key_id,
        nonce,
        candidate_payload_sha256,
        registered_at
    ) VALUES (
        intent_document->>'trust_domain',
        intent_document->>'signer_key_id',
        intent_document->>'nonce',
        requested_candidate_payload_sha256,
        commit_timestamp
    );

    INSERT INTO np.paper_account_streams (
        account_key,
        execution_scope,
        owner_generation,
        collateral_asset,
        account_version,
        account_state,
        opening_version,
        opening_payload,
        opening_payload_sha256,
        created_at,
        updated_at
    ) VALUES (
        intent_document->>'account_key',
        intent_document->>'execution_scope',
        (intent_document->>'owner_generation')::BIGINT,
        intent_document->>'collateral_asset',
        0,
        'ACTIVE',
        1,
        opening_document,
        requested_opening_payload_sha256,
        commit_timestamp,
        commit_timestamp
    );

    INSERT INTO np.paper_account_balances (
        account_key,
        asset,
        available_decimal,
        reserved_decimal,
        updated_at
    ) VALUES (
        intent_document->>'account_key',
        intent_document->>'collateral_asset',
        opening_balance->>'available',
        '0',
        commit_timestamp
    );

    INSERT INTO np.paper_fresh_opening_provisionings (
        control_key,
        trust_domain,
        signer_key_id,
        nonce,
        logical_target,
        execution_scope,
        account_key,
        owner_generation,
        collateral_asset,
        opening_version,
        intent_payload,
        intent_payload_sha256,
        approval_payload,
        approval_payload_sha256,
        trust_policy_payload,
        trust_policy_payload_sha256,
        candidate_payload,
        candidate_payload_sha256,
        opening_payload,
        opening_payload_sha256,
        opening_receipt_payload,
        opening_receipt_payload_sha256,
        provisioning_receipt_payload,
        provisioning_receipt_payload_sha256,
        database_name,
        system_identifier,
        control_plane_role,
        opening_anchor_role,
        migration_version,
        migration_name,
        migration_checksum,
        terminal_catalog_sha256,
        deployment_incarnation_id,
        database_incarnation_id,
        pin_authority_record_sha256,
        runtime_mode,
        runtime_generation,
        authority_transition_sequence,
        writer_fence,
        runtime_activation_authorized,
        trading_authorized,
        stale_on_return,
        authority_evaluated_at,
        committed_at
    ) VALUES (
        TRUE,
        intent_document->>'trust_domain',
        intent_document->>'signer_key_id',
        intent_document->>'nonce',
        intent_document->>'logical_target',
        intent_document->>'execution_scope',
        intent_document->>'account_key',
        (intent_document->>'owner_generation')::BIGINT,
        intent_document->>'collateral_asset',
        1,
        requested_intent_payload,
        requested_intent_payload_sha256,
        requested_approval_payload,
        requested_approval_payload_sha256,
        requested_trust_policy_payload,
        requested_trust_policy_payload_sha256,
        requested_candidate_payload,
        requested_candidate_payload_sha256,
        requested_opening_payload,
        requested_opening_payload_sha256,
        requested_opening_receipt_payload,
        requested_opening_receipt_payload_sha256,
        requested_provisioning_receipt_payload,
        requested_provisioning_receipt_payload_sha256,
        fence_row.database_name,
        fence_row.system_identifier,
        fence_row.control_plane_role,
        fence_row.opening_anchor_role,
        fence_row.migration_version,
        fence_row.migration_name,
        fence_row.migration_checksum,
        fence_row.terminal_catalog_sha256,
        provisioning_receipt_document->>'deployment_incarnation_id',
        derived_database_incarnation_id,
        provisioning_receipt_document->>'pin_authority_record_sha256',
        'LEGACY',
        0,
        0,
        0,
        FALSE,
        FALSE,
        TRUE,
        fence_row.evaluated_at,
        commit_timestamp
    );

    RETURN QUERY SELECT
        'CREATED'::TEXT,
        commit_timestamp,
        requested_opening_receipt_payload,
        requested_opening_receipt_payload_sha256,
        requested_provisioning_receipt_payload,
        requested_provisioning_receipt_payload_sha256;
EXCEPTION
    WHEN NO_DATA_FOUND OR TOO_MANY_ROWS THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'paper fresh opening document relationship is invalid';
END
$function$;

REVOKE ALL
ON FUNCTION np.commit_paper_fresh_opening(
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT,
    TEXT
)
FROM PUBLIC;

CREATE FUNCTION np.read_paper_fresh_opening(
    requested_trust_domain TEXT,
    requested_signer_key_id TEXT,
    requested_nonce TEXT
)
RETURNS TABLE(
    resolution TEXT,
    evaluated_at TIMESTAMPTZ,
    database_name TEXT,
    system_identifier NUMERIC,
    control_plane_role TEXT,
    opening_anchor_role TEXT,
    migration_version INTEGER,
    migration_name TEXT,
    migration_checksum TEXT,
    terminal_catalog_sha256 TEXT,
    pin_authority_record_sha256 TEXT,
    deployment_incarnation_id TEXT,
    database_incarnation_id TEXT,
    runtime_mode TEXT,
    runtime_generation BIGINT,
    authority_transition_sequence BIGINT,
    writer_fence BIGINT,
    v2_empty BOOLEAN,
    stored_authority_evaluated_at TIMESTAMPTZ,
    committed_at TIMESTAMPTZ,
    intent_payload TEXT,
    intent_sha256 TEXT,
    approval_payload TEXT,
    approval_sha256 TEXT,
    trust_policy_payload TEXT,
    trust_policy_sha256 TEXT,
    candidate_payload TEXT,
    candidate_sha256 TEXT,
    opening_payload TEXT,
    opening_sha256 TEXT,
    opening_receipt_payload TEXT,
    opening_receipt_sha256 TEXT,
    provisioning_receipt_payload TEXT,
    provisioning_receipt_sha256 TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    stored_candidate_sha256 TEXT;
BEGIN
    SELECT candidate_payload_sha256
    INTO stored_candidate_sha256
    FROM np.paper_fresh_opening_nonces
    WHERE trust_domain = requested_trust_domain
      AND signer_key_id = requested_signer_key_id
      AND nonce = requested_nonce;

    IF stored_candidate_sha256 IS NULL THEN
        stored_candidate_sha256 := repeat('0', 64);
    END IF;

    RETURN QUERY
    SELECT *
    FROM np.acquire_paper_fresh_opening_fence(
        requested_trust_domain,
        requested_signer_key_id,
        requested_nonce,
        stored_candidate_sha256
    );
END
$function$;

REVOKE ALL
ON FUNCTION np.read_paper_fresh_opening(TEXT, TEXT, TEXT)
FROM PUBLIC;
