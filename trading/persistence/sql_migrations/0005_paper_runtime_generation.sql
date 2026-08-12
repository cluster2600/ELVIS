CREATE TABLE np.paper_runtime_generations (
    runtime_generation BIGINT PRIMARY KEY,
    activation_id VARCHAR(255) NOT NULL,
    execution_scope VARCHAR(128) NOT NULL,
    account_key VARCHAR(255) NOT NULL,
    owner_generation BIGINT NOT NULL,
    opening_version SMALLINT NOT NULL,
    opening_payload_sha256 CHAR(64) NOT NULL,
    activated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT paper_runtime_generations_generation_positive CHECK (
        runtime_generation > 0
    ),
    CONSTRAINT paper_runtime_generations_activation_id_clean CHECK (
        activation_id = BTRIM(activation_id) AND activation_id <> ''
    ),
    CONSTRAINT paper_runtime_generations_activation_id_uq UNIQUE (
        activation_id
    ),
    CONSTRAINT paper_runtime_generations_execution_scope_clean CHECK (
        execution_scope = BTRIM(execution_scope) AND execution_scope <> ''
    ),
    CONSTRAINT paper_runtime_generations_account_key_clean CHECK (
        account_key = BTRIM(account_key) AND account_key <> ''
    ),
    CONSTRAINT paper_runtime_generations_owner_generation_positive CHECK (
        owner_generation > 0
    ),
    CONSTRAINT paper_runtime_generations_opening_version_known CHECK (
        opening_version = 1
    ),
    CONSTRAINT paper_runtime_generations_opening_sha256_valid CHECK (
        opening_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_runtime_generations_activated_at_finite CHECK (
        isfinite(activated_at)
    ),
    CONSTRAINT paper_runtime_generations_manifest_ref_uq UNIQUE (
        runtime_generation,
        execution_scope,
        account_key,
        owner_generation,
        opening_version,
        opening_payload_sha256
    ),
    CONSTRAINT paper_runtime_generations_opening_fk FOREIGN KEY (
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
    ) ON DELETE RESTRICT
);

CREATE FUNCTION np.reject_paper_runtime_generation_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
BEGIN
    RAISE EXCEPTION USING
        ERRCODE = '55000',
        MESSAGE = 'paper runtime generations are append-only';
END
$function$;

CREATE TRIGGER paper_runtime_generations_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE
ON np.paper_runtime_generations
FOR EACH STATEMENT
EXECUTE FUNCTION np.reject_paper_runtime_generation_mutation();

ALTER TABLE np.paper_runtime_generations
ENABLE ALWAYS TRIGGER paper_runtime_generations_append_only;

ALTER TABLE np.paper_account_batch_manifests
ADD COLUMN runtime_generation BIGINT;

ALTER TABLE np.paper_account_batch_manifests
DROP CONSTRAINT paper_account_batch_manifests_version_known;

ALTER TABLE np.paper_account_batch_manifests
ADD CONSTRAINT paper_account_batch_manifests_version_known CHECK (
    (batch_version = 1 AND runtime_generation IS NULL)
    OR
    (
        batch_version = 2
        AND runtime_generation IS NOT NULL
        AND runtime_generation > 0
    )
);

ALTER TABLE np.paper_account_batch_manifests
ADD CONSTRAINT paper_account_batch_manifests_runtime_generation_fk FOREIGN KEY (
    runtime_generation,
    execution_scope,
    account_key,
    owner_generation,
    opening_version,
    opening_payload_sha256
) REFERENCES np.paper_runtime_generations (
    runtime_generation,
    execution_scope,
    account_key,
    owner_generation,
    opening_version,
    opening_payload_sha256
) ON DELETE RESTRICT;
