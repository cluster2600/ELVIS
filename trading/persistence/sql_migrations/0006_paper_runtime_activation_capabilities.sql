CREATE FUNCTION np.acquire_paper_runtime_activation_fence()
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
BEGIN
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
END
$function$;

REVOKE ALL
ON FUNCTION np.acquire_paper_runtime_activation_fence()
FROM PUBLIC;

CREATE FUNCTION np.activate_paper_runtime_generation(
    expected_mode TEXT,
    expected_generation BIGINT,
    target_generation BIGINT,
    requested_activation_id TEXT,
    requested_execution_scope TEXT,
    requested_account_key TEXT,
    requested_owner_generation BIGINT,
    requested_opening_payload_sha256 TEXT
)
RETURNS TABLE(mode TEXT, runtime_generation BIGINT)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
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
END
$function$;

REVOKE ALL
ON FUNCTION np.activate_paper_runtime_generation(
    TEXT,
    BIGINT,
    BIGINT,
    TEXT,
    TEXT,
    TEXT,
    BIGINT,
    TEXT
)
FROM PUBLIC;
