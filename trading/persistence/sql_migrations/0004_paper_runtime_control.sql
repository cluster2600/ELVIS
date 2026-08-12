CREATE TABLE np.paper_runtime_control (
    control_key BOOLEAN PRIMARY KEY DEFAULT TRUE,
    mode TEXT NOT NULL,
    runtime_generation BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT paper_runtime_control_singleton CHECK (control_key),
    CONSTRAINT paper_runtime_control_mode CHECK (
        mode IN ('LEGACY', 'SHADOW', 'PAUSED', 'ACTIVE')
    ),
    CONSTRAINT paper_runtime_control_generation_nonnegative CHECK (
        runtime_generation >= 0
    )
);

INSERT INTO np.paper_runtime_control (
    control_key,
    mode,
    runtime_generation
) VALUES (
    TRUE,
    'LEGACY',
    0
);

CREATE FUNCTION np.enforce_legacy_paper_runtime_fence()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
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
END
$function$;

CREATE TRIGGER legacy_paper_runtime_fence_account_balances
BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE
ON np.account_balances
FOR EACH STATEMENT
EXECUTE FUNCTION np.enforce_legacy_paper_runtime_fence();

ALTER TABLE np.account_balances
ENABLE ALWAYS TRIGGER legacy_paper_runtime_fence_account_balances;

CREATE TRIGGER legacy_paper_runtime_fence_liquidations
BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE
ON np.liquidations
FOR EACH STATEMENT
EXECUTE FUNCTION np.enforce_legacy_paper_runtime_fence();

ALTER TABLE np.liquidations
ENABLE ALWAYS TRIGGER legacy_paper_runtime_fence_liquidations;

CREATE TRIGGER legacy_paper_runtime_fence_margin_history
BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE
ON np.margin_history
FOR EACH STATEMENT
EXECUTE FUNCTION np.enforce_legacy_paper_runtime_fence();

ALTER TABLE np.margin_history
ENABLE ALWAYS TRIGGER legacy_paper_runtime_fence_margin_history;

CREATE TRIGGER legacy_paper_runtime_fence_model_predictions
BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE
ON np.model_predictions
FOR EACH STATEMENT
EXECUTE FUNCTION np.enforce_legacy_paper_runtime_fence();

ALTER TABLE np.model_predictions
ENABLE ALWAYS TRIGGER legacy_paper_runtime_fence_model_predictions;

CREATE TRIGGER legacy_paper_runtime_fence_open_positions
BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE
ON np.open_positions
FOR EACH STATEMENT
EXECUTE FUNCTION np.enforce_legacy_paper_runtime_fence();

ALTER TABLE np.open_positions
ENABLE ALWAYS TRIGGER legacy_paper_runtime_fence_open_positions;

CREATE TRIGGER legacy_paper_runtime_fence_trades
BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE
ON np.trades
FOR EACH STATEMENT
EXECUTE FUNCTION np.enforce_legacy_paper_runtime_fence();

ALTER TABLE np.trades
ENABLE ALWAYS TRIGGER legacy_paper_runtime_fence_trades;

CREATE TRIGGER legacy_paper_runtime_fence_trading_session_resets
BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE
ON np.trading_session_resets
FOR EACH STATEMENT
EXECUTE FUNCTION np.enforce_legacy_paper_runtime_fence();

ALTER TABLE np.trading_session_resets
ENABLE ALWAYS TRIGGER legacy_paper_runtime_fence_trading_session_resets;
