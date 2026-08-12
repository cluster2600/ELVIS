CREATE UNIQUE INDEX orders_paper_account_batch_ref_uq
ON np.orders (
    position_key,
    client_order_id,
    execution_scope,
    instruction_payload_sha256
);

CREATE UNIQUE INDEX orders_paper_account_symbol_ref_uq
ON np.orders (
    position_key,
    client_order_id,
    symbol
);

CREATE UNIQUE INDEX order_events_paper_account_submission_ref_uq
ON np.order_events (
    position_key,
    position_version,
    client_order_id,
    event_id,
    event_type,
    occurred_at,
    event_payload_sha256
);

CREATE UNIQUE INDEX order_events_paper_account_fill_ref_uq
ON np.order_events (
    position_key,
    position_version,
    client_order_id,
    event_id,
    trade_id,
    event_type,
    event_payload_sha256
);

CREATE TABLE np.paper_account_streams (
    account_key VARCHAR(255) PRIMARY KEY,
    execution_scope VARCHAR(128) NOT NULL,
    owner_generation BIGINT NOT NULL,
    collateral_asset VARCHAR(64) NOT NULL,
    account_version BIGINT NOT NULL DEFAULT 0,
    account_state VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
    opening_version SMALLINT NOT NULL,
    opening_payload JSONB NOT NULL,
    opening_payload_sha256 CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT paper_account_streams_account_key_clean CHECK (
        account_key = BTRIM(account_key) AND account_key <> ''
    ),
    CONSTRAINT paper_account_streams_execution_scope_clean CHECK (
        execution_scope = BTRIM(execution_scope) AND execution_scope <> ''
    ),
    CONSTRAINT paper_account_streams_owner_generation_positive CHECK (
        owner_generation > 0
    ),
    CONSTRAINT paper_account_streams_collateral_asset_clean CHECK (
        collateral_asset = BTRIM(collateral_asset)
        AND collateral_asset <> ''
    ),
    CONSTRAINT paper_account_streams_version_non_negative CHECK (
        account_version >= 0
    ),
    CONSTRAINT paper_account_streams_state_known CHECK (
        account_state IN ('ACTIVE', 'INSOLVENT')
    ),
    CONSTRAINT paper_account_streams_opening_version_known CHECK (
        opening_version = 1
    ),
    CONSTRAINT paper_account_streams_opening_payload_object CHECK (
        jsonb_typeof(opening_payload) = 'object'
    ),
    CONSTRAINT paper_account_streams_opening_sha256_valid CHECK (
        opening_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_account_streams_opening_identity_uq UNIQUE (
        execution_scope,
        account_key,
        owner_generation
    ),
    CONSTRAINT paper_account_streams_opening_envelope_uq UNIQUE (
        execution_scope,
        account_key,
        owner_generation,
        opening_version,
        opening_payload_sha256
    ),
    CONSTRAINT paper_account_streams_scope_identity_uq UNIQUE (
        account_key,
        execution_scope
    ),
    CONSTRAINT paper_account_streams_collateral_identity_uq UNIQUE (
        account_key,
        collateral_asset
    )
);

CREATE TABLE np.paper_account_balances (
    account_key VARCHAR(255) NOT NULL,
    asset VARCHAR(64) NOT NULL,
    available_decimal TEXT NOT NULL,
    reserved_decimal TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT paper_account_balances_pk PRIMARY KEY (
        account_key,
        asset
    ),
    CONSTRAINT paper_account_balances_asset_clean CHECK (
        asset = BTRIM(asset) AND asset <> ''
    ),
    CONSTRAINT paper_account_balances_available_clean CHECK (
        available_decimal = BTRIM(available_decimal)
        AND available_decimal <> ''
    ),
    CONSTRAINT paper_account_balances_reserved_clean CHECK (
        reserved_decimal = BTRIM(reserved_decimal)
        AND reserved_decimal <> ''
    ),
    CONSTRAINT paper_account_balances_stream_fk FOREIGN KEY (
        account_key
    ) REFERENCES np.paper_account_streams (
        account_key
    ) ON DELETE RESTRICT
);

CREATE TABLE np.paper_margin_reservations (
    account_key VARCHAR(255) NOT NULL,
    execution_scope VARCHAR(128) NOT NULL,
    position_key VARCHAR(255) NOT NULL,
    amount_decimal TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT paper_margin_reservations_pk PRIMARY KEY (
        account_key,
        position_key
    ),
    CONSTRAINT paper_margin_reservations_execution_scope_clean CHECK (
        execution_scope = BTRIM(execution_scope) AND execution_scope <> ''
    ),
    CONSTRAINT paper_margin_reservations_position_key_clean CHECK (
        position_key = BTRIM(position_key) AND position_key <> ''
    ),
    CONSTRAINT paper_margin_reservations_amount_clean CHECK (
        amount_decimal = BTRIM(amount_decimal)
        AND amount_decimal <> ''
    ),
    CONSTRAINT paper_margin_reservations_account_scope_fk FOREIGN KEY (
        account_key,
        execution_scope
    ) REFERENCES np.paper_account_streams (
        account_key,
        execution_scope
    ) ON DELETE RESTRICT,
    CONSTRAINT paper_margin_reservations_position_scope_fk FOREIGN KEY (
        position_key,
        execution_scope
    ) REFERENCES np.position_streams (
        position_key,
        execution_scope
    ) ON DELETE RESTRICT
);

CREATE TABLE np.paper_account_batch_manifests (
    account_key VARCHAR(255) NOT NULL,
    client_order_id VARCHAR(255) NOT NULL,
    execution_scope VARCHAR(128) NOT NULL,
    owner_generation BIGINT NOT NULL,
    opening_version SMALLINT NOT NULL,
    opening_payload_sha256 CHAR(64) NOT NULL,
    position_key VARCHAR(255) NOT NULL,
    instruction_payload_sha256 CHAR(64) NOT NULL,
    submission_event_id VARCHAR(255) NOT NULL,
    submission_event_type VARCHAR(32) NOT NULL,
    submission_position_version BIGINT NOT NULL,
    submission_observed_at TIMESTAMPTZ NOT NULL,
    submission_event_payload_sha256 CHAR(64) NOT NULL,
    first_account_version BIGINT NOT NULL,
    last_account_version BIGINT NOT NULL,
    last_position_version BIGINT NOT NULL,
    fill_count BIGINT NOT NULL,
    batch_version SMALLINT NOT NULL,
    batch_payload JSONB NOT NULL,
    batch_payload_sha256 CHAR(64) NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT paper_account_batch_manifests_pk PRIMARY KEY (
        account_key,
        client_order_id
    ),
    CONSTRAINT paper_account_batch_manifests_order_owner_uq UNIQUE (
        client_order_id
    ),
    CONSTRAINT paper_account_batch_manifests_membership_uq UNIQUE (
        account_key,
        client_order_id,
        first_account_version,
        submission_position_version,
        fill_count
    ),
    CONSTRAINT paper_account_batch_manifests_client_order_id_clean CHECK (
        client_order_id = BTRIM(client_order_id) AND client_order_id <> ''
    ),
    CONSTRAINT paper_account_batch_manifests_execution_scope_clean CHECK (
        execution_scope = BTRIM(execution_scope) AND execution_scope <> ''
    ),
    CONSTRAINT paper_account_batch_manifests_owner_generation_positive CHECK (
        owner_generation > 0
    ),
    CONSTRAINT paper_account_batch_manifests_opening_sha256_valid CHECK (
        opening_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_account_batch_manifests_opening_version_known CHECK (
        opening_version = 1
    ),
    CONSTRAINT paper_account_batch_manifests_position_key_clean CHECK (
        position_key = BTRIM(position_key) AND position_key <> ''
    ),
    CONSTRAINT paper_account_batch_manifests_instruction_sha256_valid CHECK (
        instruction_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_account_batch_manifests_submission_event_id_clean CHECK (
        submission_event_id = BTRIM(submission_event_id)
        AND submission_event_id <> ''
    ),
    CONSTRAINT paper_account_batch_manifests_submission_event_type_known CHECK (
        submission_event_type = 'SUBMISSION_ACKNOWLEDGED'
    ),
    CONSTRAINT paper_account_batch_manifests_submission_version_positive CHECK (
        submission_position_version > 0
    ),
    CONSTRAINT paper_account_batch_manifests_submission_sha256_valid CHECK (
        submission_event_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_account_batch_manifests_first_account_version_positive CHECK (
        first_account_version > 0
    ),
    CONSTRAINT paper_account_batch_manifests_last_account_version_positive CHECK (
        last_account_version > 0
    ),
    CONSTRAINT paper_account_batch_manifests_last_position_version_positive CHECK (
        last_position_version > 0
    ),
    CONSTRAINT paper_account_batch_manifests_fill_count_positive CHECK (
        fill_count > 0
    ),
    CONSTRAINT paper_account_batch_manifests_position_range_exact CHECK (
        last_position_version::NUMERIC
        - submission_position_version::NUMERIC = fill_count::NUMERIC
    ),
    CONSTRAINT paper_account_batch_manifests_account_range_exact CHECK (
        last_account_version::NUMERIC
        - first_account_version::NUMERIC + 1 = fill_count::NUMERIC
    ),
    CONSTRAINT paper_account_batch_manifests_version_known CHECK (
        batch_version = 1
    ),
    CONSTRAINT paper_account_batch_manifests_payload_object CHECK (
        jsonb_typeof(batch_payload) = 'object'
    ),
    CONSTRAINT paper_account_batch_manifests_payload_sha256_valid CHECK (
        batch_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_account_batch_manifests_opening_fk FOREIGN KEY (
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
    CONSTRAINT paper_account_batch_manifests_order_fk FOREIGN KEY (
        position_key,
        client_order_id,
        execution_scope,
        instruction_payload_sha256
    ) REFERENCES np.orders (
        position_key,
        client_order_id,
        execution_scope,
        instruction_payload_sha256
    ) ON DELETE RESTRICT,
    CONSTRAINT paper_account_batch_manifests_submission_fk FOREIGN KEY (
        position_key,
        submission_position_version,
        client_order_id,
        submission_event_id,
        submission_event_type,
        submission_observed_at,
        submission_event_payload_sha256
    ) REFERENCES np.order_events (
        position_key,
        position_version,
        client_order_id,
        event_id,
        event_type,
        occurred_at,
        event_payload_sha256
    ) ON DELETE RESTRICT
);

CREATE TABLE np.paper_account_settlements (
    account_key VARCHAR(255) NOT NULL,
    account_version BIGINT NOT NULL,
    client_order_id VARCHAR(255) NOT NULL,
    fill_ordinal BIGINT NOT NULL,
    batch_first_account_version BIGINT NOT NULL,
    batch_submission_position_version BIGINT NOT NULL,
    batch_fill_count BIGINT NOT NULL,
    collateral_asset VARCHAR(64) NOT NULL,
    position_key VARCHAR(255) NOT NULL,
    position_version BIGINT NOT NULL,
    event_id VARCHAR(255) NOT NULL,
    trade_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    event_payload_sha256 CHAR(64) NOT NULL,
    symbol VARCHAR(64) NOT NULL,
    base_asset VARCHAR(64) NOT NULL,
    quote_asset VARCHAR(64) NOT NULL,
    instrument_version SMALLINT NOT NULL,
    settlement_version SMALLINT NOT NULL,
    settlement_payload JSONB NOT NULL,
    settlement_payload_sha256 CHAR(64) NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT paper_account_settlements_pk PRIMARY KEY (
        account_key,
        account_version
    ),
    CONSTRAINT paper_account_settlements_batch_ordinal_uq UNIQUE (
        account_key,
        client_order_id,
        fill_ordinal
    ),
    CONSTRAINT paper_account_settlements_position_version_uq UNIQUE (
        position_key,
        position_version
    ),
    CONSTRAINT paper_account_settlements_event_identity_uq UNIQUE (
        client_order_id,
        event_id
    ),
    CONSTRAINT paper_account_settlements_fill_identity_uq UNIQUE (
        client_order_id,
        trade_id
    ),
    CONSTRAINT paper_account_settlements_account_version_positive CHECK (
        account_version > 0
    ),
    CONSTRAINT paper_account_settlements_client_order_id_clean CHECK (
        client_order_id = BTRIM(client_order_id) AND client_order_id <> ''
    ),
    CONSTRAINT paper_account_settlements_fill_ordinal_positive CHECK (
        fill_ordinal > 0
    ),
    CONSTRAINT paper_account_settlements_batch_fill_count_positive CHECK (
        batch_fill_count > 0
    ),
    CONSTRAINT paper_account_settlements_ordinal_in_batch CHECK (
        fill_ordinal <= batch_fill_count
    ),
    CONSTRAINT paper_account_settlements_account_ordinal_exact CHECK (
        account_version::NUMERIC
        - batch_first_account_version::NUMERIC + 1 = fill_ordinal::NUMERIC
    ),
    CONSTRAINT paper_account_settlements_position_ordinal_exact CHECK (
        position_version::NUMERIC
        - batch_submission_position_version::NUMERIC = fill_ordinal::NUMERIC
    ),
    CONSTRAINT paper_account_settlements_collateral_asset_clean CHECK (
        collateral_asset = BTRIM(collateral_asset)
        AND collateral_asset <> ''
    ),
    CONSTRAINT paper_account_settlements_position_key_clean CHECK (
        position_key = BTRIM(position_key) AND position_key <> ''
    ),
    CONSTRAINT paper_account_settlements_position_version_positive CHECK (
        position_version > 0
    ),
    CONSTRAINT paper_account_settlements_event_id_clean CHECK (
        event_id = BTRIM(event_id) AND event_id <> ''
    ),
    CONSTRAINT paper_account_settlements_trade_id_clean CHECK (
        trade_id = BTRIM(trade_id) AND trade_id <> ''
    ),
    CONSTRAINT paper_account_settlements_event_type_known CHECK (
        event_type = 'CONFIRMED_FILL'
    ),
    CONSTRAINT paper_account_settlements_event_sha256_valid CHECK (
        event_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_account_settlements_symbol_clean CHECK (
        symbol = BTRIM(symbol) AND symbol <> ''
    ),
    CONSTRAINT paper_account_settlements_base_asset_clean CHECK (
        base_asset = BTRIM(base_asset) AND base_asset <> ''
    ),
    CONSTRAINT paper_account_settlements_quote_asset_clean CHECK (
        quote_asset = BTRIM(quote_asset) AND quote_asset <> ''
    ),
    CONSTRAINT paper_account_settlements_instrument_assets_distinct CHECK (
        base_asset <> quote_asset
    ),
    CONSTRAINT paper_account_settlements_quote_is_collateral CHECK (
        quote_asset = collateral_asset
    ),
    CONSTRAINT paper_account_settlements_instrument_version_known CHECK (
        instrument_version = 1
    ),
    CONSTRAINT paper_account_settlements_version_known CHECK (
        settlement_version = 1
    ),
    CONSTRAINT paper_account_settlements_payload_object CHECK (
        jsonb_typeof(settlement_payload) = 'object'
    ),
    CONSTRAINT paper_account_settlements_payload_sha256_valid CHECK (
        settlement_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT paper_account_settlements_account_collateral_fk FOREIGN KEY (
        account_key,
        collateral_asset
    ) REFERENCES np.paper_account_streams (
        account_key,
        collateral_asset
    ) ON DELETE RESTRICT,
    CONSTRAINT paper_account_settlements_batch_fk FOREIGN KEY (
        account_key,
        client_order_id,
        batch_first_account_version,
        batch_submission_position_version,
        batch_fill_count
    ) REFERENCES np.paper_account_batch_manifests (
        account_key,
        client_order_id,
        first_account_version,
        submission_position_version,
        fill_count
    ) ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT paper_account_settlements_fill_fk FOREIGN KEY (
        position_key,
        position_version,
        client_order_id,
        event_id,
        trade_id,
        event_type,
        event_payload_sha256
    ) REFERENCES np.order_events (
        position_key,
        position_version,
        client_order_id,
        event_id,
        trade_id,
        event_type,
        event_payload_sha256
    ) ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT paper_account_settlements_order_symbol_fk FOREIGN KEY (
        position_key,
        client_order_id,
        symbol
    ) REFERENCES np.orders (
        position_key,
        client_order_id,
        symbol
    ) ON DELETE RESTRICT
);

CREATE TABLE np.paper_account_postings (
    account_key VARCHAR(255) NOT NULL,
    account_version BIGINT NOT NULL,
    posting_ordinal BIGINT NOT NULL,
    asset VARCHAR(64) NOT NULL,
    bucket VARCHAR(32) NOT NULL,
    amount_decimal TEXT NOT NULL,
    CONSTRAINT paper_account_postings_pk PRIMARY KEY (
        account_key,
        account_version,
        posting_ordinal
    ),
    CONSTRAINT paper_account_postings_bucket_identity_uq UNIQUE (
        account_key,
        account_version,
        asset,
        bucket
    ),
    CONSTRAINT paper_account_postings_ordinal_positive CHECK (
        posting_ordinal > 0
    ),
    CONSTRAINT paper_account_postings_asset_clean CHECK (
        asset = BTRIM(asset) AND asset <> ''
    ),
    CONSTRAINT paper_account_postings_bucket_known CHECK (
        bucket IN ('AVAILABLE', 'RESERVED_MARGIN')
    ),
    CONSTRAINT paper_account_postings_amount_clean CHECK (
        amount_decimal = BTRIM(amount_decimal) AND amount_decimal <> ''
    ),
    CONSTRAINT paper_account_postings_settlement_fk FOREIGN KEY (
        account_key,
        account_version
    ) REFERENCES np.paper_account_settlements (
        account_key,
        account_version
    ) ON DELETE RESTRICT
);
