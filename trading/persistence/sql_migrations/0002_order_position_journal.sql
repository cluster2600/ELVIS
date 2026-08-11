CREATE TABLE np.position_streams (
    position_key VARCHAR(255) PRIMARY KEY,
    execution_scope VARCHAR(128) NOT NULL,
    stream_version BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT position_streams_position_key_clean CHECK (
        position_key = BTRIM(position_key) AND position_key <> ''
    ),
    CONSTRAINT position_streams_execution_scope_clean CHECK (
        execution_scope = BTRIM(execution_scope) AND execution_scope <> ''
    ),
    CONSTRAINT position_streams_version_non_negative CHECK (stream_version >= 0),
    CONSTRAINT position_streams_scope_identity_uq UNIQUE (
        position_key,
        execution_scope
    )
);

CREATE TABLE np.orders (
    client_order_id VARCHAR(255) PRIMARY KEY,
    decision_id VARCHAR(255) NOT NULL,
    position_key VARCHAR(255) NOT NULL,
    execution_scope VARCHAR(128) NOT NULL,
    symbol VARCHAR(64) NOT NULL,
    position_effect VARCHAR(16) NOT NULL,
    instruction_version SMALLINT NOT NULL,
    instruction_payload JSONB NOT NULL,
    instruction_payload_sha256 CHAR(64) NOT NULL,
    venue_order_id VARCHAR(255),
    registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT orders_client_order_id_clean CHECK (
        client_order_id = BTRIM(client_order_id) AND client_order_id <> ''
    ),
    CONSTRAINT orders_decision_id_clean CHECK (
        decision_id = BTRIM(decision_id) AND decision_id <> ''
    ),
    CONSTRAINT orders_position_key_clean CHECK (
        position_key = BTRIM(position_key) AND position_key <> ''
    ),
    CONSTRAINT orders_execution_scope_clean CHECK (
        execution_scope = BTRIM(execution_scope) AND execution_scope <> ''
    ),
    CONSTRAINT orders_symbol_clean CHECK (
        symbol = BTRIM(symbol) AND symbol <> ''
    ),
    CONSTRAINT orders_position_effect_known CHECK (
        position_effect IN ('OPEN', 'REDUCE_ONLY')
    ),
    CONSTRAINT orders_instruction_version_known CHECK (instruction_version = 1),
    CONSTRAINT orders_instruction_payload_object CHECK (
        jsonb_typeof(instruction_payload) = 'object'
    ),
    CONSTRAINT orders_instruction_payload_sha256_valid CHECK (
        instruction_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT orders_venue_order_id_clean CHECK (
        venue_order_id IS NULL
        OR (
            venue_order_id = BTRIM(venue_order_id)
            AND venue_order_id <> ''
        )
    ),
    CONSTRAINT orders_position_client_uq UNIQUE (
        position_key,
        client_order_id
    ),
    CONSTRAINT orders_scope_decision_uq UNIQUE (
        execution_scope,
        decision_id
    ),
    CONSTRAINT orders_position_stream_fk FOREIGN KEY (
        position_key,
        execution_scope
    ) REFERENCES np.position_streams (
        position_key,
        execution_scope
    ) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX orders_venue_identity_uq
ON np.orders (execution_scope, symbol, venue_order_id)
WHERE venue_order_id IS NOT NULL;

CREATE TABLE np.order_events (
    position_key VARCHAR(255) NOT NULL,
    position_version BIGINT NOT NULL,
    client_order_id VARCHAR(255) NOT NULL,
    event_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    event_version SMALLINT NOT NULL,
    event_payload JSONB NOT NULL,
    event_payload_sha256 CHAR(64) NOT NULL,
    trade_id VARCHAR(255),
    occurred_at TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT order_events_position_version_positive CHECK (
        position_version > 0
    ),
    CONSTRAINT order_events_client_order_id_clean CHECK (
        client_order_id = BTRIM(client_order_id) AND client_order_id <> ''
    ),
    CONSTRAINT order_events_event_id_clean CHECK (
        event_id = BTRIM(event_id) AND event_id <> ''
    ),
    CONSTRAINT order_events_type_known CHECK (event_type IN (
        'SUBMISSION_ACKNOWLEDGED',
        'SUBMISSION_AMBIGUOUS',
        'SUBMISSION_FAILED',
        'CONFIRMED_FILL',
        'CANCELLATION_REQUESTED',
        'CANCELLATION_CONFIRMED',
        'CANCELLATION_REJECTED'
    )),
    CONSTRAINT order_events_version_known CHECK (event_version = 1),
    CONSTRAINT order_events_payload_object CHECK (
        jsonb_typeof(event_payload) = 'object'
    ),
    CONSTRAINT order_events_payload_sha256_valid CHECK (
        event_payload_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT order_events_trade_id_clean CHECK (
        trade_id IS NULL
        OR (trade_id = BTRIM(trade_id) AND trade_id <> '')
    ),
    CONSTRAINT order_events_fill_identity_present CHECK (
        (event_type = 'CONFIRMED_FILL' AND trade_id IS NOT NULL)
        OR (event_type <> 'CONFIRMED_FILL' AND trade_id IS NULL)
    ),
    CONSTRAINT order_events_position_version_pk PRIMARY KEY (
        position_key,
        position_version
    ),
    CONSTRAINT order_events_event_identity_uq UNIQUE (
        client_order_id,
        event_id
    ),
    CONSTRAINT order_events_order_fk FOREIGN KEY (
        position_key,
        client_order_id
    ) REFERENCES np.orders (
        position_key,
        client_order_id
    ) ON DELETE RESTRICT
);

CREATE INDEX order_events_order_replay_idx
ON np.order_events (client_order_id, position_version);

CREATE UNIQUE INDEX order_events_fill_identity_uq
ON np.order_events (client_order_id, trade_id)
WHERE event_type = 'CONFIRMED_FILL';
