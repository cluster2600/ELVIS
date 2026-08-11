CREATE SCHEMA IF NOT EXISTS np;

CREATE TABLE IF NOT EXISTS np.trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    symbol TEXT,
    side TEXT,
    price REAL,
    quantity REAL,
    pnl REAL,
    fee REAL
);

CREATE TABLE IF NOT EXISTS np.open_positions (
    id SERIAL PRIMARY KEY,
    symbol TEXT,
    side TEXT,
    entry_price REAL,
    quantity REAL,
    leverage REAL,
    entry_time TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS np.liquidations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    symbol TEXT,
    entry_price REAL,
    liquidation_price REAL,
    quantity REAL,
    leverage REAL,
    liquidation_fee REAL
);

CREATE TABLE IF NOT EXISTS np.margin_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    balance REAL,
    used_margin REAL,
    open_positions INTEGER
);

CREATE TABLE IF NOT EXISTS np.trading_session_resets (
    id SERIAL PRIMARY KEY,
    reset_timestamp TIMESTAMP DEFAULT NOW(),
    reason TEXT
);

CREATE TABLE IF NOT EXISTS np.model_predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    symbol TEXT,
    side TEXT,
    model TEXT,
    vote TEXT,
    scored BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS np.account_balances (
    id SERIAL PRIMARY KEY,
    asset TEXT UNIQUE NOT NULL,
    balance REAL NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_scored
ON np.model_predictions (scored, created_at);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts
ON np.trades (symbol, timestamp);

DO $migration$
DECLARE
    layout_mismatch BOOLEAN;
    constraint_mismatch BOOLEAN;
    relation_mismatch BOOLEAN;
    sequence_mismatch BOOLEAN;
    behavior_mismatch BOOLEAN;
    index_mismatch BOOLEAN;
    unexpected_unique_index BOOLEAN;
BEGIN
    WITH expected (
        table_name,
        ordinal_position,
        column_name,
        udt_name,
        is_nullable,
        default_kind
    ) AS (
        VALUES
            ('trades', 1, 'id', 'int4', 'NO', 'serial'),
            ('trades', 2, 'timestamp', 'timestamp', 'YES', 'now'),
            ('trades', 3, 'symbol', 'text', 'YES', 'none'),
            ('trades', 4, 'side', 'text', 'YES', 'none'),
            ('trades', 5, 'price', 'float4', 'YES', 'none'),
            ('trades', 6, 'quantity', 'float4', 'YES', 'none'),
            ('trades', 7, 'pnl', 'float4', 'YES', 'none'),
            ('trades', 8, 'fee', 'float4', 'YES', 'none'),
            ('open_positions', 1, 'id', 'int4', 'NO', 'serial'),
            ('open_positions', 2, 'symbol', 'text', 'YES', 'none'),
            ('open_positions', 3, 'side', 'text', 'YES', 'none'),
            ('open_positions', 4, 'entry_price', 'float4', 'YES', 'none'),
            ('open_positions', 5, 'quantity', 'float4', 'YES', 'none'),
            ('open_positions', 6, 'leverage', 'float4', 'YES', 'none'),
            ('open_positions', 7, 'entry_time', 'timestamp', 'YES', 'now'),
            ('liquidations', 1, 'id', 'int4', 'NO', 'serial'),
            ('liquidations', 2, 'timestamp', 'timestamp', 'YES', 'now'),
            ('liquidations', 3, 'symbol', 'text', 'YES', 'none'),
            ('liquidations', 4, 'entry_price', 'float4', 'YES', 'none'),
            ('liquidations', 5, 'liquidation_price', 'float4', 'YES', 'none'),
            ('liquidations', 6, 'quantity', 'float4', 'YES', 'none'),
            ('liquidations', 7, 'leverage', 'float4', 'YES', 'none'),
            ('liquidations', 8, 'liquidation_fee', 'float4', 'YES', 'none'),
            ('margin_history', 1, 'id', 'int4', 'NO', 'serial'),
            ('margin_history', 2, 'timestamp', 'timestamp', 'YES', 'now'),
            ('margin_history', 3, 'balance', 'float4', 'YES', 'none'),
            ('margin_history', 4, 'used_margin', 'float4', 'YES', 'none'),
            ('margin_history', 5, 'open_positions', 'int4', 'YES', 'none'),
            ('trading_session_resets', 1, 'id', 'int4', 'NO', 'serial'),
            ('trading_session_resets', 2, 'reset_timestamp', 'timestamp', 'YES', 'now'),
            ('trading_session_resets', 3, 'reason', 'text', 'YES', 'none'),
            ('model_predictions', 1, 'id', 'int4', 'NO', 'serial'),
            ('model_predictions', 2, 'created_at', 'timestamp', 'YES', 'now'),
            ('model_predictions', 3, 'symbol', 'text', 'YES', 'none'),
            ('model_predictions', 4, 'side', 'text', 'YES', 'none'),
            ('model_predictions', 5, 'model', 'text', 'YES', 'none'),
            ('model_predictions', 6, 'vote', 'text', 'YES', 'none'),
            ('model_predictions', 7, 'scored', 'bool', 'YES', 'false'),
            ('account_balances', 1, 'id', 'int4', 'NO', 'serial'),
            ('account_balances', 2, 'asset', 'text', 'NO', 'none'),
            ('account_balances', 3, 'balance', 'float4', 'NO', 'zero'),
            ('account_balances', 4, 'last_updated', 'timestamp', 'YES', 'now')
    ),
    actual AS (
        SELECT
            table_name,
            ordinal_position,
            column_name,
            udt_name,
            is_nullable,
            CASE
                WHEN column_default IS NULL THEN 'none'
                WHEN column_name = 'id'
                 AND column_default = FORMAT(
                     'nextval(%L::regclass)',
                     pg_get_serial_sequence(
                         FORMAT('%I.%I', table_schema, table_name),
                         column_name
                     )
                 ) THEN 'serial'
                WHEN LOWER(column_default) IN ('now()', 'current_timestamp')
                    THEN 'now'
                WHEN LOWER(column_default) = 'false' THEN 'false'
                WHEN column_default = '0' THEN 'zero'
                ELSE 'other'
            END AS default_kind
        FROM information_schema.columns
        WHERE table_schema = 'np'
          AND table_name IN (
              'trades',
              'open_positions',
              'liquidations',
              'margin_history',
              'trading_session_resets',
              'model_predictions',
              'account_balances'
          )
    ),
    differences AS (
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        UNION ALL
        (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT EXISTS (SELECT 1 FROM differences) INTO layout_mismatch;

    WITH expected_constraints (
        table_name,
        constraint_count,
        has_asset_unique
    ) AS (
        VALUES
            ('trades', 1, FALSE),
            ('open_positions', 1, FALSE),
            ('liquidations', 1, FALSE),
            ('margin_history', 1, FALSE),
            ('trading_session_resets', 1, FALSE),
            ('model_predictions', 1, FALSE),
            ('account_balances', 2, TRUE)
    ),
    actual_constraints AS (
        SELECT
            table_row.relname AS table_name,
            COUNT(*)::INTEGER AS constraint_count,
            BOOL_OR(
                constraint_row.contype = 'p'
                AND constraint_row.conkey = ARRAY[1]::SMALLINT[]
                AND NOT constraint_row.condeferrable
                AND NOT constraint_row.condeferred
                AND constraint_row.convalidated
            ) AS has_id_primary_key,
            BOOL_OR(
                constraint_row.contype = 'u'
                AND constraint_row.conkey = ARRAY[2]::SMALLINT[]
                AND NOT constraint_row.condeferrable
                AND NOT constraint_row.condeferred
                AND constraint_row.convalidated
            ) AS has_asset_unique
        FROM pg_constraint constraint_row
        JOIN pg_class table_row
          ON table_row.oid = constraint_row.conrelid
        JOIN pg_namespace namespace_row
          ON namespace_row.oid = table_row.relnamespace
        WHERE namespace_row.nspname = 'np'
          AND table_row.relname IN (
              'trades',
              'open_positions',
              'liquidations',
              'margin_history',
              'trading_session_resets',
              'model_predictions',
              'account_balances'
          )
        GROUP BY table_row.relname
    )
    SELECT EXISTS (
        SELECT 1
        FROM expected_constraints expected_row
        LEFT JOIN actual_constraints actual_row
          ON actual_row.table_name = expected_row.table_name
        WHERE actual_row.table_name IS NULL
           OR actual_row.constraint_count <> expected_row.constraint_count
           OR NOT actual_row.has_id_primary_key
           OR actual_row.has_asset_unique <> expected_row.has_asset_unique
    ) INTO constraint_mismatch;

    SELECT EXISTS (
        SELECT 1
        FROM (
            VALUES
                ('trades'),
                ('open_positions'),
                ('liquidations'),
                ('margin_history'),
                ('trading_session_resets'),
                ('model_predictions'),
                ('account_balances')
        ) AS required(table_name)
        WHERE NOT EXISTS (
            SELECT 1
            FROM pg_class table_row
            JOIN pg_namespace namespace_row
              ON namespace_row.oid = table_row.relnamespace
            WHERE namespace_row.nspname = 'np'
              AND table_row.relname = required.table_name
              AND table_row.relkind = 'r'
              AND table_row.relpersistence = 'p'
        )
    ) INTO relation_mismatch;

    SELECT EXISTS (
        SELECT 1
        FROM (
            VALUES
                ('trades'),
                ('open_positions'),
                ('liquidations'),
                ('margin_history'),
                ('trading_session_resets'),
                ('model_predictions'),
                ('account_balances')
        ) AS required(table_name)
        WHERE NOT EXISTS (
            SELECT 1
            FROM pg_class sequence_row
            WHERE sequence_row.oid = pg_get_serial_sequence(
                FORMAT('%I.%I', 'np', required.table_name),
                'id'
            )::REGCLASS
              AND sequence_row.relkind = 'S'
              AND sequence_row.relpersistence = 'p'
        )
    ) INTO sequence_mismatch;

    SELECT EXISTS (
        SELECT 1
        FROM (
            VALUES
                ('trades'),
                ('open_positions'),
                ('liquidations'),
                ('margin_history'),
                ('trading_session_resets'),
                ('model_predictions'),
                ('account_balances')
        ) AS required(table_name)
        JOIN pg_class table_row
          ON table_row.relname = required.table_name
        JOIN pg_namespace namespace_row
          ON namespace_row.oid = table_row.relnamespace
         AND namespace_row.nspname = 'np'
        WHERE table_row.relhasrules
           OR table_row.relhastriggers
           OR table_row.relrowsecurity
           OR table_row.relforcerowsecurity
           OR EXISTS (
               SELECT 1
               FROM pg_inherits inheritance_row
               WHERE inheritance_row.inhrelid = table_row.oid
                  OR inheritance_row.inhparent = table_row.oid
           )
           OR EXISTS (
               SELECT 1
               FROM pg_policy policy_row
               WHERE policy_row.polrelid = table_row.oid
           )
    ) INTO behavior_mismatch;

    WITH expected_indexes (table_name, index_name, key_columns) AS (
        VALUES
            (
                'model_predictions',
                'idx_model_predictions_scored',
                '7 2'
            ),
            (
                'trades',
                'idx_trades_symbol_ts',
                '3 2'
            )
    )
    SELECT EXISTS (
        SELECT 1
        FROM expected_indexes expected_index
        WHERE NOT EXISTS (
            SELECT 1
            FROM pg_index index_row
            JOIN pg_class table_row
              ON table_row.oid = index_row.indrelid
            JOIN pg_namespace table_namespace
              ON table_namespace.oid = table_row.relnamespace
            JOIN pg_class index_relation
              ON index_relation.oid = index_row.indexrelid
            JOIN pg_am access_method
              ON access_method.oid = index_relation.relam
            WHERE table_namespace.nspname = 'np'
              AND table_row.relname = expected_index.table_name
              AND index_relation.relname = expected_index.index_name
              AND access_method.amname = 'btree'
              AND index_row.indkey::TEXT = expected_index.key_columns
              AND index_row.indnkeyatts = 2
              AND index_row.indnatts = 2
              AND index_row.indisvalid
              AND index_row.indisready
              AND NOT index_row.indisunique
              AND index_row.indpred IS NULL
              AND index_row.indexprs IS NULL
        )
    ) INTO index_mismatch;

    SELECT EXISTS (
        SELECT 1
        FROM pg_index index_row
        JOIN pg_class table_row
          ON table_row.oid = index_row.indrelid
        JOIN pg_namespace namespace_row
          ON namespace_row.oid = table_row.relnamespace
        LEFT JOIN pg_constraint constraint_row
          ON constraint_row.conindid = index_row.indexrelid
        WHERE namespace_row.nspname = 'np'
          AND table_row.relname IN (
              'trades',
              'open_positions',
              'liquidations',
              'margin_history',
              'trading_session_resets',
              'model_predictions',
              'account_balances'
          )
          AND index_row.indisunique
          AND constraint_row.oid IS NULL
    ) INTO unexpected_unique_index;

    IF layout_mismatch
       OR constraint_mismatch
       OR relation_mismatch
       OR sequence_mismatch
       OR behavior_mismatch
       OR index_mismatch
       OR unexpected_unique_index THEN
        RAISE EXCEPTION 'legacy table layout is incompatible with baseline 0001';
    END IF;
END
$migration$;
