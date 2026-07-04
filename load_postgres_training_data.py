#!/usr/bin/env python3
"""
Load all available trading data from PostgreSQL for LLM training.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.logging_utils import setup_logger
from utils.paper_trade_db import get_conn


def load_all_postgres_trades(include_test_trades=False):
    """Load all trading data from PostgreSQL database"""

    logger = setup_logger("postgres_loader")
    logger.info("🐘 Loading all training data from PostgreSQL...")

    try:
        conn = get_conn()

        # Build the query based on whether to include TEST trades
        if include_test_trades:
            query = """
            SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
            FROM trades
            ORDER BY timestamp ASC
            """
            logger.info("📊 Including TEST trades in training data")
        else:
            query = """
            SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
            FROM trades
            WHERE side != 'TEST'
            ORDER BY timestamp ASC
            """
            logger.info("📊 Excluding TEST trades, using only BUY/SELL trades")

        # Load data into DataFrame
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.warning("⚠️ No trading data found in PostgreSQL!")
            return pd.DataFrame()

        logger.info(f"✅ Loaded {len(df)} trades from PostgreSQL")
        logger.info(
            f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
        )
        logger.info(f"   Symbols: {df['symbol'].unique().tolist()}")
        logger.info(f"   Sides: {df['side'].value_counts().to_dict()}")

        return df

    except Exception as e:
        logger.error(f"❌ Failed to load PostgreSQL data: {e}")
        raise


def postgres_trades_to_ohlcv(df, timeframe="5min"):
    """Convert individual trades to OHLCV format for model training"""

    logger = setup_logger("ohlcv_converter")
    logger.info(f"🔄 Converting {len(df)} trades to {timeframe} OHLCV format...")

    if df.empty:
        return pd.DataFrame()

    # Convert timestamp to datetime and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # For small datasets, we'll aggregate by shorter intervals to get more samples
    if len(df) < 100:
        # Use 1-minute intervals for very small datasets
        freq = "1T"
        logger.info("📈 Using 1-minute intervals (small dataset)")
    elif len(df) < 500:
        # Use 5-minute intervals for small datasets
        freq = "5T"
        logger.info("📈 Using 5-minute intervals")
    else:
        # Use 15-minute intervals for larger datasets
        freq = "15T"
        logger.info("📈 Using 15-minute intervals")

    # Group by time intervals and create OHLCV
    try:
        # Create OHLCV aggregation
        ohlcv = (
            df.groupby(
                [
                    pd.Grouper(freq=freq),  # Time grouping
                    "symbol",  # Group by symbol as well
                ]
            )
            .agg(
                {
                    "price": ["first", "max", "min", "last"],  # OHLC
                    "quantity": "sum",  # Volume (sum of quantities)
                    "pnl": "sum",  # Total PnL for the period
                    "fee": "sum",  # Total fees
                    "id": "count",  # Number of trades in this period
                }
            )
            .dropna()
        )

        # Flatten column names
        ohlcv.columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "pnl_total",
            "fees_total",
            "trade_count",
        ]

        # Reset index to get timestamp and symbol as columns
        ohlcv = ohlcv.reset_index()

        # Filter out periods with no trades
        ohlcv = ohlcv[ohlcv["trade_count"] > 0]

        if len(ohlcv) == 0:
            logger.warning("⚠️ No OHLCV data generated after aggregation")
            return pd.DataFrame()

        logger.info(f"✅ Created {len(ohlcv)} OHLCV periods")
        logger.info(
            f"   Price range: ${ohlcv['low'].min():,.2f} - ${ohlcv['high'].max():,.2f}"
        )
        logger.info(f"   Average trades per period: {ohlcv['trade_count'].mean():.1f}")

        return ohlcv

    except Exception as e:
        logger.error(f"❌ Failed to convert to OHLCV: {e}")
        raise


def enhance_postgres_training_data(df):
    """Add features and targets to PostgreSQL trading data"""

    logger = setup_logger("data_enhancer")
    logger.info(f"🎯 Enhancing {len(df)} samples with features and targets...")

    if df.empty:
        return df

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Add basic features
    df["price_change"] = df["close"].pct_change()
    df["price_change_abs"] = df["price_change"].abs()
    df["price_volatility"] = df["price_change"].rolling(5, min_periods=1).std()

    # Add moving averages
    df["sma_5"] = df["close"].rolling(5, min_periods=1).mean()
    df["sma_10"] = df["close"].rolling(10, min_periods=1).mean()

    # Add RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(10, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Add MACD
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Add Bollinger Bands
    bb_window = 10
    bb_middle = df["close"].rolling(bb_window, min_periods=1).mean()
    bb_std = df["close"].rolling(bb_window, min_periods=1).std()
    df["bb_upper"] = bb_middle + (bb_std * 2)
    df["bb_lower"] = bb_middle - (bb_std * 2)
    df["bb_middle"] = bb_middle

    # Add volume features
    df["volume_sma"] = df["volume"].rolling(5, min_periods=1).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)

    # Create targets for next period prediction
    df["future_price"] = df["close"].shift(-1)
    df["future_return"] = (df["future_price"] / df["close"]) - 1

    # Classification target: will price go up in next period?
    df["target_up"] = (df["future_return"] > 0).astype(int)

    # Regression target: how much will price change?
    df["target_return"] = df["future_return"]

    # Fill NaN values
    df = df.fillna(method="ffill").fillna(0)

    # Remove the last row (no future data)
    df = df[:-1]

    logger.info(f"✅ Enhanced data with {len(df.columns)} features")
    logger.info(
        f"   Target distribution: UP={df['target_up'].sum()}, DOWN={len(df) - df['target_up'].sum()}"
    )
    logger.info(
        f"   Return range: {df['target_return'].min():.4f} to {df['target_return'].max():.4f}"
    )

    return df


def main():
    """Test PostgreSQL data loading"""

    logger = setup_logger("postgres_test")

    try:
        # Load all trades
        trades_df = load_all_postgres_trades(include_test_trades=True)
        if trades_df.empty:
            logger.error("No trades loaded!")
            return

        # Convert to OHLCV
        ohlcv_df = postgres_trades_to_ohlcv(trades_df)
        if ohlcv_df.empty:
            logger.error("No OHLCV data generated!")
            return

        # Enhance with features
        enhanced_df = enhance_postgres_training_data(ohlcv_df)

        logger.info(f"\n📊 Final Training Dataset:")
        logger.info(f"   Samples: {len(enhanced_df)}")
        logger.info(f"   Features: {len(enhanced_df.columns)}")
        logger.info(
            f"   Date range: {enhanced_df['timestamp'].min()} to {enhanced_df['timestamp'].max()}"
        )

        # Save to CSV for inspection
        output_file = "postgres_training_data.csv"
        enhanced_df.to_csv(output_file, index=False)
        logger.info(f"💾 Saved training data to: {output_file}")

        return enhanced_df

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
