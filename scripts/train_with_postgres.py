#!/usr/bin/env python3
"""
Training script that works with existing PostgreSQL setup
No Vault required - uses your existing databases
"""

# Make the repo root importable no matter where this script is run from
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psycopg2


def check_postgres_connection():
    """Check PostgreSQL connection and find the right database"""
    print("🗄️  Checking PostgreSQL connection...")

    # Try different database configurations
    configs = [
        {
            "host": "localhost",
            "port": 5432,
            "user": "maxime",
            "password": "",  # Try without password first
            "dbname": "elvis_trading",
        },
        {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "elvis_password",
            "dbname": "elvis_trading",
        },
        {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "",
            "dbname": "trading_bot",
        },
    ]

    for config in configs:
        try:
            # Try to connect
            conn_str = f"host={config['host']} port={config['port']} dbname={config['dbname']} user={config['user']}"
            if config["password"]:
                conn_str += f" password={config['password']}"

            conn = psycopg2.connect(conn_str)
            cursor = conn.cursor()

            # Check for trades table
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name IN ('trades', 'trade_history', 'paper_trades')
            """)
            table_count = cursor.fetchone()[0]

            if table_count > 0:
                # Get trade count
                for table_name in ["trades", "trade_history", "paper_trades"]:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        trade_count = cursor.fetchone()[0]
                        print(
                            f"✅ Connected to {config['dbname']} - Found {trade_count} trades in {table_name}"
                        )

                        conn.close()
                        return config, table_name, trade_count
                    except:
                        continue

            conn.close()

        except Exception as e:
            print(
                f"❌ Failed to connect with config {config['dbname']}: {str(e)[:50]}..."
            )
            continue

    print("❌ Could not find a working PostgreSQL connection with trade data")
    return None, None, 0


def setup_postgres_environment(db_config):
    """Setup environment for PostgreSQL connection"""
    print("🔧 Setting up PostgreSQL environment...")

    # Disable Vault
    os.environ["VAULT_ENABLED"] = "false"
    os.environ["USE_VAULT"] = "false"

    # Set database configuration
    os.environ["POSTGRES_HOST"] = db_config["host"]
    os.environ["POSTGRES_PORT"] = str(db_config["port"])
    os.environ["POSTGRES_USER"] = db_config["user"]
    os.environ["POSTGRES_PASSWORD"] = db_config.get("password", "")
    os.environ["POSTGRES_DBNAME"] = db_config["dbname"]

    # Set other required variables
    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = "6379"
    os.environ["BINANCE_API_KEY"] = "training_key"
    os.environ["BINANCE_API_SECRET"] = "training_secret"

    print(f"✅ Environment configured for database: {db_config['dbname']}")


def create_sample_data_if_needed():
    """Create sample trade data if database is empty"""
    print("📊 Creating sample trade data for training...")

    sample_script = """
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import os

# Create sample trade data
np.random.seed(42)
start_date = datetime.now() - timedelta(days=30)

trades = []
for i in range(1000):
    trade_time = start_date + timedelta(minutes=i*15)
    price = 50000 + np.random.normal(0, 1000)  # BTC price around 50k
    quantity = np.random.uniform(0.001, 0.1)
    side = np.random.choice(['BUY', 'SELL'])
    pnl = np.random.normal(10, 50)  # Average small profit
    
    trades.append({
        'timestamp': trade_time,
        'symbol': 'BTCUSDT',
        'side': side,
        'price': price,
        'quantity': quantity,
        'pnl': pnl,
        'fee': abs(pnl) * 0.1,
        'strategy': 'paper_trading'
    })

# Connect and insert
try:
    conn = psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT'],
        dbname=os.environ['POSTGRES_DBNAME'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD']
    )
    
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute(\"\"\"
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            symbol VARCHAR(20),
            side VARCHAR(10),
            price DECIMAL(20,8),
            quantity DECIMAL(20,8),
            pnl DECIMAL(20,8),
            fee DECIMAL(20,8),
            strategy VARCHAR(50)
        )
    \"\"\")
    
    # Insert sample data
    for trade in trades:
        cursor.execute(\"\"\"
            INSERT INTO trades (timestamp, symbol, side, price, quantity, pnl, fee, strategy)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        \"\"\", (
            trade['timestamp'], trade['symbol'], trade['side'], 
            trade['price'], trade['quantity'], trade['pnl'], 
            trade['fee'], trade['strategy']
        ))
    
    conn.commit()
    print(f"✅ Inserted {len(trades)} sample trades")
    
except Exception as e:
    print(f"❌ Failed to create sample data: {e}")
finally:
    if conn:
        conn.close()
"""

    with open("create_sample_trades.py", "w") as f:
        f.write(sample_script)

    try:
        subprocess.run([sys.executable, "create_sample_trades.py"], check=True)
        print("✅ Sample data created successfully")
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")


def run_training_with_postgres(args, db_config, table_name):
    """Run training with PostgreSQL configuration"""
    print("🚀 Starting training with PostgreSQL...")

    # Setup environment
    setup_postgres_environment(db_config)

    # Set the table name for the training script to use
    os.environ["POSTGRES_TABLE_NAME"] = table_name

    # Build command
    cmd_parts = [
        sys.executable,
        "training/train_models.py",
        "--include-trade-history",
        "--config",
        args.config,
        "--output",
        args.output,
    ]

    if args.debug:
        cmd_parts.append("--debug")
        os.environ["LOG_LEVEL"] = "DEBUG"

    # Set training parameters
    os.environ["TRADE_LIMIT"] = str(args.limit)
    os.environ["PREDICTION_HORIZON"] = str(args.prediction_horizon)
    os.environ["TRAINING_EPOCHS"] = str(args.epochs)

    print(f"📊 Training Configuration:")
    print(f"   Database: {db_config['dbname']} ({table_name})")
    print(f"   Trades: {args.limit} max")
    print(f"   Epochs: {args.epochs}")
    print(f"   Debug: {args.debug}")
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print()

    try:
        result = subprocess.run(cmd_parts, env=os.environ.copy())

        if result.returncode == 0:
            print(f"\n✅ Training completed successfully!")
            print(f"🕒 Finished: {datetime.now().strftime('%H:%M:%S')}")
        else:
            print(f"\n❌ Training failed with code {result.returncode}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Training error: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train ELVIS with PostgreSQL")
    parser.add_argument("--limit", type=int, default=1000, help="Max trades")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument(
        "--prediction-horizon", type=int, default=5, help="Prediction horizon"
    )
    parser.add_argument(
        "--config", default="training/config/model_config.yaml", help="Config"
    )
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--create-sample-data", action="store_true", help="Create sample data"
    )

    args = parser.parse_args()

    print("🎯 ELVIS Training - PostgreSQL Mode")
    print("=" * 40)
    print("🗄️  Using PostgreSQL database")
    print("🔓 Vault authentication: DISABLED")
    print()

    # Check PostgreSQL connection
    db_config, table_name, trade_count = check_postgres_connection()

    if not db_config:
        print("💡 No PostgreSQL connection found. Try:")
        print("   1. Start PostgreSQL: brew services start postgresql")
        print("   2. Check database exists: psql -l")
        print("   3. Create database: createdb elvis_trading")
        return

    if trade_count == 0 and args.create_sample_data:
        create_sample_data_if_needed()
        # Recheck after creating data
        db_config, table_name, trade_count = check_postgres_connection()

    if trade_count > 0:
        print(f"📈 Found {trade_count} trades - proceeding with training")
        os.makedirs(args.output, exist_ok=True)
        run_training_with_postgres(args, db_config, table_name)
    else:
        print("⚠️  No trade data found. Use --create-sample-data to generate test data")
        print("   python train_with_postgres.py --create-sample-data --debug")


if __name__ == "__main__":
    main()
