#!/usr/bin/env python3
"""
Analyze recent trades to understand current trading performance
"""

from datetime import datetime, timedelta

import psycopg2

from utils.paper_trade_db import *


def analyze_recent_trades():
    """Analyze trades from the last 2 hours"""
    conn = get_conn()
    if not conn:
        print("Could not connect to database")
        return

    try:
        c = conn.cursor()

        # Get trades from last 2 hours
        two_hours_ago = datetime.now() - timedelta(hours=2)
        c.execute(
            """
            SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
            FROM trades
            WHERE timestamp >= %s AND side != 'TEST'
            ORDER BY timestamp DESC
        """,
            (two_hours_ago,),
        )

        recent_trades = c.fetchall()
        print(f"Recent trades (last 2 hours): {len(recent_trades)} trades")

        if recent_trades:
            print("\nRecent trades:")
            for trade in recent_trades[:10]:  # Show first 10
                print(
                    f"  {trade[1]} | {trade[2]} | {trade[3]} | Price: ${trade[4]:.2f} | Qty: {trade[5]:.4f} | PnL: ${trade[6]:.2f} | Fee: ${trade[7]:.2f}"
                )

            # Calculate statistics
            total_pnl = sum(trade[6] for trade in recent_trades)
            total_fees = sum(trade[7] for trade in recent_trades)
            winning_trades = [t for t in recent_trades if t[6] > 0]
            losing_trades = [t for t in recent_trades if t[6] < 0]

            print(f"\nStats for last 2 hours:")
            print(f"  Total trades: {len(recent_trades)}")
            print(f"  Total PnL: ${total_pnl:.2f}")
            print(f"  Total fees: ${total_fees:.2f}")
            print(f"  Net result: ${total_pnl - total_fees:.2f}")
            print(
                f"  Winning trades: {len(winning_trades)} ({len(winning_trades)/len(recent_trades)*100:.1f}%)"
            )
            print(
                f"  Losing trades: {len(losing_trades)} ({len(losing_trades)/len(recent_trades)*100:.1f}%)"
            )

            if winning_trades:
                avg_win = sum(t[6] for t in winning_trades) / len(winning_trades)
                print(f"  Average win: ${avg_win:.2f}")
            if losing_trades:
                avg_loss = sum(t[6] for t in losing_trades) / len(losing_trades)
                print(f"  Average loss: ${avg_loss:.2f}")

            # Check time intervals between trades
            if len(recent_trades) > 1:
                intervals = []
                for i in range(len(recent_trades) - 1):
                    time_diff = recent_trades[i][1] - recent_trades[i + 1][1]
                    intervals.append(
                        time_diff.total_seconds() / 60
                    )  # Convert to minutes

                avg_interval = sum(intervals) / len(intervals)
                print(f"  Average time between trades: {avg_interval:.1f} minutes")
                print(f"  Min interval: {min(intervals):.1f} minutes")
                print(f"  Max interval: {max(intervals):.1f} minutes")
        else:
            print("No recent trades found")

    except Exception as e:
        print(f"Error analyzing trades: {e}")
    finally:
        conn.close()


def get_all_trades_for_rl():
    """Get all trades for RL training"""
    conn = get_conn()
    if not conn:
        print("Could not connect to database")
        return []

    try:
        c = conn.cursor()
        c.execute("""
            SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
            FROM trades
            WHERE side != 'TEST'
            ORDER BY timestamp ASC
        """)

        trades = c.fetchall()
        print(f"Total trades available for RL: {len(trades)}")
        return trades

    except Exception as e:
        print(f"Error getting trades for RL: {e}")
        return []
    finally:
        conn.close()


if __name__ == "__main__":
    analyze_recent_trades()
    print("\n" + "=" * 50)
    get_all_trades_for_rl()
