#!/usr/bin/env python3
"""
Test position risk calculations
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import statistics

from utils.paper_trade_db import get_all_trades

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_position_risk_calculations():
    """Test position risk calculations for dashboard"""
    logger.info("=== Testing Position Risk Calculations ===")

    # Get portfolio data
    portfolio_value = 12500.0  # Example current portfolio

    # Get recent trades for risk assessment
    recent_trades = get_all_trades(limit=20)
    logger.info(f"Found {len(recent_trades)} recent trades for risk analysis")

    if recent_trades:
        # Calculate risk metrics
        recent_volumes = []
        recent_pnls = []

        logger.info("\nRecent trades analysis:")
        for i, trade in enumerate(recent_trades[:5]):
            if len(trade) >= 7:
                volume = float(trade[4]) * float(trade[5])  # price * quantity
                pnl = float(trade[6])
                recent_volumes.append(volume)
                recent_pnls.append(pnl)

                logger.info(
                    f"Trade {i+1}: {trade[3]} | Volume: ${volume:,.2f} | PnL: ${pnl:.2f}"
                )

        if recent_volumes:
            avg_trade_size = sum(recent_volumes) / len(recent_volumes)
            max_trade_size = max(recent_volumes)
            min_trade_size = min(recent_volumes)

            # Calculate risk percentages
            avg_risk_pct = (
                (avg_trade_size / portfolio_value * 100) if portfolio_value > 0 else 0
            )
            max_risk_pct = (
                (max_trade_size / portfolio_value * 100) if portfolio_value > 0 else 0
            )

            logger.info(f"\n=== RISK METRICS ===")
            logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
            logger.info(
                f"Average Trade Size: ${avg_trade_size:,.2f} ({avg_risk_pct:.1f}% of portfolio)"
            )
            logger.info(
                f"Maximum Trade Size: ${max_trade_size:,.2f} ({max_risk_pct:.1f}% of portfolio)"
            )
            logger.info(f"Minimum Trade Size: ${min_trade_size:,.2f}")

            # Calculate PnL volatility
            if len(recent_pnls) > 1:
                pnl_std = statistics.stdev(recent_pnls)
                pnl_mean = sum(recent_pnls) / len(recent_pnls)
                logger.info(f"PnL Volatility (Std Dev): ${pnl_std:.2f}")
                logger.info(f"Average PnL per Trade: ${pnl_mean:.2f}")

                # Risk assessment
                if avg_risk_pct > 10:
                    logger.warning(
                        "⚠️  HIGH RISK: Average trade size > 10% of portfolio"
                    )
                elif avg_risk_pct > 5:
                    logger.info(
                        "⚡ MODERATE RISK: Average trade size 5-10% of portfolio"
                    )
                else:
                    logger.info("✅ LOW RISK: Average trade size < 5% of portfolio")

    # Test theoretical position sizing
    logger.info(f"\n=== THEORETICAL POSITION SIZING ===")
    current_price = 107500.0
    leverage = 10
    risk_levels = [0.01, 0.02, 0.05]  # 1%, 2%, 5% risk

    logger.info(f"Current BTC Price: ${current_price:,.2f}")
    logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
    logger.info(f"Leverage: {leverage}x")
    logger.info(f"\nPosition Sizing Options:")

    for risk_pct in risk_levels:
        risk_amount = portfolio_value * risk_pct
        position_value = risk_amount * leverage
        position_size = position_value / current_price

        logger.info(
            f"  {risk_pct*100:.0f}% Risk: {position_size:.4f} BTC (${position_value:,.0f} exposure)"
        )


def test_dashboard_display_format():
    """Test the dashboard display format"""
    logger.info(f"\n=== DASHBOARD DISPLAY FORMAT ===")

    # Simulate what would be shown in dashboard
    portfolio_value = 12500.0
    recent_trades = get_all_trades(limit=20)

    if recent_trades:
        recent_volumes = []
        recent_pnls = []

        for trade in recent_trades:
            if len(trade) >= 7:
                volume = float(trade[4]) * float(trade[5])
                pnl = float(trade[6])
                recent_volumes.append(volume)
                recent_pnls.append(pnl)

        if recent_volumes:
            avg_trade_size = sum(recent_volumes) / len(recent_volumes)
            max_trade_size = max(recent_volumes)
            avg_risk_pct = avg_trade_size / portfolio_value * 100
            max_risk_pct = max_trade_size / portfolio_value * 100

            if len(recent_pnls) > 1:
                pnl_std = statistics.stdev(recent_pnls)
            else:
                pnl_std = 0

            logger.info("--- Position Risk ---")
            logger.info(f"No open positions")
            logger.info(f"Avg Trade Size: ${avg_trade_size:,.2f} ({avg_risk_pct:.1f}%)")
            logger.info(f"Max Trade Size: ${max_trade_size:,.2f} ({max_risk_pct:.1f}%)")
            logger.info(f"PnL Volatility: ${pnl_std:.2f}")


if __name__ == "__main__":
    logger.info("Starting Position Risk Tests...")

    # Test 1: Risk calculations
    test_position_risk_calculations()

    # Test 2: Dashboard format
    test_dashboard_display_format()

    logger.info("Position risk tests completed!")
