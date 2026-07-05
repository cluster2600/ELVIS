#!/usr/bin/env python3
"""
Test script for live trading with RL integration
This script simulates real trading conditions and tests the RL model performance
"""

import logging
import random
import time
from datetime import datetime, timedelta

from analyze_trades import analyze_recent_trades
from trading.strategies.ensemble_strategy import EnsembleStrategy
from utils.paper_trade_db import get_all_trades, record_trade

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simulate_market_data(base_price=119000.0, volatility=0.02):
    """Simulate realistic market data"""
    # Add some random price movement
    price_change = random.uniform(-volatility, volatility)
    current_price = base_price * (1 + price_change)

    # Generate realistic technical indicators
    market_data = {
        "price": current_price,
        "close": current_price,
        "high": current_price * 1.001,
        "low": current_price * 0.999,
        "volume": random.uniform(800, 1200),
        "rsi": random.uniform(30, 70),
        "macd": random.uniform(-0.5, 0.5),
        "macd_signal": random.uniform(-0.3, 0.3),
        "macd_histogram": random.uniform(-0.2, 0.2),
        "bb_upper": current_price * 1.02,
        "bb_lower": current_price * 0.98,
        "bb_middle": current_price,
        "atr": random.uniform(300, 800),
        "adx": random.uniform(20, 40),
        "sma_20": current_price * random.uniform(0.995, 1.005),
        "sma_50": current_price * random.uniform(0.990, 1.010),
    }

    return market_data


def simulate_trade_execution(signal, price, confidence):
    """Simulate trade execution and calculate results"""
    # Simulate some randomness in trade outcomes
    base_pnl = 0.0

    if signal == "BUY":
        # Simulate buy trade outcome
        success_prob = min(
            0.4 + confidence * 0.3, 0.8
        )  # Higher confidence = better success rate
        if random.random() < success_prob:
            base_pnl = random.uniform(0.5, 2.0)  # Profitable trade
        else:
            base_pnl = random.uniform(-1.5, -0.2)  # Loss
    elif signal == "SELL":
        # Simulate sell trade outcome
        success_prob = min(0.4 + confidence * 0.3, 0.8)
        if random.random() < success_prob:
            base_pnl = random.uniform(0.5, 2.0)  # Profitable trade
        else:
            base_pnl = random.uniform(-1.5, -0.2)  # Loss
    else:
        base_pnl = 0.0  # HOLD

    # Add trading fees
    fee = 0.48 if signal != "HOLD" else 0.0

    return {
        "price": price,
        "pnl": base_pnl,
        "fees": fee,
        "side": signal,
        "quantity": 0.001,
        "net_result": base_pnl - fee,
    }


def run_trading_simulation(duration_minutes=10, trade_interval=30):
    """Run a trading simulation with RL integration"""
    logger.info(f"Starting RL trading simulation for {duration_minutes} minutes...")

    # Initialize ensemble strategy with RL enabled
    ensemble_strategy = EnsembleStrategy(
        logger=logger,
        symbols=["BTCUSDT"],
        enable_rl_strategy=True,
        enable_research_strategy=False,  # Disable for simpler testing
    )

    # Simulation parameters
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    base_price = 119000.0

    # Track simulation results
    simulation_results = {
        "total_trades": 0,
        "profitable_trades": 0,
        "total_pnl": 0.0,
        "total_fees": 0.0,
        "rl_predictions": [],
        "trade_results": [],
    }

    logger.info("Starting simulation loop...")

    while datetime.now() < end_time:
        try:
            # Generate market data
            market_data = simulate_market_data(base_price)
            current_price = market_data["price"]

            # Get RL prediction
            signal, confidence = ensemble_strategy.generate_signal(
                "BTCUSDT", market_data
            )

            # Log the prediction
            logger.info(
                f"RL Signal: {signal} (confidence: {confidence:.3f}) at price ${current_price:.2f}"
            )

            # Only execute trades on BUY/SELL signals
            if signal in ["BUY", "SELL"]:
                # Simulate trade execution
                trade_result = simulate_trade_execution(
                    signal, current_price, confidence
                )

                # Record the trade in database
                record_trade(
                    symbol="BTCUSDT",
                    side=signal,
                    price=current_price,
                    quantity=trade_result["quantity"],
                    pnl=trade_result["pnl"],
                    fee=trade_result["fees"],
                )

                # Update RL model with trade result
                ensemble_strategy.update_rl_with_trade_result(trade_result)

                # Track simulation results
                simulation_results["total_trades"] += 1
                simulation_results["total_pnl"] += trade_result["pnl"]
                simulation_results["total_fees"] += trade_result["fees"]

                if trade_result["net_result"] > 0:
                    simulation_results["profitable_trades"] += 1

                simulation_results["trade_results"].append(
                    {
                        "timestamp": datetime.now(),
                        "signal": signal,
                        "confidence": confidence,
                        "price": current_price,
                        "pnl": trade_result["pnl"],
                        "fees": trade_result["fees"],
                        "net_result": trade_result["net_result"],
                    }
                )

                logger.info(
                    f"Trade executed: {signal} at ${current_price:.2f} -> PnL: ${trade_result['pnl']:.2f}, Fee: ${trade_result['fees']:.2f}"
                )

            # Track all predictions
            simulation_results["rl_predictions"].append(
                {
                    "timestamp": datetime.now(),
                    "signal": signal,
                    "confidence": confidence,
                    "price": current_price,
                }
            )

            # Update base price for next iteration (simulate market movement)
            base_price = current_price * random.uniform(0.998, 1.002)

            # Wait before next iteration
            time.sleep(trade_interval)

        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            continue

    logger.info("Simulation completed!")

    # Calculate final results
    net_result = simulation_results["total_pnl"] - simulation_results["total_fees"]
    win_rate = (
        (simulation_results["profitable_trades"] / simulation_results["total_trades"])
        * 100
        if simulation_results["total_trades"] > 0
        else 0
    )

    logger.info("=== SIMULATION RESULTS ===")
    logger.info(f"Duration: {duration_minutes} minutes")
    logger.info(f"Total predictions: {len(simulation_results['rl_predictions'])}")
    logger.info(f"Total trades: {simulation_results['total_trades']}")
    logger.info(f"Profitable trades: {simulation_results['profitable_trades']}")
    logger.info(f"Win rate: {win_rate:.1f}%")
    logger.info(f"Total PnL: ${simulation_results['total_pnl']:.2f}")
    logger.info(f"Total fees: ${simulation_results['total_fees']:.2f}")
    logger.info(f"Net result: ${net_result:.2f}")

    # Get RL performance metrics
    rl_metrics = ensemble_strategy.get_rl_performance_metrics()
    logger.info(f"RL Performance Metrics: {rl_metrics}")

    return simulation_results


def main():
    """Main function"""
    try:
        # Run simulation
        results = run_trading_simulation(duration_minutes=5, trade_interval=10)

        # Analyze recent trades after simulation
        logger.info("\n" + "=" * 60)
        logger.info("ANALYZING RECENT TRADES AFTER SIMULATION")
        logger.info("=" * 60)
        analyze_recent_trades()

        # Show final summary
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Simulation completed successfully!")
        logger.info(f"Total trades executed: {results['total_trades']}")
        logger.info(f"Net result: ${results['total_pnl'] - results['total_fees']:.2f}")
        logger.info(
            f"The RL model has been updated with {results['total_trades']} new trading experiences"
        )

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
