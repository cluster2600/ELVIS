#!/usr/bin/env python3
"""
Comprehensive Binance fee analysis including BNB discounts and BNB trading options
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set environment
os.environ["VAULT_ENABLED"] = "false"


class EnhancedBinanceFeeCalculator:
    """
    Enhanced Binance fee calculator with BNB discount support and multi-asset trading
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # Binance Fee Structure 2025
        # Standard Futures Fees (VIP 0)
        self.futures_maker_fee = 0.0002  # 0.02%
        self.futures_taker_fee = 0.0004  # 0.04% (was 0.05% in older versions)

        # Standard Spot Fees (VIP 0)
        self.spot_maker_fee = 0.001  # 0.1%
        self.spot_taker_fee = 0.001  # 0.1%

        # BNB Discounts (2025 rates)
        self.bnb_futures_discount = 0.10  # 10% discount on futures when paying with BNB
        self.bnb_spot_discount = 0.25  # 25% discount on spot when paying with BNB

        # Funding rates and borrowing
        self.typical_funding_rate = 0.0001  # 0.01% every 8 hours
        self.funding_interval_hours = 8
        self.borrowing_interest_annual = 0.0365  # 3.65% annual

        # BNB current price (would be fetched live in real implementation)
        self.bnb_price_usd = 300.0  # Approximate BNB price

    def get_effective_fee_rate(
        self, base_fee: float, use_bnb: bool, is_futures: bool = True
    ) -> float:
        """Get effective fee rate with BNB discount applied"""
        if use_bnb:
            discount = (
                self.bnb_futures_discount if is_futures else self.bnb_spot_discount
            )
            return base_fee * (1 - discount)
        return base_fee

    def calculate_trading_fee(
        self,
        price: float,
        quantity: float,
        is_maker: bool = False,
        is_futures: bool = True,
        use_bnb: bool = False,
    ) -> dict:
        """
        Calculate trading fee with optional BNB discount

        Returns dict with fee amounts in both USDT and BNB
        """
        trade_value = price * quantity

        if is_futures:
            base_fee_rate = (
                self.futures_maker_fee if is_maker else self.futures_taker_fee
            )
        else:
            base_fee_rate = self.spot_maker_fee if is_maker else self.spot_taker_fee

        # Calculate fees with and without BNB
        standard_fee_usdt = trade_value * base_fee_rate
        effective_fee_rate = self.get_effective_fee_rate(
            base_fee_rate, use_bnb, is_futures
        )
        discounted_fee_usdt = trade_value * effective_fee_rate

        # BNB equivalent (BNB is used to pay the discounted fee)
        fee_in_bnb = discounted_fee_usdt / self.bnb_price_usd if use_bnb else 0

        savings = standard_fee_usdt - discounted_fee_usdt if use_bnb else 0

        return {
            "trade_value": trade_value,
            "standard_fee_usdt": standard_fee_usdt,
            "discounted_fee_usdt": discounted_fee_usdt,
            "fee_in_bnb": fee_in_bnb,
            "savings_usdt": savings,
            "discount_percent": (
                (savings / standard_fee_usdt * 100) if standard_fee_usdt > 0 else 0
            ),
            "effective_fee_rate": effective_fee_rate,
        }

    def analyze_bnb_vs_btc_trading(
        self, btc_price: float, bnb_price: float, trade_amount_usdt: float
    ) -> dict:
        """
        Compare trading BNB/USDT vs BTC/USDT with fee considerations
        """
        # BTC trade analysis
        btc_quantity = trade_amount_usdt / btc_price
        btc_fee_standard = self.calculate_trading_fee(
            btc_price, btc_quantity, is_futures=True, use_bnb=False
        )
        btc_fee_bnb = self.calculate_trading_fee(
            btc_price, btc_quantity, is_futures=True, use_bnb=True
        )

        # BNB trade analysis
        bnb_quantity = trade_amount_usdt / bnb_price
        bnb_fee_standard = self.calculate_trading_fee(
            bnb_price, bnb_quantity, is_futures=True, use_bnb=False
        )
        bnb_fee_bnb = self.calculate_trading_fee(
            bnb_price, bnb_quantity, is_futures=True, use_bnb=True
        )

        return {
            "trade_amount_usdt": trade_amount_usdt,
            "btc": {
                "price": btc_price,
                "quantity": btc_quantity,
                "fees_standard": btc_fee_standard,
                "fees_with_bnb": btc_fee_bnb,
            },
            "bnb": {
                "price": bnb_price,
                "quantity": bnb_quantity,
                "fees_standard": bnb_fee_standard,
                "fees_with_bnb": bnb_fee_bnb,
            },
        }

    def calculate_bnb_breakeven_volume(self, is_futures: bool = True) -> dict:
        """
        Calculate minimum trading volume to justify buying BNB for fee discounts
        """
        discount = self.bnb_futures_discount if is_futures else self.bnb_spot_discount
        base_fee = self.futures_taker_fee if is_futures else self.spot_taker_fee

        # Assume we need to buy minimum 0.1 BNB for fee payments
        min_bnb_purchase = 0.1
        min_bnb_cost = min_bnb_purchase * self.bnb_price_usd

        # Calculate breakeven trading volume
        # Savings per dollar traded = base_fee * discount
        savings_per_dollar = base_fee * discount
        breakeven_volume = min_bnb_cost / savings_per_dollar

        return {
            "min_bnb_purchase": min_bnb_purchase,
            "min_bnb_cost_usdt": min_bnb_cost,
            "savings_per_dollar_traded": savings_per_dollar,
            "breakeven_volume_usdt": breakeven_volume,
            "discount_percent": discount * 100,
            "market_type": "futures" if is_futures else "spot",
        }


def fetch_current_prices():
    """Fetch current BTC and BNB prices"""
    try:
        from utils.price_fetcher import PriceFetcher

        price_fetcher = PriceFetcher(logger=logger, symbols=["BTCUSDT", "BNBUSDT"])

        # Get BTC price
        btc_df = price_fetcher.get_historical_klines("BTCUSDT", "5m", 1)
        btc_price = float(btc_df.iloc[-1]["close"]) if not btc_df.empty else 100000.0

        # Get BNB price
        bnb_df = price_fetcher.get_historical_klines("BNBUSDT", "5m", 1)
        bnb_price = float(bnb_df.iloc[-1]["close"]) if not bnb_df.empty else 300.0

        return btc_price, bnb_price

    except Exception as e:
        logger.warning(f"Could not fetch live prices: {e}")
        return 100000.0, 300.0  # Fallback prices


def main():
    """Main analysis function"""
    logger.info("🔍 Binance Fee Analysis with BNB Integration")
    logger.info("=" * 70)

    # Initialize calculator
    calc = EnhancedBinanceFeeCalculator(logger)

    # Fetch current prices
    btc_price, bnb_price = fetch_current_prices()
    calc.bnb_price_usd = bnb_price

    logger.info(f"📊 Current Prices:")
    logger.info(f"BTC: ${btc_price:,.2f}")
    logger.info(f"BNB: ${bnb_price:,.2f}")
    logger.info("-" * 70)

    # 1. Fee comparison for different trade sizes
    trade_sizes = [100, 1000, 10000, 100000]  # USDT

    logger.info("💰 Trading Fee Comparison (Futures, Taker Fees)")
    logger.info(
        f"{'Trade Size':<12} {'Standard Fee':<15} {'With BNB':<15} {'Savings':<12} {'Savings %'}"
    )
    logger.info("-" * 70)

    for trade_size in trade_sizes:
        btc_quantity = trade_size / btc_price

        fee_standard = calc.calculate_trading_fee(
            btc_price, btc_quantity, is_futures=True, use_bnb=False
        )
        fee_bnb = calc.calculate_trading_fee(
            btc_price, btc_quantity, is_futures=True, use_bnb=True
        )

        logger.info(
            f"${trade_size:<11,} ${fee_standard['standard_fee_usdt']:<14.6f} "
            f"${fee_bnb['discounted_fee_usdt']:<14.6f} ${fee_bnb['savings_usdt']:<11.6f} "
            f"{fee_bnb['discount_percent']:<11.1f}%"
        )

    logger.info("-" * 70)

    # 2. BNB vs BTC trading comparison
    logger.info("🆚 BNB vs BTC Trading Comparison ($10,000 trade)")
    comparison = calc.analyze_bnb_vs_btc_trading(btc_price, bnb_price, 10000)

    logger.info("\nBTC/USDT Trade:")
    logger.info(f"  Quantity: {comparison['btc']['quantity']:.6f} BTC")
    logger.info(
        f"  Standard fee: ${comparison['btc']['fees_standard']['standard_fee_usdt']:.6f}"
    )
    logger.info(
        f"  With BNB fee: ${comparison['btc']['fees_with_bnb']['discounted_fee_usdt']:.6f}"
    )
    logger.info(
        f"  BNB savings: ${comparison['btc']['fees_with_bnb']['savings_usdt']:.6f}"
    )

    logger.info("\nBNB/USDT Trade:")
    logger.info(f"  Quantity: {comparison['bnb']['quantity']:.6f} BNB")
    logger.info(
        f"  Standard fee: ${comparison['bnb']['fees_standard']['standard_fee_usdt']:.6f}"
    )
    logger.info(
        f"  With BNB fee: ${comparison['bnb']['fees_with_bnb']['discounted_fee_usdt']:.6f}"
    )
    logger.info(
        f"  BNB savings: ${comparison['bnb']['fees_with_bnb']['savings_usdt']:.6f}"
    )

    # 3. BNB breakeven analysis
    logger.info("\n💡 BNB Investment Breakeven Analysis")
    futures_breakeven = calc.calculate_bnb_breakeven_volume(is_futures=True)
    spot_breakeven = calc.calculate_bnb_breakeven_volume(is_futures=False)

    logger.info(f"\nFutures Trading:")
    logger.info(
        f"  Min BNB needed: {futures_breakeven['min_bnb_purchase']} BNB (${futures_breakeven['min_bnb_cost_usdt']:.2f})"
    )
    logger.info(f"  Discount: {futures_breakeven['discount_percent']:.0f}%")
    logger.info(
        f"  Breakeven volume: ${futures_breakeven['breakeven_volume_usdt']:,.0f}"
    )

    logger.info(f"\nSpot Trading:")
    logger.info(
        f"  Min BNB needed: {spot_breakeven['min_bnb_purchase']} BNB (${spot_breakeven['min_bnb_cost_usdt']:.2f})"
    )
    logger.info(f"  Discount: {spot_breakeven['discount_percent']:.0f}%")
    logger.info(f"  Breakeven volume: ${spot_breakeven['breakeven_volume_usdt']:,.0f}")

    # 4. Recommendations
    logger.info("\n🎯 RECOMMENDATIONS:")
    logger.info("-" * 70)

    if futures_breakeven["breakeven_volume_usdt"] < 1000:
        logger.info("✅ BNB fee discount is HIGHLY RECOMMENDED for futures trading")
        logger.info(
            f"   You break even with just ${futures_breakeven['breakeven_volume_usdt']:,.0f} in trading volume"
        )
    else:
        logger.info("⚠️ BNB fee discount for futures: consider your trading volume")

    if spot_breakeven["breakeven_volume_usdt"] < 1000:
        logger.info("✅ BNB fee discount is HIGHLY RECOMMENDED for spot trading")
    else:
        logger.info(
            "⚠️ BNB fee discount for spot: only beneficial for high-volume traders"
        )

    logger.info("\n💡 Additional Benefits of holding BNB:")
    logger.info("   • Priority access to Binance Launchpool")
    logger.info("   • Higher withdrawal limits")
    logger.info("   • Access to special BNB-only trading pairs")
    logger.info("   • Potential BNB price appreciation")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
