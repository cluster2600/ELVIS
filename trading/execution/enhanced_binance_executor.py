#!/usr/bin/env python3.14
"""
Enhanced Binance executor with BNB trading support and BNB fee payment optimization
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from binance.error import ClientError
    from binance.um_futures import UMFutures

    FUTURES_AVAILABLE = True
except ImportError:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException

    FUTURES_AVAILABLE = False

# Import existing components
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import API_CONFIG, TRADING_CONFIG
from trading.execution.binance_executor import BinanceExecutor


class EnhancedBinanceExecutor(BinanceExecutor):
    """
    Enhanced Binance executor with BNB trading and fee optimization
    """

    def __init__(
        self,
        logger: logging.Logger = None,
        enable_bnb_fees: bool = True,
        bnb_trading_enabled: bool = True,
        min_bnb_balance: float = 0.1,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)

        # BNB-specific configuration
        self.enable_bnb_fees = enable_bnb_fees
        self.bnb_trading_enabled = bnb_trading_enabled
        self.min_bnb_balance = min_bnb_balance
        self.bnb_symbols = ["BNBUSDT", "BNBBTC"]

        # Enhanced fee rates with BNB discounts
        self.fee_rates = {
            "futures": {
                "maker": {"standard": 0.0002, "bnb_discount": 0.10},
                "taker": {"standard": 0.0004, "bnb_discount": 0.10},
            },
            "spot": {
                "maker": {"standard": 0.001, "bnb_discount": 0.25},
                "taker": {"standard": 0.001, "bnb_discount": 0.25},
            },
        }

        # Cache for BNB price and balance
        self._bnb_price_cache = None
        self._bnb_price_cache_time = 0
        self._bnb_balance_cache = None
        self._bnb_balance_cache_time = 0
        self.cache_ttl = 60  # 1 minute cache

    def initialize(self) -> bool:
        """Initialize with BNB fee settings"""
        success = super().initialize()

        if not success:
            return False

        if self.client and self.enable_bnb_fees:
            try:
                self._setup_bnb_fee_payment()
            except Exception as e:
                self.logger.warning(f"Could not setup BNB fee payment: {e}")

        return True

    def _setup_bnb_fee_payment(self):
        """Enable BNB fee payment if possible"""
        try:
            if hasattr(self.client, "toggle_bnb_burn"):
                # Enable BNB burn for trading fees
                self.client.toggle_bnb_burn(spotBNBBurn=True, interestBNBBurn=True)
                self.logger.info("✅ Enabled BNB fee payment for trading")
            else:
                self.logger.info(
                    "ℹ️ BNB fee payment setting not available for this client type"
                )
        except Exception as e:
            self.logger.warning(f"Could not enable BNB fee payment: {e}")

    def get_bnb_price(self, force_refresh: bool = False) -> float:
        """Get current BNB price with caching"""
        current_time = time.time()

        if (
            not force_refresh
            and self._bnb_price_cache
            and current_time - self._bnb_price_cache_time < self.cache_ttl
        ):
            return self._bnb_price_cache

        try:
            if self.client:
                if FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                    ticker = self.client.ticker_price(symbol="BNBUSDT")
                    price = float(ticker["price"])
                else:
                    ticker = self.client.get_symbol_ticker(symbol="BNBUSDT")
                    price = float(ticker["price"])

                self._bnb_price_cache = price
                self._bnb_price_cache_time = current_time
                return price
            else:
                return 300.0  # Fallback price
        except Exception as e:
            self.logger.error(f"Error getting BNB price: {e}")
            return self._bnb_price_cache or 300.0

    def get_bnb_balance(self, force_refresh: bool = False) -> float:
        """Get current BNB balance with caching"""
        current_time = time.time()

        if (
            not force_refresh
            and self._bnb_balance_cache
            and current_time - self._bnb_balance_cache_time < self.cache_ttl
        ):
            return self._bnb_balance_cache

        try:
            balances = self.get_balance()
            bnb_balance = balances.get("BNB", 0.0)

            self._bnb_balance_cache = bnb_balance
            self._bnb_balance_cache_time = current_time
            return bnb_balance
        except Exception as e:
            self.logger.error(f"Error getting BNB balance: {e}")
            return 0.0

    def calculate_fee_with_bnb(
        self, trade_value: float, is_maker: bool = False, is_futures: bool = True
    ) -> Dict[str, float]:
        """Calculate trading fees with and without BNB discount"""
        market_type = "futures" if is_futures else "spot"
        order_type = "maker" if is_maker else "taker"

        fee_config = self.fee_rates[market_type][order_type]
        standard_rate = fee_config["standard"]
        bnb_discount = fee_config["bnb_discount"]

        standard_fee = trade_value * standard_rate
        discounted_rate = standard_rate * (1 - bnb_discount)
        discounted_fee = trade_value * discounted_rate

        bnb_price = self.get_bnb_price()
        fee_in_bnb = discounted_fee / bnb_price if bnb_price > 0 else 0

        return {
            "standard_fee_usdt": standard_fee,
            "discounted_fee_usdt": discounted_fee,
            "fee_in_bnb": fee_in_bnb,
            "savings_usdt": standard_fee - discounted_fee,
            "discount_percent": bnb_discount * 100,
            "bnb_price": bnb_price,
        }

    def should_auto_buy_bnb(self, upcoming_trade_value: float = 0) -> Dict[str, Any]:
        """Determine if we should automatically buy BNB for fee discounts"""
        bnb_balance = self.get_bnb_balance()
        bnb_price = self.get_bnb_price()

        # Calculate estimated BNB needed for upcoming trades
        estimated_fee = upcoming_trade_value * 0.0004  # Taker fee
        bnb_needed_for_fee = estimated_fee / bnb_price if bnb_price > 0 else 0

        # Add buffer for multiple trades
        total_bnb_needed = max(self.min_bnb_balance, bnb_needed_for_fee * 10)

        should_buy = bnb_balance < total_bnb_needed
        buy_amount = total_bnb_needed - bnb_balance if should_buy else 0

        return {
            "should_buy": should_buy,
            "current_balance": bnb_balance,
            "needed_balance": total_bnb_needed,
            "buy_amount_bnb": buy_amount,
            "buy_cost_usdt": buy_amount * bnb_price,
            "bnb_price": bnb_price,
        }

    def buy_bnb_for_fees(self, amount_bnb: Optional[float] = None) -> Dict[str, Any]:
        """Buy BNB for fee payments"""
        if not self.bnb_trading_enabled:
            return {"success": False, "error": "BNB trading disabled"}

        try:
            bnb_analysis = self.should_auto_buy_bnb()
            buy_amount = amount_bnb or bnb_analysis["buy_amount_bnb"]

            if buy_amount <= 0:
                return {"success": False, "error": "No BNB purchase needed"}

            # Execute BNB purchase
            symbol = "BNBUSDT"
            result = self.execute_trade(
                symbol=symbol, side="BUY", quantity=buy_amount, trade_type="MARKET"
            )

            if result.get("success"):
                self.logger.info(f"✅ Purchased {buy_amount:.6f} BNB for fee payments")
                # Clear cache to get fresh balance
                self._bnb_balance_cache = None
                return {
                    "success": True,
                    "amount_bnb": buy_amount,
                    "cost_usdt": buy_amount * bnb_analysis["bnb_price"],
                    "order_result": result,
                }
            else:
                return {
                    "success": False,
                    "error": f"BNB purchase failed: {result.get('error')}",
                }

        except Exception as e:
            self.logger.error(f"Error buying BNB: {e}")
            return {"success": False, "error": str(e)}

    def execute_trade_with_bnb_optimization(
        self,
        symbol: str,
        side: str,
        quantity: float,
        trade_type: str = "MARKET",
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute trade with automatic BNB fee optimization"""

        # Calculate trade value for fee estimation
        current_price = self.get_current_price(symbol)
        trade_value = current_price * quantity

        # Check if we should buy BNB first
        if self.enable_bnb_fees and self.bnb_trading_enabled:
            bnb_check = self.should_auto_buy_bnb(trade_value)

            if (
                bnb_check["should_buy"]
                and bnb_check["buy_cost_usdt"] < trade_value * 0.1
            ):  # Max 10% of trade value
                self.logger.info(f"🔄 Auto-buying BNB for fee optimization...")
                bnb_result = self.buy_bnb_for_fees()

                if bnb_result["success"]:
                    self.logger.info(
                        f"✅ BNB purchased: {bnb_result['amount_bnb']:.6f} BNB"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ BNB auto-purchase failed: {bnb_result['error']}"
                    )

        # Calculate expected fees
        fee_analysis = self.calculate_fee_with_bnb(
            trade_value, is_futures=self.use_futures
        )

        # Execute the main trade
        result = self.execute_trade(symbol, side, quantity, trade_type, **kwargs)

        # Add fee analysis to result
        if result.get("success"):
            result["fee_analysis"] = fee_analysis
            result["expected_savings_usdt"] = fee_analysis["savings_usdt"]

            self.logger.info(f"💰 Trade executed with BNB optimization:")
            self.logger.info(
                f"   Expected fee savings: ${fee_analysis['savings_usdt']:.6f}"
            )
            self.logger.info(
                f"   BNB discount: {fee_analysis['discount_percent']:.1f}%"
            )

        return result

    def get_trading_pairs_for_asset(self, asset: str) -> List[str]:
        """Get available trading pairs for an asset"""
        if asset == "BNB":
            return ["BNBUSDT", "BNBBTC"]
        elif asset == "BTC":
            return ["BTCUSDT", "BTCBNB"]
        else:
            return [f"{asset}USDT", f"{asset}BTC", f"{asset}BNB"]

    def get_enhanced_balance(self) -> Dict[str, Any]:
        """Get balance with BNB-related information"""
        balance = self.get_balance()
        bnb_balance = balance.get("BNB", 0.0)
        bnb_price = self.get_bnb_price()

        return {
            "balances": balance,
            "bnb_info": {
                "balance": bnb_balance,
                "price_usdt": bnb_price,
                "value_usdt": bnb_balance * bnb_price,
                "sufficient_for_fees": bnb_balance >= self.min_bnb_balance,
                "min_required": self.min_bnb_balance,
            },
            "fee_optimization": {
                "enabled": self.enable_bnb_fees,
                "auto_buy_enabled": self.bnb_trading_enabled,
                "futures_discount": self.fee_rates["futures"]["taker"]["bnb_discount"]
                * 100,
                "spot_discount": self.fee_rates["spot"]["taker"]["bnb_discount"] * 100,
            },
        }


def test_enhanced_executor():
    """Test the enhanced Binance executor"""
    import os

    os.environ["VAULT_ENABLED"] = "false"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("🧪 Testing Enhanced Binance Executor with BNB Support")

    # Initialize executor
    executor = EnhancedBinanceExecutor(
        logger=logger,
        is_testnet=True,
        use_futures=True,
        enable_bnb_fees=True,
        bnb_trading_enabled=True,
    )

    executor.initialize()

    # Test balance with BNB info
    balance_info = executor.get_enhanced_balance()
    logger.info("📊 Enhanced Balance Info:")
    logger.info(f"BNB Balance: {balance_info['bnb_info']['balance']:.6f} BNB")
    logger.info(f"BNB Value: ${balance_info['bnb_info']['value_usdt']:.2f}")
    logger.info(
        f"Fee optimization enabled: {balance_info['fee_optimization']['enabled']}"
    )

    # Test fee calculation
    trade_value = 1000  # $1000 trade
    fee_analysis = executor.calculate_fee_with_bnb(trade_value, is_futures=True)
    logger.info(f"\n💰 Fee Analysis for ${trade_value} trade:")
    logger.info(f"Standard fee: ${fee_analysis['standard_fee_usdt']:.6f}")
    logger.info(f"With BNB discount: ${fee_analysis['discounted_fee_usdt']:.6f}")
    logger.info(
        f"Savings: ${fee_analysis['savings_usdt']:.6f} ({fee_analysis['discount_percent']:.1f}%)"
    )

    # Test BNB auto-buy analysis
    bnb_analysis = executor.should_auto_buy_bnb(trade_value)
    logger.info(f"\n🔄 BNB Auto-buy Analysis:")
    logger.info(f"Should buy: {bnb_analysis['should_buy']}")
    logger.info(f"Current balance: {bnb_analysis['current_balance']:.6f} BNB")
    logger.info(f"Needed balance: {bnb_analysis['needed_balance']:.6f} BNB")

    if bnb_analysis["should_buy"]:
        logger.info(
            f"Buy amount: {bnb_analysis['buy_amount_bnb']:.6f} BNB (${bnb_analysis['buy_cost_usdt']:.2f})"
        )


if __name__ == "__main__":
    test_enhanced_executor()
