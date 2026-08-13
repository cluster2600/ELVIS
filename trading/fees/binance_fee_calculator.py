#!/usr/bin/env python3.14
"""
Comprehensive Binance fee calculator for futures trading with leverage
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple


class BinanceFeeCalculator:
    """
    Calculate all Binance fees for futures trading including:
    - Trading fees (maker/taker)
    - Funding rates
    - Leverage borrowing costs
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # Binance Futures Fee Structure (VIP 0 - default user)
        self.futures_maker_fee = 0.0002  # 0.02%
        self.futures_taker_fee = 0.0004  # 0.04%

        # Spot trading fees (for comparison)
        self.spot_maker_fee = 0.001  # 0.1%
        self.spot_taker_fee = 0.001  # 0.1%

        # Funding rate (typical range, actual is variable)
        self.typical_funding_rate = 0.0001  # 0.01% every 8 hours
        self.funding_interval_hours = 8

        # Borrowing interest for margin/leverage (annual rate)
        self.borrowing_interest_annual = 0.0365  # 3.65% annual (typical)

    def calculate_trading_fee(
        self,
        price: float,
        quantity: float,
        is_maker: bool = False,
        is_futures: bool = True,
    ) -> float:
        """
        Calculate trading fee for a single trade

        Args:
            price: Trade price
            quantity: Trade quantity
            is_maker: Whether this is a maker order (better fee)
            is_futures: Whether this is futures trading

        Returns:
            Fee amount in USDT
        """
        trade_value = price * quantity

        if is_futures:
            fee_rate = self.futures_maker_fee if is_maker else self.futures_taker_fee
        else:
            fee_rate = self.spot_maker_fee if is_maker else self.spot_taker_fee

        fee = trade_value * fee_rate

        self.logger.debug(
            f"Trading fee: ${fee:.6f} ({fee_rate*100:.3f}% of ${trade_value:.2f})"
        )
        return fee

    def calculate_funding_fee(
        self, position_value: float, hours_held: float, funding_rate: float = None
    ) -> float:
        """
        Calculate funding fee for holding a futures position

        Args:
            position_value: Notional value of position (with leverage)
            hours_held: Hours the position was held
            funding_rate: Current funding rate (if known), otherwise use typical rate

        Returns:
            Total funding fee paid/received
        """
        if funding_rate is None:
            funding_rate = self.typical_funding_rate

        # Funding is paid every 8 hours
        funding_periods = hours_held / self.funding_interval_hours
        total_funding_fee = position_value * funding_rate * funding_periods

        self.logger.debug(
            f"Funding fee: ${total_funding_fee:.6f} ({funding_periods:.2f} periods x {funding_rate*100:.4f}%)"
        )
        return total_funding_fee

    def calculate_borrowing_cost(
        self, borrowed_amount: float, hours_held: float
    ) -> float:
        """
        Calculate borrowing cost for leveraged position

        Args:
            borrowed_amount: Amount borrowed (position_value - margin)
            hours_held: Hours the position was held

        Returns:
            Borrowing interest cost
        """
        # Convert annual rate to hourly
        hourly_rate = self.borrowing_interest_annual / (365 * 24)
        borrowing_cost = borrowed_amount * hourly_rate * hours_held

        self.logger.debug(
            f"Borrowing cost: ${borrowing_cost:.6f} (${borrowed_amount:.2f} for {hours_held:.1f}h)"
        )
        return borrowing_cost

    def calculate_total_position_cost(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        leverage: int,
        hours_held: float,
        funding_rate: float = None,
    ) -> Dict[str, float]:
        """
        Calculate total cost for a complete leveraged position

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            leverage: Leverage used
            hours_held: Hours position was held
            funding_rate: Current funding rate

        Returns:
            Dictionary with all fee components
        """
        position_value = quantity * entry_price
        margin_used = position_value / leverage
        borrowed_amount = position_value - margin_used

        # Calculate all fee components
        entry_fee = self.calculate_trading_fee(
            entry_price, quantity, is_maker=False, is_futures=True
        )
        exit_fee = self.calculate_trading_fee(
            exit_price, quantity, is_maker=False, is_futures=True
        )
        funding_fee = self.calculate_funding_fee(
            position_value, hours_held, funding_rate
        )
        borrowing_cost = self.calculate_borrowing_cost(borrowed_amount, hours_held)

        # Gross P&L before fees
        gross_pnl = (exit_price - entry_price) * quantity

        # Total fees
        total_fees = entry_fee + exit_fee + funding_fee + borrowing_cost

        # Net P&L after all fees
        net_pnl = gross_pnl - total_fees

        return {
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "total_fees": total_fees,
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "funding_fee": funding_fee,
            "borrowing_cost": borrowing_cost,
            "position_value": position_value,
            "margin_used": margin_used,
            "borrowed_amount": borrowed_amount,
            "leverage": leverage,
        }

    def get_fee_summary(self, calculation: Dict[str, float]) -> str:
        """Get human-readable fee summary"""
        return f"""
Position Summary:
  Position Value: ${calculation['position_value']:,.2f}
  Margin Used: ${calculation['margin_used']:,.2f}
  Leverage: {calculation['leverage']}x
  
P&L:
  Gross P&L: ${calculation['gross_pnl']:+.2f}
  Total Fees: ${calculation['total_fees']:.6f}
  Net P&L: ${calculation['net_pnl']:+.2f}
  
Fee Breakdown:
  Entry Fee: ${calculation['entry_fee']:.6f}
  Exit Fee: ${calculation['exit_fee']:.6f}
  Funding Fee: ${calculation['funding_fee']:.6f}
  Borrowing Cost: ${calculation['borrowing_cost']:.6f}
        """.strip()


# Example usage for current BTC position
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    fee_calc = BinanceFeeCalculator()

    # Example: 1 BTC position with 10x leverage, held for 24 hours
    calculation = fee_calc.calculate_total_position_cost(
        entry_price=107500.0,
        exit_price=107600.0,  # $100 profit
        quantity=1.0,
        leverage=10,
        hours_held=24.0,
    )

    print("=== BINANCE FUTURES TRADING COST ANALYSIS ===")
    print(fee_calc.get_fee_summary(calculation))

    print(f"\n📊 IMPACT ON P&L:")
    print(f"Without fees: ${calculation['gross_pnl']:+.2f}")
    print(f"With all fees: ${calculation['net_pnl']:+.2f}")
    print(
        f"Fee impact: ${calculation['gross_pnl'] - calculation['net_pnl']:.6f} ({(calculation['total_fees']/abs(calculation['gross_pnl'])*100):.2f}% of gross P&L)"
    )
