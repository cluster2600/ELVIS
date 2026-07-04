import logging
from datetime import datetime


class AdvancedRiskManager:
    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.starting_balance = kwargs.get("starting_balance", 1000.0)
        self.margin_balance = self.starting_balance
        self.used_margin = 0.0
        self.open_positions = (
            []
        )  # List of dicts: [{'symbol': ..., 'entry_price': ..., 'quantity': ..., 'leverage': ...}]
        self.maintenance_margin_rate = 0.005  # 0.5%
        self.liquidation_fee_rate = 0.005  # 0.5%

    def open_position(
        self, symbol: str, entry_price: float, quantity: float, leverage: float
    ):
        notional_value = entry_price * quantity
        initial_margin = notional_value / leverage

        if initial_margin > self.margin_balance:
            self.logger.warning(
                f"Insufficient margin to open position: Required={initial_margin}, Available={self.margin_balance}"
            )
            return False

        self.margin_balance -= initial_margin
        self.used_margin += initial_margin
        self.open_positions.append(
            {
                "symbol": symbol,
                "entry_price": entry_price,
                "quantity": quantity,
                "leverage": leverage,
            }
        )
        self.logger.info(
            f"Opened position: {symbol} {quantity} @ {entry_price}x{leverage} | Used Margin: {self.used_margin:.2f} | Free Margin: {self.margin_balance:.2f}"
        )
        return True

    def check_liquidation(self, current_price: float):
        for position in self.open_positions[:]:
            notional = position["entry_price"] * position["quantity"]
            maintenance_margin = notional * self.maintenance_margin_rate

            unrealized_pnl = (
                (current_price - position["entry_price"])
                * position["quantity"]
                * (1 if position["leverage"] > 0 else -1)
            )
            free_margin = self.margin_balance + unrealized_pnl

            if free_margin <= maintenance_margin:
                self.logger.error(
                    f"Liquidation triggered for {position['symbol']}! Free Margin: {free_margin:.2f} <= Maintenance: {maintenance_margin:.2f}"
                )
                liquidation_fee = notional * self.liquidation_fee_rate
                self.margin_balance -= liquidation_fee
                self.used_margin -= notional / position["leverage"]
                self.open_positions.remove(position)
                self.logger.info(
                    f"Position liquidated. Liquidation Fee: {liquidation_fee:.2f} | New Balance: {self.margin_balance:.2f}"
                )

    def close_position(self, symbol: str, exit_price: float):
        for position in self.open_positions:
            if position["symbol"] == symbol:
                notional = position["entry_price"] * position["quantity"]
                pnl = (
                    (exit_price - position["entry_price"])
                    * position["quantity"]
                    * (1 if position["leverage"] > 0 else -1)
                )
                self.margin_balance += (notional / position["leverage"]) + pnl
                self.used_margin -= notional / position["leverage"]
                self.logger.info(
                    f"Closed position {symbol} @ {exit_price} | PnL: {pnl:.2f} | Balance: {self.margin_balance:.2f}"
                )
                self.open_positions.remove(position)
                return
        self.logger.warning(f"No open position found for {symbol} to close.")

    def get_status(self):
        return {
            "balance": self.margin_balance,
            "used_margin": self.used_margin,
            "open_positions": len(self.open_positions),
        }


if __name__ == "__main__":
    logger = logging.getLogger("RiskManager")
    logging.basicConfig(level=logging.INFO)

    arm = AdvancedRiskManager(logger, starting_balance=1000.0)
    arm.open_position("BTCUSDT", 95000, 0.01, leverage=10)
    arm.check_liquidation(90000)
    arm.close_position("BTCUSDT", 96000)
    print(arm.get_status())
