class BNBToBTCStrategy:
    """Simple strategy to convert BNB to BTC when conditions are favorable"""

    def __init__(self, logger):
        self.logger = logger
        self.min_bnb_balance = 0.1
        self.conversion_threshold = 0.02  # Convert when BNB allocation > 2%

    def should_convert_bnb_to_btc(self, balance_info, market_data):
        """Determine if we should convert BNB to BTC"""
        bnb_balance = balance_info.get("BNB", 0)
        total_value = balance_info.get("total_usdt", 1000)

        if bnb_balance < self.min_bnb_balance:
            return False, "Insufficient BNB balance"

        # Calculate BNB allocation percentage
        bnb_value = bnb_balance * market_data.get("bnb_price_usdt", 300)
        bnb_allocation = bnb_value / total_value

        if bnb_allocation > self.conversion_threshold:
            return (
                True,
                f"BNB allocation {bnb_allocation:.1%} > {self.conversion_threshold:.1%}",
            )

        return False, f"BNB allocation {bnb_allocation:.1%} below threshold"

    def calculate_conversion_amount(self, bnb_balance, target_allocation=0.01):
        """Calculate how much BNB to convert to BTC"""
        # Keep 1% allocation in BNB, convert the rest
        excess_bnb = bnb_balance * (1 - target_allocation)
        return max(0, excess_bnb)
