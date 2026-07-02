
# BNB → BTC Trading Integration Guide

## Problem Identified
The bot was configured for BNBBTC futures trading, but BNBBTC only exists on Binance Spot market.

## Solution
1. **Use Spot Trading for BNBBTC**: Switch to spot executor when trading BNBBTC
2. **Separate Strategy**: Create dedicated BNB → BTC conversion logic
3. **Multi-Executor Setup**: Use futures for BTCUSDT, spot for BNBBTC

## Implementation Steps

### 1. Update main.py
Add BNBBTC to trading symbols and create dual executor setup:

```python
# In main.py trading loop
symbols_to_trade = ['BTCUSDT', 'BNBBTC']

for symbol in symbols_to_trade:
    if symbol == 'BNBBTC':
        # Use spot executor for BNBBTC
        if not hasattr(container, 'spot_executor'):
            from trading.execution.binance_executor import BinanceExecutor
            spot_executor = BinanceExecutor(
                logger=logger,
                is_testnet=True,
                use_futures=False  # Spot only
            )
            spot_executor.initialize()
            container.register_singleton('spot_executor', lambda: spot_executor)
        
        executor = container.get('spot_executor')
    else:
        # Use futures executor for BTCUSDT
        executor = container.get('executor')
    
    # Process symbol with appropriate executor
    data = price_fetcher.get_historical_klines(symbol, "1m")
    # ... rest of trading logic
```

### 2. Update bootstrap.py
Add BNBBTC to symbols list:

```python
# Change this line:
symbols=['BTCUSDT'],

# To this:
symbols=['BTCUSDT', 'BNBBTC'],
```

### 3. Create BNB → BTC Conversion Logic
Add this to your strategy:

```python
def check_bnb_conversion(self, balance, market_data):
    bnb_balance = balance.get('BNB', 0)
    
    # Convert excess BNB to BTC when BNB allocation > 2%
    if bnb_balance > 0.1:  # Minimum balance
        bnb_value = bnb_balance * market_data['bnb_price']
        total_value = sum(balance.values() * prices)
        
        if bnb_value / total_value > 0.02:  # 2% threshold
            # Convert excess BNB to BTC
            conversion_amount = bnb_balance * 0.5  # Convert 50%
            return 'CONVERT', conversion_amount
    
    return 'HOLD', 0
```

### 4. Test Configuration
1. Start bot with dual executors
2. Monitor BNBBTC price fetching
3. Test BNB → BTC conversion in paper mode
4. Verify BTC balance increases when BNB is sold

## Expected Behavior
- Bot monitors both BTCUSDT (futures) and BNBBTC (spot)
- When BNB allocation exceeds threshold, bot sells BNB for BTC
- BTC accumulates in spot balance
- Fee optimization continues with remaining BNB

## Key Points
- BNBBTC trades on SPOT market only
- Use separate executor for spot vs futures
- Monitor both balances (BNB and BTC)
- Set reasonable conversion thresholds
