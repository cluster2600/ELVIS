
def _execute_symbol_trade(symbol, signal, confidence, data, executor):
    """Helper method to execute trades for a specific symbol"""
    try:
        current_price = data.iloc[-1]['close']
        available_balance = executor.get_account_balance()
        
        # Calculate position size based on symbol
        if symbol == "BTCUSDT":
            # Use existing logic for BTCUSDT
            position_size = min(0.001, available_balance / current_price * 0.05)
        elif symbol == "BNBBTC":
            # For BNBBTC, use smaller position sizes
            position_size = min(0.01, available_balance / current_price * 0.02)
        else:
            position_size = 0.001  # Default
        
        if position_size <= 0:
            position_size = 0.001
        
        logger.info(f"🎯 {symbol} {signal}: Price=${current_price:.2f}, Size={position_size:.6f}")
        
        if signal == 'BUY':
            order_result = executor.place_order(symbol, 'buy', position_size, current_price)
        else:
            order_result = executor.place_order(symbol, 'sell', position_size, current_price)
        
        if order_result:
            logger.info(f"✅ {symbol} {signal} executed successfully")
        else:
            logger.error(f"❌ {symbol} {signal} execution failed")
            
    except Exception as e:
        logger.error(f"❌ Error executing {symbol} trade: {e}")
