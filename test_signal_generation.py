#!/usr/bin/env python3
"""
Test signal generation to ensure no NaN values
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import pandas as pd
import numpy as np
from trading.strategies.ensemble_strategy import EnsembleStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_signal_generation():
    """Test signal generation with real market data"""
    logger.info("=== Testing Signal Generation ===")
    
    # Initialize strategy
    strategy = EnsembleStrategy(logger=logger)
    
    # Create realistic market data with indicators
    data = pd.DataFrame({
        'open': [107000, 107050, 107100, 107150, 107200] * 10,
        'high': [107100, 107150, 107200, 107250, 107300] * 10,
        'low': [106900, 106950, 107000, 107050, 107100] * 10,
        'close': [107050, 107100, 107150, 107200, 107250] * 10,
        'volume': [1000, 1100, 1200, 1300, 1400] * 10
    })
    
    # Add technical indicators
    data['sma_20'] = data['close'].rolling(20).mean()
    data['rsi'] = 45.0  # Slightly oversold
    data['macd'] = -10.0  # Bearish
    data['signal_line'] = -5.0  # MACD below signal
    data['atr'] = 50.0
    
    logger.info(f"Created test data with shape: {data.shape}")
    logger.info(f"Data columns: {list(data.columns)}")
    logger.info(f"Latest close: {data.iloc[-1]['close']}")
    
    # Generate signals
    signals = strategy.generate_signals({"BTCUSDT": data})
    
    logger.info(f"Generated signals: {signals}")
    
    for symbol, signal_info in signals.items():
        signal = signal_info.get('signal', 'HOLD')
        confidence = signal_info.get('confidence', 0.0)
        
        logger.info(f"Symbol: {symbol}")
        logger.info(f"  Signal: {signal}")
        logger.info(f"  Confidence: {confidence}")
        logger.info(f"  Confidence type: {type(confidence)}")
        logger.info(f"  Is NaN: {pd.isna(confidence) if hasattr(pd, 'isna') else np.isnan(confidence)}")
        
        # Check if confidence is valid
        if pd.isna(confidence) or np.isnan(confidence):
            logger.error("❌ FAILED: Confidence is NaN!")
            return False
        elif confidence < 0 or confidence > 1:
            logger.error(f"❌ FAILED: Confidence {confidence} is out of range [0,1]!")
            return False
        else:
            logger.info("✅ SUCCESS: Valid confidence value")
    
    return True

def test_technical_analysis_fallback():
    """Test technical analysis fallback directly"""
    logger.info("=== Testing Technical Analysis Fallback ===")
    
    strategy = EnsembleStrategy(logger=logger)
    
    # Test with various indicator scenarios
    test_cases = [
        {
            'name': 'Normal indicators',
            'features': {
                'price': 107000.0,
                'rsi': 45.0,
                'macd': -10.0,
                'signal_line': -5.0,
                'sma': 106900.0,
                'volume': 1000.0
            }
        },
        {
            'name': 'NaN indicators',
            'features': {
                'price': 107000.0,
                'rsi': float('nan'),
                'macd': float('nan'),
                'signal_line': float('nan'),
                'sma': float('nan'),
                'volume': 1000.0
            }
        },
        {
            'name': 'Missing indicators',
            'features': {
                'price': 107000.0,
                'volume': 1000.0
            }
        },
        {
            'name': 'Oversold RSI',
            'features': {
                'price': 107000.0,
                'rsi': 25.0,  # Oversold
                'macd': 5.0,
                'signal_line': 2.0,
                'sma': 106500.0,  # Price above SMA
                'volume': 1000.0
            }
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"Testing: {test_case['name']}")
        
        try:
            prediction = strategy._technical_analysis_prediction(test_case['features'])
            
            logger.info(f"  Prediction shape: {prediction.shape}")
            logger.info(f"  Prediction values: {prediction}")
            logger.info(f"  Has NaN: {np.any(np.isnan(prediction))}")
            logger.info(f"  Sum: {np.sum(prediction)}")
            
            if len(prediction) != 3:
                logger.error(f"❌ FAILED: Wrong prediction shape {prediction.shape}, expected (3,)")
                return False
            elif np.any(np.isnan(prediction)):
                logger.error("❌ FAILED: Prediction contains NaN values!")
                return False
            elif np.any(prediction < 0) or np.any(prediction > 1):
                logger.error("❌ FAILED: Prediction values out of [0,1] range!")
                return False
            else:
                best_idx = np.argmax(prediction)
                decision = strategy.CLASSES[best_idx]
                confidence = float(prediction[best_idx])
                logger.info(f"  ✅ SUCCESS: {decision} with confidence {confidence:.3f}")
                
        except Exception as e:
            logger.error(f"❌ FAILED: Exception during prediction: {e}")
            return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting Signal Generation Tests...")
    
    # Test 1: Technical analysis fallback
    success1 = test_technical_analysis_fallback()
    print()
    
    # Test 2: Full signal generation
    success2 = test_signal_generation()
    
    if success1 and success2:
        logger.info("🎉 ALL TESTS PASSED! Signal generation is working correctly.")
    else:
        logger.error("❌ SOME TESTS FAILED! Check the logs above.")
    
    logger.info("Signal generation tests completed!")