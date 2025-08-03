#!/usr/bin/env python3
"""
Trade Data Analysis Script
Analyzes available trade data for training and provides insights.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from training.data.trade_history_processor import TradeHistoryProcessor
from utils.paper_trade_db import get_trade_count, get_all_trades

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🔍 ELVIS Trade Data Analysis")
    print("=" * 50)
    
    # Check database connection and trade count
    try:
        total_trades = get_trade_count(exclude_test=True)
        all_trades = get_trade_count(exclude_test=False)
        
        print(f"📊 Database Trade Summary:")
        print(f"   Total trades (all): {all_trades}")
        print(f"   Training trades (excluding test): {total_trades}")
        print(f"   Test trades: {all_trades - total_trades}")
        
        if total_trades == 0:
            print("⚠️  No trade data available for training!")
            print("💡 Run Elvis in paper trading mode to generate trade history")
            return
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        print("💡 Make sure PostgreSQL is running and vault is configured")
        return
    
    print("\n🔬 Processing Trade Data for Training...")
    
    # Initialize trade processor
    processor = TradeHistoryProcessor(logger)
    
    try:
        # Process trade data
        features_df, targets_df = processor.process_for_training(limit=1000)
        
        if features_df.empty:
            print("❌ No processable trade data found")
            return
            
        print(f"\n✅ Trade Data Processing Results:")
        print(f"   📈 Samples processed: {len(features_df)}")
        print(f"   🔧 Features extracted: {len(features_df.columns)}")
        print(f"   🎯 Targets created: {len(targets_df.columns)}")
        
        print(f"\n📋 Feature Categories:")
        
        # Categorize features
        market_features = [col for col in features_df.columns if any(x in col.lower() for x in ['price', 'volume', 'momentum', 'volatility', 'sma'])]
        trading_features = [col for col in features_df.columns if any(x in col.lower() for x in ['pnl', 'fee', 'profitable', 'win', 'loss', 'streak', 'position'])]
        time_features = [col for col in features_df.columns if any(x in col.lower() for x in ['hour', 'day', 'minute'])]
        
        print(f"   🏪 Market features ({len(market_features)}): {', '.join(market_features[:5])}{'...' if len(market_features) > 5 else ''}")
        print(f"   💰 Trading features ({len(trading_features)}): {', '.join(trading_features[:5])}{'...' if len(trading_features) > 5 else ''}")
        print(f"   ⏰ Time features ({len(time_features)}): {', '.join(time_features)}")
        
        print(f"\n🎯 Target Variables:")
        for target in targets_df.columns:
            print(f"   - {target}")
        
        # Get feature importance data
        feature_info = processor.get_feature_importance_data()
        
        if feature_info:
            print(f"\n📊 Data Quality Metrics:")
            print(f"   📏 Sample count: {feature_info.get('sample_count', 'N/A')}")
            print(f"   🔢 Numeric features: {feature_info.get('numeric_feature_count', 'N/A')}")
            
            # Show some basic stats
            means = feature_info.get('feature_means', {})
            if means:
                print(f"   💹 Avg PnL: {means.get('pnl', 'N/A'):.4f}" if 'pnl' in means else "")
                print(f"   💸 Avg Fee: {means.get('fee', 'N/A'):.4f}" if 'fee' in means else "")
                print(f"   🎯 Win Rate: {means.get('is_profitable', 'N/A'):.2%}" if 'is_profitable' in means else "")
        
        print(f"\n💾 Processed data saved to: data/processed/trade_history/")
        print(f"✨ Ready for training! Use: python train_with_trades.py")
        
    except Exception as e:
        logger.error(f"Error processing trade data: {e}")
        print(f"❌ Trade processing failed: {e}")

if __name__ == "__main__":
    main()