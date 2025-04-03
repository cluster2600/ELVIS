"""
Telegram bot for providing ML insights and analysis.
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from datetime import datetime, timedelta

from services.ml_engine.feature_analyzer import FeatureAnalyzer
from services.ml_engine.market_regime_detector import MarketRegimeDetector
from services.ml_engine.position_sizer import PositionSizingAgent
from services.ml_engine.ensemble_model import EnsembleModel
from config.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class TelegramBot:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        
        # Initialize ML components
        self.feature_analyzer = FeatureAnalyzer(config)
        self.regime_detector = MarketRegimeDetector(config)
        self.position_sizer = PositionSizingAgent(config)
        self.ensemble_model = EnsembleModel(config)
        
        # Initialize Telegram bot
        self.application = Application.builder().token(self.bot_token).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("analyze", self.analyze))
        self.application.add_handler(CommandHandler("regime", self.regime))
        self.application.add_handler(CommandHandler("position", self.position))
        self.application.add_handler(CommandHandler("ensemble", self.ensemble))
        self.application.add_handler(CallbackQueryHandler(self.button))
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        keyboard = [
            [
                InlineKeyboardButton("Feature Analysis", callback_data='analyze'),
                InlineKeyboardButton("Market Regime", callback_data='regime')
            ],
            [
                InlineKeyboardButton("Position Sizing", callback_data='position'),
                InlineKeyboardButton("Ensemble Predictions", callback_data='ensemble')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            'Welcome to ELVIS ML Insights Bot! Choose an option:',
            reply_markup=reply_markup
        )
        
    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send feature analysis results."""
        try:
            # Get recent market data
            data = self._get_recent_data()
            
            # Generate analysis report
            report = self.feature_analyzer.generate_report(
                data, 
                self.ensemble_model,
                self._get_targets(data)
            )
            
            # Format message
            message = "🔍 *Feature Analysis Report*\n\n"
            
            # Add permutation importance
            message += "*Top 5 Most Important Features:*\n"
            sorted_idx = np.argsort(report['permutation_importance']['importance_scores'])[-5:]
            for idx in reversed(sorted_idx):
                feature = report['permutation_importance']['feature_names'][idx]
                score = report['permutation_importance']['importance_scores'][idx]
                message += f"• {feature}: {score:.4f}\n"
            
            # Add SHAP analysis
            message += "\n*SHAP Analysis:*\n"
            sorted_idx = np.argsort(report['shap_analysis']['mean_shap_values'])[-5:]
            for idx in reversed(sorted_idx):
                feature = report['shap_analysis']['feature_names'][idx]
                value = report['shap_analysis']['mean_shap_values'][idx]
                message += f"• {feature}: {value:.4f}\n"
            
            # Add correlation insights
            message += "\n*Feature Correlations:*\n"
            corr_matrix = report['correlations']['correlation_matrix']
            high_corr = np.where(np.abs(corr_matrix) > 0.7)
            for i, j in zip(*high_corr):
                if i < j:  # Avoid duplicates
                    feat1 = report['correlations']['feature_names'][i]
                    feat2 = report['correlations']['feature_names'][j]
                    corr = corr_matrix[i, j]
                    message += f"• {feat1} ↔️ {feat2}: {corr:.2f}\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in analyze command: {str(e)}")
            await update.message.reply_text("Sorry, there was an error generating the analysis.")
            
    async def regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send current market regime analysis."""
        try:
            # Get recent market data
            data = self._get_recent_data()
            
            # Detect current regime
            regime, confidence = self.regime_detector.detect_current_regime(data)
            
            # Format message
            message = "📊 *Market Regime Analysis*\n\n"
            message += f"*Current Regime:* {regime}\n"
            message += f"*Confidence:* {confidence:.2%}\n\n"
            
            # Add regime description
            if "High Volatility" in regime:
                message += "⚠️ *High Volatility Warning*\n"
                message += "• Consider reducing position sizes\n"
                message += "• Tighten stop-losses\n"
                message += "• Monitor for potential trend reversals\n"
            else:
                message += "✅ *Low Volatility Environment*\n"
                message += "• Normal position sizing\n"
                message += "• Standard stop-losses\n"
                message += "• Focus on trend following\n"
                
            if "Uptrend" in regime:
                message += "\n📈 *Uptrend Detected*\n"
                message += "• Favor long positions\n"
                message += "• Use pullbacks for entries\n"
            else:
                message += "\n📉 *Downtrend Detected*\n"
                message += "• Favor short positions\n"
                message += "• Use rallies for entries\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in regime command: {str(e)}")
            await update.message.reply_text("Sorry, there was an error analyzing the market regime.")
            
    async def position(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send position sizing recommendations."""
        try:
            # Get recent market data
            data = self._get_recent_data()
            
            # Get position sizing recommendation
            position_size = self.position_sizer.get_position_size(data)
            
            # Format message
            message = "💰 *Position Sizing Recommendation*\n\n"
            message += f"*Recommended Position Size:* {position_size:.2%} of portfolio\n\n"
            
            # Add rationale
            message += "*Rationale:*\n"
            message += "• Based on current market conditions\n"
            message += "• Adjusted for portfolio risk\n"
            message += "• Optimized for expected returns\n"
            
            # Add risk warnings
            message += "\n⚠️ *Risk Warnings:*\n"
            if position_size > 0.1:
                message += "• High position size detected\n"
                message += "• Consider reducing exposure\n"
            elif position_size < 0.01:
                message += "• Very small position size\n"
                message += "• May not be worth the trade\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in position command: {str(e)}")
            await update.message.reply_text("Sorry, there was an error generating position sizing recommendations.")
            
    async def ensemble(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send ensemble model predictions."""
        try:
            # Get recent market data
            data = self._get_recent_data()
            
            # Get ensemble predictions
            predictions = self.ensemble_model.predict(data)
            
            # Format message
            message = "🤖 *Ensemble Model Predictions*\n\n"
            message += f"*Prediction:* {predictions['prediction']:.2%}\n"
            message += f"*Confidence:* {predictions['confidence']:.2%}\n\n"
            
            # Add base model predictions
            message += "*Base Model Predictions:*\n"
            for model_name, pred in predictions['base_predictions'].items():
                message += f"• {model_name}: {pred:.2%}\n"
            
            # Add interpretation
            message += "\n*Interpretation:*\n"
            if predictions['prediction'] > 0.6:
                message += "• Strong buy signal\n"
                message += "• High confidence in prediction\n"
            elif predictions['prediction'] < 0.4:
                message += "• Strong sell signal\n"
                message += "• High confidence in prediction\n"
            else:
                message += "• Neutral signal\n"
                message += "• Consider waiting for clearer signals\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in ensemble command: {str(e)}")
            await update.message.reply_text("Sorry, there was an error generating ensemble predictions.")
            
    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button presses."""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'analyze':
            await self.analyze(update, context)
        elif query.data == 'regime':
            await self.regime(update, context)
        elif query.data == 'position':
            await self.position(update, context)
        elif query.data == 'ensemble':
            await self.ensemble(update, context)
            
    def _get_recent_data(self) -> pd.DataFrame:
        """Get recent market data for analysis."""
        # This should be replaced with actual data fetching logic
        # For now, return mock data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.normal(50000, 1000, 100),
            'high': np.random.normal(51000, 1000, 100),
            'low': np.random.normal(49000, 1000, 100),
            'close': np.random.normal(50000, 1000, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)
        return data
        
    def _get_targets(self, data: pd.DataFrame) -> np.ndarray:
        """Get target values for analysis."""
        # This should be replaced with actual target calculation
        # For now, return mock targets
        return np.random.randint(0, 2, len(data))
        
    def run(self):
        """Run the bot."""
        self.application.run_polling() 