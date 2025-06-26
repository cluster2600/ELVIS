import numpy as np
import pandas as pd
import os
import requests
import coremltools as ct
import logging
from typing import Dict, Any, List
from datetime import datetime
from trading.strategies.base_strategy import BaseStrategy
import ta

# Handle YDF import with fallback
try:
    import ydf
    YDF_AVAILABLE = True
except ImportError:
    YDF_AVAILABLE = False
    ydf = None
    print("YDF not available, ensemble strategy will fallback to other models")

class EnsembleStrategy(BaseStrategy):
    """
    EnsembleStrategy combines predictions from multiple models:
    - YDF Random Forest model
    - CoreML Neural Network model
    - Trade-learned model (trained on actual trading history)
    - (optional) MLX Large Language Model for additional decision support
    
    This strategy averages model outputs to determine a consensus BUY, SELL, or HOLD signal.
    The trade-learned model provides adaptive learning from actual trading experience.
    """

    def __init__(self, logger: logging.Logger, 
                 symbols: List[str] = ['BTCUSDT'],
                 ydf_model_path: str = "models/model_rf.ydf",
                 coreml_model_path: str = "models/NNModel.mlpackage",
                 trade_learned_model_path: str = "training/models/trade_learned_model.pkl",
                 mlx_url: str = None,
                 risk_per_trade: float = 0.01,
                 min_position_size: float = 0.001,
                 max_position_size: float = 0.1,
                 order_flow_analyzer=None,
                 price_fetcher=None,
                 exchange_manager=None):
        """
        Initialize the ensemble strategy, loading models and setting parameters.

        Args:
            logger (logging.Logger): The logger for debugging/info output.
            symbols (List[str]): The trading pairs to manage.
            ydf_model_path (str): Path to the YDF Random Forest model.
            coreml_model_path (str): Path to the CoreML Neural Network model.
            mlx_url (str, optional): URL to MLX server for LLM support.
            risk_per_trade (float): The percentage of the portfolio to risk on a single trade.
            min_position_size (float): The minimum position size in BTC.
            max_position_size (float): The maximum position size in BTC.
            order_flow_analyzer: The order flow analyzer instance.
            price_fetcher: The price fetcher instance.
        """
        super().__init__(logger)
        self.logger = logger
        self.symbols = symbols
        self.order_flow_analyzer = order_flow_analyzer
        self.price_fetcher = price_fetcher
        self.exchange_manager = exchange_manager
        self.REQUIRED_FEATURES = [
            "price", "Order_Amount", "sma", "Filled", "Total", "future_price", "atr",
            "vol_adjusted_price", "volume_ma", "macd", "signal_line", "lower_bb",
            "sma_bb", "upper_bb", "news_sentiment", "social_feature", "adx", "rsi",
            "order_book_depth", "volume"
        ]  # Exactly 20 features to match CoreML model expectations
        self.CLASSES = ["BUY", "HOLD", "SELL"]

        self.risk_per_trade = risk_per_trade
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size

        self.mlx_url = mlx_url or os.getenv('MLX_URL', '')
        self.mlx_available = False

        # Load models
        # self.ydf_model = self._load_ydf_model(ydf_model_path)
        self.ydf_model = None
        self.nn_model = self._load_coreml_model(coreml_model_path)
        self.trade_learned_model = self._load_trade_learned_model(trade_learned_model_path)
        self._check_mlx_connectivity()
        
        # Initialize DRL agent
        self.drl_agent = self._initialize_drl_agent()

    # def _load_ydf_model(self, model_path: str):
    #     """Load the YDF model from disk."""
    #     try:
    #         if not os.path.exists(model_path):
    #             raise FileNotFoundError(f"YDF model file not found at {model_path}")
    #         model = ydf.from_tensorflow_decision_forests(model_path)
    #         self.logger.info(f"YDF model loaded from {model_path}")
    #         return model
    #     except Exception as e:
    #         self.logger.error(f"Failed to load YDF model: {e}")
    #         return None

    def _load_coreml_model(self, model_path: str):
        """Load the CoreML model from disk."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"CoreML model file not found at {model_path}")
            model = ct.models.MLModel(model_path)
            self.logger.info(f"CoreML model loaded from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load CoreML model: {e}")
            return None

    def _load_trade_learned_model(self, model_path: str):
        """Load the trade-learned model from disk."""
        try:
            import joblib
            if not os.path.exists(model_path):
                self.logger.warning(f"Trade-learned model file not found at {model_path}")
                return None
            
            model_data = joblib.load(model_path)
            self.logger.info(f"Trade-learned model loaded from {model_path}")
            self.logger.info(f"Model type: {model_data.get('model_type', 'unknown')}")
            self.logger.info(f"Training samples: {model_data.get('training_samples', 'unknown')}")
            self.logger.info(f"CV score: {model_data.get('cv_score', 'unknown')}")
            
            return model_data
        except Exception as e:
            self.logger.error(f"Failed to load trade-learned model: {e}")
            return None

    def _check_mlx_connectivity(self):
        """Check if MLX server is available."""
        if not self.mlx_url:
            return
        try:
            resp = requests.get(f"{self.mlx_url.split('/v1/')[0]}/v1/models", timeout=5)
            resp.raise_for_status()
            self.mlx_available = True
            self.logger.info("MLX server available.")
        except Exception as e:
            self.logger.warning(f"MLX server not available: {e}")

    def _initialize_drl_agent(self):
        """Initialize DRL agent if available."""
        try:
            from core.di import container
            try:
                agent = container.get('drl_agent')
                if agent is not None:
                    self.logger.info("DRL agent loaded from dependency injection container")
                    return agent
            except Exception:
                pass
            
            from drl_agents.elegantrl_models import DRLAgent
            from core.models.reinforcement_learning_model import TradingEnv

            drl_model_path = "models/checkpoints"
            if os.path.exists(drl_model_path):
                price_array = []  # Replace with actual data
                tech_array = []   # Replace with actual data
                env_params = {}   # Replace with actual params
                
                agent = DRLAgent(
                    env=TradingEnv,
                    price_array=price_array,
                    tech_array=tech_array,
                    env_params=env_params,
                    if_log=True
                )
                self.logger.info("DRL agent initialized successfully")
                return agent
            else:
                self.logger.info("No DRL model found, DRL predictions disabled")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize DRL agent: {e}")
            return None

    def _get_drl_prediction(self, features: dict) -> str:
        """Get prediction from DRL agent."""
        try:
            # Convert features to state vector expected by DRL agent
            state_vector = self._features_to_state_vector(features)
            
            # Get action from DRL agent
            action = self.drl_agent.get_action(state_vector, if_greedy=True)
            
            # Convert action to trading signal
            if action == 0:
                return "SELL"
            elif action == 1:
                return "HOLD" 
            elif action == 2:
                return "BUY"
            else:
                return "HOLD"
        except Exception as e:
            self.logger.warning(f"DRL prediction failed: {e}")
            return "HOLD"

    def _features_to_state_vector(self, features: dict) -> np.ndarray:
        """Convert feature dictionary to state vector for DRL agent."""
        try:
            # Extract key features for DRL agent state
            state_features = [
                features.get('price', 0.0),
                features.get('volume', 0.0),
                features.get('rsi', 50.0),
                features.get('macd', 0.0),
                features.get('sma', 0.0),
                features.get('atr', 0.0),
                features.get('high', 0.0),
                features.get('low', 0.0),
                features.get('close', 0.0)
            ]
            
            # Normalize features (simple normalization)
            state_vector = np.array(state_features, dtype=np.float32)
            # Basic normalization - could be improved with proper scaling
            state_vector = (state_vector - np.mean(state_vector)) / (np.std(state_vector) + 1e-8)
            
            return state_vector
        except Exception as e:
            self.logger.warning(f"Failed to convert features to state vector: {e}")
            # Return default state vector
            return np.zeros(9, dtype=np.float32)

    def _mlx_generate(self, prompt: str) -> str:
        """Generate a decision using MLX server."""
        if not self.mlx_available:
            return "HOLD"
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "llama-3.2-3b-instruct",
                "prompt": prompt,
                "max_tokens": 10
            }
            resp = requests.post(self.mlx_url, headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
            decision = resp.json()["choices"][0]["text"].strip().upper()
            return decision
        except Exception as e:
            self.logger.warning(f"MLX generation error: {e}")
            return "HOLD"

    def _parse_mlx_decision(self, text: str) -> str:
        """Parse the MLX model text output."""
        for word in text.split():
            if word in self.CLASSES:
                return word
        return "HOLD"

    def _get_model_predictions(self, features: dict) -> Dict[str, np.ndarray]:
        """Predict using available models, with technical analysis fallback."""
        preds = {}
        
        # Try YDF model if available
        if YDF_AVAILABLE and self.ydf_model is not None:
            try:
                import subprocess
                import json
                ydf_env_path = "/path/to/env-ydf/bin/python"
                ydf_script_path = "/Users/maxime/BTC_BOT/BTC_BOT/predict_with_ydf.py"
                result = subprocess.run(
                    [ydf_env_path, ydf_script_path],
                    input=json.dumps(features).encode(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                output = json.loads(result.stdout)
                if "probabilities" in output:
                    preds['ydf'] = np.array(output["probabilities"])
            except Exception as e:
                self.logger.warning(f"YDF prediction failed: {e}")
        
        # Try CoreML model if available
        if self.nn_model is not None:
            try:
                # Create safe feature array with defaults
                feature_values = []
                for col in self.REQUIRED_FEATURES:
                    value = features.get(col, 0.0)
                    if pd.isna(value) or not isinstance(value, (int, float)):
                        value = 0.0
                    feature_values.append(float(value))
                
                # Ensure we have exactly 20 features for the CoreML model
                if len(feature_values) != 20:
                    self.logger.warning(f"CoreML model expects 20 features, got {len(feature_values)}. Adjusting array size.")
                    if len(feature_values) < 20:
                        # Pad with zeros if we have fewer features
                        feature_values.extend([0.0] * (20 - len(feature_values)))
                    else:
                        # Truncate if we have more features
                        feature_values = feature_values[:20]
                
                # Reshape to exactly (1, 20) as expected by the model
                nn_input = {'features': np.array(feature_values, dtype=np.float32).reshape(1, 20)}
                nn_pred = self.nn_model.predict(nn_input)
                probs = nn_pred.get('classLabel_probs') or nn_pred.get('classProbability', {})
                preds['nn'] = np.array([probs.get(cls, 0.0) for cls in self.CLASSES])
                self.logger.debug(f"CoreML prediction successful with input shape {nn_input['features'].shape}")
            except Exception as e:
                self.logger.warning(f"CoreML prediction failed: {e}")
                self.logger.debug(f"Current REQUIRED_FEATURES count: {len(self.REQUIRED_FEATURES)}")
                self.logger.debug(f"Features list: {self.REQUIRED_FEATURES}")
        
        # Try trade-learned model if available
        if self.trade_learned_model is not None:
            try:
                model_data = self.trade_learned_model
                model = model_data['model']
                feature_names = model_data['feature_names']
                
                # Extract features for the trade-learned model
                trade_features = self._extract_trade_learned_features(features)
                
                # Create feature array in the correct order
                feature_array = np.array([[trade_features.get(name, 0.0) for name in feature_names]])
                
                # Get prediction probabilities
                probabilities = model.predict_proba(feature_array)[0]
                
                # Convert to [BUY, HOLD, SELL] format
                # Model was trained with 0=loss, 1=profit
                # Map: profit->BUY, loss->SELL, neutral->HOLD
                if len(probabilities) == 2:  # Binary classifier
                    # probabilities[0] = loss, probabilities[1] = profit
                    buy_prob = probabilities[1]   # Profit probability -> BUY
                    sell_prob = probabilities[0]  # Loss probability -> SELL
                    hold_prob = 1.0 - max(buy_prob, sell_prob)  # Neutral
                    preds['trade_learned'] = np.array([buy_prob, hold_prob, sell_prob])
                
                self.logger.debug(f"Trade-learned model prediction successful")
            except Exception as e:
                self.logger.warning(f"Trade-learned model prediction failed: {e}")
        
        # Try MLX if available
        if self.mlx_available:
            try:
                mlx_decision = self._parse_mlx_decision(
                    self._mlx_generate(
                        f"Predict market move for features: {features} -> BUY, SELL, or HOLD."
                    )
                )
                preds['mlx'] = np.array([1.0 if c == mlx_decision else 0.0 for c in self.CLASSES])
            except Exception as e:
                self.logger.warning(f"MLX prediction error: {e}")
        
        # Try DRL agent if available
        if hasattr(self, 'drl_agent') and self.drl_agent is not None:
            try:
                drl_decision = self._get_drl_prediction(features)
                preds['drl'] = np.array([1.0 if c == drl_decision else 0.0 for c in self.CLASSES])
                self.logger.debug(f"DRL agent prediction: {drl_decision}")
            except Exception as e:
                self.logger.warning(f"DRL agent prediction error: {e}")
        
        # Technical analysis fallback if no models available or all failed
        valid_preds = {k: v for k, v in preds.items() if v is not None and len(v) == 3 and not np.any(np.isnan(v))}
        if not valid_preds:
            self.logger.info("Using technical analysis fallback (no valid model predictions)")
            preds['technical'] = self._technical_analysis_prediction(features)
        else:
            self.logger.info(f"Using {len(valid_preds)} valid model predictions: {list(valid_preds.keys())}")
        
        return preds

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for all symbols based on ensemble voting.

        Args:
            data (Dict[str, pd.DataFrame]): A dictionary of market data for each symbol.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of signals for each symbol.
        """
        signals = {}
        for symbol in self.symbols:
            if symbol not in data or data[symbol].empty:
                signals[symbol] = {"signal": "HOLD", "confidence": 0.0}
                continue

            df = data[symbol]
            try:
                # Create features from the dataframe
                features = self._create_features_from_data(df)
                
                preds = self._get_model_predictions(features)
                
                if preds:
                    # Filter out invalid predictions (None, wrong size, NaN values)
                    pred_arrays = [p for p in preds.values() if p is not None and len(p) == 3 and not np.any(np.isnan(p))]
                    if pred_arrays:
                        pred_array = np.mean(pred_arrays, axis=0)
                        # Ensure no NaN values in final prediction
                        if not np.any(np.isnan(pred_array)):
                            best_idx = np.argmax(pred_array)
                            decision = self.CLASSES[best_idx]
                            confidence = float(pred_array[best_idx])
                        else:
                            self.logger.warning("Final prediction array contains NaN values, defaulting to HOLD")
                            decision, confidence = "HOLD", 0.0
                    else:
                        self.logger.warning("No valid prediction arrays found, defaulting to HOLD")
                        decision, confidence = "HOLD", 0.0
                else:
                    self.logger.warning("No predictions received, defaulting to HOLD")
                    decision, confidence = "HOLD", 0.0
                
                self.logger.info(f"Ensemble decision for {symbol}: {decision} ({confidence:.4f})")
                signals[symbol] = {"signal": decision, "confidence": confidence}
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                signals[symbol] = {"signal": "HOLD", "confidence": 0.0}

        return signals

    def check_arbitrage_opportunities(self, symbol: str = 'BTCUSDT') -> List[Dict[str, Any]]:
        """Check for arbitrage opportunities across exchanges"""
        if not self.exchange_manager:
            self.logger.warning("Exchange manager not available for arbitrage detection")
            return []
        
        try:
            opportunities = self.exchange_manager.detect_arbitrage_opportunities(symbol)
            
            if opportunities:
                self.logger.info(f"Found {len(opportunities)} arbitrage opportunities for {symbol}")
                for opp in opportunities:
                    self.logger.info(
                        f"Arbitrage: Buy {opp['buy_exchange']} @ {opp['buy_price']:.2f}, "
                        f"Sell {opp['sell_exchange']} @ {opp['sell_price']:.2f}, "
                        f"Profit: {opp['profit_pct']*100:.2f}%"
                    )
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error checking arbitrage opportunities: {e}")
            return []
    
    def execute_multi_exchange_order(self, symbol: str, side: str, quantity: float, 
                                   use_best_price: bool = True) -> Dict[str, Any]:
        """Execute order using the best available exchange"""
        if not self.exchange_manager:
            self.logger.warning("Exchange manager not available, using single exchange")
            return {}
        
        try:
            if use_best_price:
                result = self.exchange_manager.execute_smart_order(symbol, side, quantity)
                self.logger.info(
                    f"Smart order executed on {result.get('selected_exchange', 'unknown')} "
                    f"at price {result.get('selected_price', 0):.2f}"
                )
                return result
            else:
                # Use first available exchange
                available_exchanges = self.exchange_manager.get_available_exchanges()
                if available_exchanges:
                    exchange = self.exchange_manager.get_exchange(available_exchanges[0])
                    if side.lower() == 'buy':
                        return exchange.execute_buy(symbol, quantity)
                    else:
                        return exchange.execute_sell(symbol, quantity)
                
        except Exception as e:
            self.logger.error(f"Error executing multi-exchange order: {e}")
            return {'error': str(e)}
    
    def get_consolidated_portfolio(self) -> Dict[str, Any]:
        """Get consolidated portfolio across all exchanges"""
        if not self.exchange_manager:
            return {}
        
        try:
            consolidated_balance = self.exchange_manager.get_consolidated_balance()
            
            portfolio_summary = {
                'total_value_usd': 0.0,
                'balances': consolidated_balance,
                'exchange_count': len(self.exchange_manager.get_available_exchanges()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate approximate USD value for major cryptocurrencies
            major_cryptos = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
            
            for crypto in major_cryptos:
                if crypto in consolidated_balance:
                    try:
                        symbol = f"{crypto}USDT"
                        prices = self.exchange_manager.get_prices_all_exchanges(symbol)
                        if prices:
                            avg_price = sum(prices.values()) / len(prices)
                            crypto_value = consolidated_balance[crypto]['total_balance'] * avg_price
                            portfolio_summary['total_value_usd'] += crypto_value
                    except Exception:
                        pass
            
            # Add USD/USDT holdings directly
            for stablecoin in ['USD', 'USDT', 'USDC']:
                if stablecoin in consolidated_balance:
                    portfolio_summary['total_value_usd'] += consolidated_balance[stablecoin]['total_balance']
            
            return portfolio_summary
            
        except Exception as e:
            self.logger.error(f"Error getting consolidated portfolio: {e}")
            return {}
    
    def get_market_overview(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get market overview across all exchanges"""
        if not self.exchange_manager:
            return {}
        
        symbols = symbols or self.symbols
        
        try:
            market_summary = self.exchange_manager.get_market_summary(symbols)
            
            overview = {
                'markets': market_summary,
                'exchange_health': self.exchange_manager.check_all_exchanges_health(),
                'available_exchanges': self.exchange_manager.get_available_exchanges(),
                'timestamp': datetime.now().isoformat()
            }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error getting market overview: {e}")
            return {}

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate the Average True Range (ATR) as a measure of volatility.

        Args:
            data (pd.DataFrame): DataFrame with high, low, and close prices.
            period (int): The period over which to calculate the ATR.

        Returns:
            float: The latest ATR value.
        """
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            self.logger.warning("ATR calculation requires 'high', 'low', 'close' columns.")
            return 0.01 # Return a default small volatility
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        tr = np.max([high_low, high_close, low_close], axis=0)
        atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]
        return atr if pd.notna(atr) else 0.01

    def calculate_position_size(self, data: pd.DataFrame, current_price: float, available_capital: float, leverage: float = 10.0) -> float:
        """
        Calculate position size based on volatility (ATR), risk per trade, and leverage.
        
        Args:
            data (pd.DataFrame): The data to calculate position size from.
            current_price (float): The current price.
            available_capital (float): The available capital.
            leverage (float): The leverage to use for the position.
            
        Returns:
            float: The position size in BTC.
        """
        try:
            self.logger.info(f"Calculating position size - Price: {current_price}, Capital: {available_capital}, Leverage: {leverage}x")
            
            # Simplified position sizing for testing
            if available_capital <= 0 or current_price <= 0:
                self.logger.warning(f"Invalid inputs - Capital: {available_capital}, Price: {current_price}")
                return self.min_position_size
            
            # Correct leverage-adjusted position sizing
            risk_percentage = 0.02  # Risk 2% of capital per trade
            risk_amount = available_capital * risk_percentage  # Dollar amount we're willing to risk
            
            # With leverage, we can control a larger position while only risking the risk_amount
            # The position value is the actual market exposure (leveraged)
            position_value = risk_amount * leverage  # This is our market exposure
            position_size = position_value / current_price
            
            # Clamp to min/max limits (without leverage multiplication)
            position_size = max(position_size, self.min_position_size)
            position_size = min(position_size, self.max_position_size)
            
            # Ensure we don't exceed what we can afford with our margin
            # With leverage, we only need to put up position_value / leverage as margin
            required_margin = position_value / leverage
            max_affordable_margin = available_capital * 0.95  # Use 95% to leave buffer
            if required_margin > max_affordable_margin:
                # Scale down position to fit available margin
                max_position_value = max_affordable_margin * leverage
                position_size = max_position_value / current_price
            
            self.logger.info(f"Calculated position size: {position_size:.6f} BTC (${position_size * current_price:.2f}, {leverage}x leveraged)")
            
            return max(position_size, self.min_position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            # Return a safe fallback - 1% of capital as market exposure
            fallback_risk = available_capital * 0.01  # 1% of capital at risk
            fallback_position_value = fallback_risk * leverage  # Market exposure with leverage
            fallback_size = fallback_position_value / current_price if current_price > 0 else self.min_position_size
            self.logger.info(f"Using fallback position size: {fallback_size:.6f}")
            return max(fallback_size, self.min_position_size)

    def calculate_stop_loss(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the stop loss price based on ATR.
        
        Args:
            data (pd.DataFrame): The data to calculate stop loss from.
            entry_price (float): The entry price.
            
        Returns:
            float: The stop loss price.
        """
        atr = self._calculate_atr(data)
        return entry_price - (atr * 2) # Example: 2 * ATR below entry

    def calculate_take_profit(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the take profit price based on ATR.
        
        Args:
            data (pd.DataFrame): The data to calculate take profit from.
            entry_price (float): The entry price.
            
        Returns:
            float: The take profit price.
        """
        atr = self._calculate_atr(data)
        return entry_price + (atr * 3) # Example: 3 * ATR above entry

    def _calculate_trend_strength(self, data: pd.DataFrame, adx_period: int = 14, rsi_period: int = 14) -> float:
        """
        Calculate the trend strength using ADX and RSI.
        
        Returns:
            float: A value between 0 and 1 representing the trend strength.
        """
        adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'], window=adx_period).adx()
        rsi = ta.momentum.RSIIndicator(data['close'], window=rsi_period).rsi()
        
        # Normalize ADX and RSI to a 0-1 scale
        adx_strength = min(adx.iloc[-1] / 50, 1.0) # ADX > 50 is a strong trend
        rsi_strength = abs(rsi.iloc[-1] - 50) / 50 # RSI further from 50 is a stronger trend
        
        return (adx_strength + rsi_strength) / 2
    
    def _create_features_from_data(self, df: pd.DataFrame) -> dict:
        """Create features from price data."""
        try:
            latest = df.iloc[-1]
            
            # Calculate basic features
            features = {
                'price': latest.get('close', 0.0),
                'volume': latest.get('volume', 0.0),
                'sma': latest.get('sma_20', 0.0),
                'rsi': latest.get('rsi', 50.0),
                'macd': latest.get('macd', 0.0),
                'signal_line': latest.get('signal_line', 0.0),
                'adx': latest.get('adx', 0.0),
                'atr': latest.get('atr', 0.0),
                'lower_bb': latest.get('lower_bb', 0.0),
                'sma_bb': latest.get('sma_bb', 0.0),
                'upper_bb': latest.get('upper_bb', 0.0),
                # Additional required features with defaults
                'Order_Amount': 0.0,
                'Filled': 0.0,
                'Total': 0.0,
                'future_price': latest.get('close', 0.0),  # Use current price as estimate
                'vol_adjusted_price': latest.get('close', 0.0),
                'volume_ma': latest.get('volume', 0.0),
                'news_sentiment': 0.0,  # Neutral sentiment
                'social_feature': 0.0,
                'order_book_depth': 0.0
            }
            
            # Ensure all required features are present with defaults if missing
            for feature in self.REQUIRED_FEATURES:
                if feature not in features:
                    features[feature] = 0.0
            
            return features
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return {k: 0.0 for k in self.REQUIRED_FEATURES}
    
    def _extract_trade_learned_features(self, features: dict) -> Dict[str, float]:
        """
        Extract features for the trade-learned model from market data
        """
        try:
            # Get current time
            now = datetime.now()
            
            # Get current price and volume data
            current_price = features.get('price', 97000.0)
            current_volume = features.get('volume', 1000.0)
            
            # Calculate trade characteristics (for simulation)
            estimated_trade_size = 0.01  # Small default trade size
            estimated_trade_value = estimated_trade_size * current_price
            
            # Extract the features that the model was trained on
            trade_features = {
                # Market timing features
                'hour_of_day': now.hour,
                'day_of_week': now.weekday(),
                'minute_of_hour': now.minute,
                
                # Price and volatility features (simulated from current market data)
                'price_ma_5': features.get('sma', current_price),
                'price_ma_20': features.get('sma', current_price),
                'price_momentum_short': 0.0,  # Would need historical data
                'price_momentum_long': 0.0,   # Would need historical data
                
                # Technical indicators
                'rsi': features.get('rsi', 50.0),
                'macd': features.get('macd', 0.0),
                'bollinger_position': 0.5,  # Default neutral position
                
                # Volume features
                'volume_ma': features.get('volume_ma', current_volume),
                'volume_ratio': 1.0,  # Default
                
                # Market structure
                'volatility_estimate': features.get('atr', current_price * 0.02) / current_price,
                'spread_estimate': 0.001,  # 0.1% default spread
                
                # Trade characteristics (estimated)
                'trade_size_btc': estimated_trade_size,
                'trade_value_usd': estimated_trade_value,
                'entry_price': current_price,
                'trade_hour': now.hour,
                'trade_dow': now.weekday(),
                'position_size_ratio': estimated_trade_value / 10000.0,  # Assume $10k portfolio
            }
            
            return trade_features
            
        except Exception as e:
            self.logger.error(f"Error extracting trade-learned features: {e}")
            # Return default features
            now = datetime.now()
            return {
                'price_ma_5': 97000.0, 'price_ma_20': 97000.0, 'price_momentum_short': 0.0,
                'price_momentum_long': 0.0, 'rsi': 50.0, 'macd': 0.0, 'bollinger_position': 0.5,
                'volume_ma': 1000.0, 'volume_ratio': 1.0, 'hour_of_day': now.hour,
                'day_of_week': now.weekday(), 'minute_of_hour': now.minute,
                'volatility_estimate': 0.02, 'spread_estimate': 0.001, 'trade_size_btc': 0.01,
                'trade_value_usd': 970.0, 'entry_price': 97000.0, 'trade_hour': now.hour,
                'trade_dow': now.weekday(), 'position_size_ratio': 0.097
            }
    
    def _technical_analysis_prediction(self, features: dict) -> np.ndarray:
        """Simple technical analysis based prediction as fallback."""
        try:
            # Get indicator values and handle NaN/None
            rsi = features.get('rsi', 50.0)
            if pd.isna(rsi) or rsi is None:
                rsi = 50.0
                
            macd = features.get('macd', 0.0)
            if pd.isna(macd) or macd is None:
                macd = 0.0
                
            signal_line = features.get('signal_line', 0.0)
            if pd.isna(signal_line) or signal_line is None:
                signal_line = 0.0
                
            price = features.get('price', 0.0)
            if pd.isna(price) or price is None:
                price = 0.0
                
            sma = features.get('sma', 0.0)
            if pd.isna(sma) or sma is None:
                sma = price  # Use price as fallback for SMA
            
            self.logger.info(f"Technical analysis - RSI: {rsi:.2f}, MACD: {macd:.4f}, Signal: {signal_line:.4f}, Price: {price:.2f}, SMA: {sma:.2f}")
            
            buy_signals = 0
            sell_signals = 0
            
            # More aggressive RSI signals for testing
            if rsi < 40:  # Oversold (relaxed from 30)
                buy_signals += 1
                self.logger.info("RSI buy signal (oversold)")
            elif rsi > 60:  # Overbought (relaxed from 70)
                sell_signals += 1
                self.logger.info("RSI sell signal (overbought)")
            
            # MACD signals
            if macd > signal_line:
                buy_signals += 1
                self.logger.info("MACD buy signal (macd > signal)")
            elif macd < signal_line:
                sell_signals += 1
                self.logger.info("MACD sell signal (macd < signal)")
            
            # Price vs SMA - more sensitive
            if sma > 0 and price > sma * 1.005:  # Price 0.5% above SMA
                buy_signals += 1
                self.logger.info("Price buy signal (above SMA)")
            elif sma > 0 and price < sma * 0.995:  # Price 0.5% below SMA
                sell_signals += 1
                self.logger.info("Price sell signal (below SMA)")
            
            self.logger.info(f"Signal count - Buy: {buy_signals}, Sell: {sell_signals}")
            
            # Convert to probability distribution with higher confidence for testing
            # For testing purposes, always generate a trade signal
            import random
            if buy_signals > sell_signals:
                confidence = min(0.5 + buy_signals * 0.2, 0.9)  # Higher base confidence
                result = np.array([confidence, 1.0 - confidence, 0.0])  # BUY
                self.logger.info(f"BUY prediction with confidence {confidence:.3f}")
                return result
            elif sell_signals > buy_signals:
                confidence = min(0.5 + sell_signals * 0.2, 0.9)  # Higher base confidence
                result = np.array([0.0, 1.0 - confidence, confidence])  # SELL
                self.logger.info(f"SELL prediction with confidence {confidence:.3f}")
                return result
            else:
                # Force a random trade for testing when no clear signal
                if random.random() > 0.5:
                    confidence = 0.7  # Good confidence for testing
                    result = np.array([confidence, 1.0 - confidence, 0.0])  # BUY
                    self.logger.info(f"FORCED BUY prediction for testing with confidence {confidence:.3f}")
                    return result
                else:
                    confidence = 0.7  # Good confidence for testing
                    result = np.array([0.0, 1.0 - confidence, confidence])  # SELL
                    self.logger.info(f"FORCED SELL prediction for testing with confidence {confidence:.3f}")
                    return result
                
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return np.array([0.0, 1.0, 0.0])  # Default to HOLD

    def calculate_cross_pair_correlation(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate the correlation matrix for the close prices of all symbols.
        """
        close_prices = pd.DataFrame({symbol: df['close'] for symbol, df in data.items()})
        return close_prices.corr()
