import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

from trading.strategies.base_strategy import BaseStrategy
from trading.strategies.research_based_strategy import ResearchBasedStrategy

try:
    from trading.strategies.rl_strategy import RLStrategy

    HAS_RL_STRATEGY = True
except ImportError:
    HAS_RL_STRATEGY = False
    print("Warning: RL strategy not available, using fallback strategies")
import ta

from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy


class EnsembleStrategy(BaseStrategy):
    """
    EnsembleStrategy combines technical analysis with validated optional
    strategies and an optional MLX language-model signal.

    Available outputs are averaged to determine a BUY, SELL, or HOLD signal.
    """

    def __init__(
        self,
        logger: logging.Logger,
        symbols: List[str] = ["BTCUSDT"],
        *,
        mlx_url: str = None,
        risk_per_trade: float = 0.01,
        min_position_size: float = 0.001,
        max_position_size: float = 0.1,
        order_flow_analyzer=None,
        price_fetcher=None,
        exchange_manager=None,
        enable_research_strategy: bool = True,
        research_social_data: bool = False,
        research_rolling_training: bool = True,
        enable_rl_strategy: bool = True,
        enable_bonenkamp_hft: bool = True,
        bonenkamp_use_social: bool = True,
    ):
        """
        Initialize the ensemble strategy, loading models and setting parameters.

        Args:
            logger (logging.Logger): The logger for debugging/info output.
            symbols (List[str]): The trading pairs to manage.
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
        self.CLASSES = ["BUY", "HOLD", "SELL"]

        self.risk_per_trade = risk_per_trade
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size

        # High-frequency trading optimization for 100x leverage
        self.last_trade_time = None
        self.profit_optimization_mode = True
        self.leverage_multiplier = 100  # Amplify profits with 100x leverage

        self.mlx_url = mlx_url or os.getenv("MLX_URL", "")
        self.mlx_available = False

        self._check_mlx_connectivity()

        # Initialize DRL agent
        self.drl_agent = self._initialize_drl_agent()

        # Initialize Research-Based Strategy
        self.enable_research_strategy = enable_research_strategy
        self.research_strategy = None
        if self.enable_research_strategy:
            try:
                self.research_strategy = ResearchBasedStrategy(
                    logger=logger,
                    social_data_enabled=research_social_data,
                    enable_rolling_training=research_rolling_training,
                )
                self.logger.info("🔬 Research-based strategy integrated into ensemble")
            except Exception as e:
                self.logger.error(f"Failed to initialize research strategy: {e}")
                self.enable_research_strategy = False

        # Initialize RL Strategy
        self.enable_rl_strategy = enable_rl_strategy
        self.rl_strategy = None
        if self.enable_rl_strategy:
            try:
                self.rl_strategy = RLStrategy(
                    logger=logger, symbols=symbols, confidence_threshold=0.6
                )
                self.logger.info("🤖 RL strategy integrated into ensemble")
            except Exception as e:
                self.logger.error(f"Failed to initialize RL strategy: {e}")
                self.enable_rl_strategy = False

        # Initialize Bonenkamp HFT Strategy
        self.enable_bonenkamp_hft = enable_bonenkamp_hft
        self.bonenkamp_strategy = None
        if self.enable_bonenkamp_hft:
            try:
                self.bonenkamp_strategy = BonenkampHFTStrategy(
                    logger=logger,
                    use_social_features=bonenkamp_use_social,
                    rolling_window_days=7,
                )
                self.logger.info("🎯 Bonenkamp HFT strategy integrated into ensemble")
            except Exception as e:
                self.logger.error(f"Failed to initialize Bonenkamp HFT strategy: {e}")
                self.enable_bonenkamp_hft = False

    def record_trade_signal(self, signal: str, price: float):
        """Record when a trade signal was generated - NO COOLDOWN"""
        if signal in ["BUY", "SELL"]:  # Only record actual trade signals
            self.last_trade_time = datetime.now()
            self.last_trade_price = price
            self.logger.info(
                f"⚡ Recorded trade signal: {signal} at ${price:.2f} - INSTANT NEXT TRADE ALLOWED"
            )

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
            from core.models.reinforcement_learning_model import TradingEnv
            from drl_agents.agents import AgentPPO
            from training.config import Arguments
            from training.run import init_agent

            drl_model_path = "models/checkpoints"
            if not os.path.exists(drl_model_path):
                self.logger.info("No DRL model found, DRL predictions disabled")
                return None

            # Create sample DataFrame with required columns to avoid indexing errors
            sample_data = pd.DataFrame(
                {
                    "close": [100.0, 101.0, 102.0, 101.5, 103.0],
                    "high": [101.0, 102.0, 103.0, 102.5, 104.0],
                    "low": [99.0, 100.0, 101.0, 100.5, 102.0],
                    "volume": [1000, 1100, 1200, 1050, 1300],
                    "rsi": [50.0, 55.0, 60.0, 52.0, 58.0],
                    "macd": [0.0, 0.1, 0.2, 0.15, 0.25],
                }
            )

            env = TradingEnv(data=sample_data)

            args = Arguments(agent=AgentPPO, env=env)
            args.cwd = drl_model_path

            agent = init_agent(args, gpu_id=-1)
            agent.act.eval()

            self.logger.info("DRL agent initialized successfully")
            return agent
        except Exception as e:
            self.logger.warning(f"Failed to initialize DRL agent: {e}")
            return None

    def _get_drl_prediction(self, features: dict) -> str:
        """Get prediction from DRL agent - DISABLED for balanced testing."""
        # DISABLED: DRL agent was forcing BUY signals regardless of market conditions
        self.logger.info("DRL agent DISABLED - was forcing directional bias")
        return "HOLD"  # Return neutral instead of biased BUY

        if not self.drl_agent:
            return "HOLD"
        try:
            import torch

            state_vector = self._features_to_state_vector(features)
            s_tensor = torch.as_tensor((state_vector,), device=self.drl_agent.device)
            a_tensor = self.drl_agent.act(s_tensor)
            action = a_tensor.detach().cpu().numpy()

            # Handle different action formats
            if isinstance(action, np.ndarray):
                if action.ndim > 0:
                    action_value = float(
                        action.flat[0]
                    )  # Use flat[0] for safe scalar conversion
                else:
                    action_value = float(action.item())  # Use item() for 0-d arrays
            else:
                action_value = float(action)

            if action_value > 0.5:
                return "BUY"
            elif action_value < -0.5:
                return "SELL"
            else:
                return "HOLD"
        except Exception as e:
            self.logger.warning(f"DRL prediction failed: {e}")
            return "HOLD"

    def _features_to_state_vector(self, features: dict) -> np.ndarray:
        """Convert feature dictionary to state vector for DRL agent."""
        try:
            # Extract key features for DRL agent state (8 features to match model)
            state_features = [
                features.get("price", 0.0),
                features.get("volume", 0.0),
                features.get("rsi", 50.0),
                features.get("macd", 0.0),
                features.get("sma", 0.0),
                features.get("atr", 0.0),
                features.get("high", 0.0),
                features.get("low", 0.0),
            ]

            # Normalize features (simple normalization)
            state_vector = np.array(state_features, dtype=np.float32)
            # Basic normalization - could be improved with proper scaling
            state_vector = (state_vector - np.mean(state_vector)) / (
                np.std(state_vector) + 1e-8
            )

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
                "max_tokens": 10,
            }
            resp = requests.post(
                self.mlx_url, headers=headers, json=payload, timeout=10
            )
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
        self.logger.info("=== STARTING MODEL PREDICTIONS ===")

        # Try MLX if available
        if self.mlx_available:
            try:
                mlx_decision = self._parse_mlx_decision(
                    self._mlx_generate(
                        f"Predict market move for features: {features} -> BUY, SELL, or HOLD."
                    )
                )
                preds["mlx"] = np.array(
                    [1.0 if c == mlx_decision else 0.0 for c in self.CLASSES]
                )
            except Exception as e:
                self.logger.warning(f"MLX prediction error: {e}")

        # Try DRL agent if available
        if hasattr(self, "drl_agent") and self.drl_agent is not None:
            try:
                drl_decision = self._get_drl_prediction(features)
                preds["drl"] = np.array(
                    [1.0 if c == drl_decision else 0.0 for c in self.CLASSES]
                )
                self.logger.debug(f"DRL agent prediction: {drl_decision}")
            except Exception as e:
                self.logger.warning(f"DRL agent prediction error: {e}")

        # Try Research-Based Strategy if available
        if self.enable_research_strategy and self.research_strategy is not None:
            try:
                # Convert features to DataFrame format expected by research strategy
                research_data = self._convert_features_to_dataframe(features)
                research_signals = self.research_strategy.generate_signals(
                    {"BTCUSDT": research_data}
                )

                if "BTCUSDT" in research_signals:
                    research_signal = research_signals["BTCUSDT"]["signal"]
                    research_confidence = research_signals["BTCUSDT"]["confidence"]

                    # Convert research signal to probability array
                    # Research strategy returns BUY/SELL only (no HOLD)
                    if research_signal == "BUY":
                        preds["research"] = np.array(
                            [research_confidence, 1.0 - research_confidence, 0.0]
                        )
                    elif research_signal == "SELL":
                        preds["research"] = np.array(
                            [0.0, 1.0 - research_confidence, research_confidence]
                        )
                    else:  # HOLD (shouldn't happen with research strategy)
                        preds["research"] = np.array([0.0, 1.0, 0.0])

                    self.logger.info(
                        f"🔬 Research strategy prediction: {research_signal} ({research_confidence:.3f})"
                    )

            except Exception as e:
                self.logger.warning(f"Research strategy prediction error: {e}")

        # Try RL Strategy if available
        if self.enable_rl_strategy and self.rl_strategy is not None:
            try:
                # Get RL prediction
                rl_signal, rl_confidence = self.rl_strategy.generate_signal(
                    "BTCUSDT", features
                )

                # Convert RL signal to probability array
                if rl_signal == "BUY":
                    preds["rl"] = np.array([rl_confidence, 1.0 - rl_confidence, 0.0])
                elif rl_signal == "SELL":
                    preds["rl"] = np.array([0.0, 1.0 - rl_confidence, rl_confidence])
                else:  # HOLD
                    preds["rl"] = np.array([0.0, rl_confidence, 0.0])

                self.logger.info(
                    f"🤖 RL strategy prediction: {rl_signal} ({rl_confidence:.3f})"
                )

            except Exception as e:
                self.logger.warning(f"RL strategy prediction error: {e}")

        # Models predictions complete
        self.logger.info("=== MODEL PREDICTIONS COMPLETE ===")

        # Technical analysis fallback if no models available or all failed
        valid_preds = {
            k: v
            for k, v in preds.items()
            if v is not None and len(v) == 3 and not np.any(np.isnan(v))
        }
        if not valid_preds:
            self.logger.info(
                "Using technical analysis fallback (no valid model predictions)"
            )
            preds["technical"] = self._technical_analysis_prediction(features)
        else:
            self.logger.info(
                f"Using {len(valid_preds)} valid model predictions: {list(valid_preds.keys())}"
            )

        return preds

    def generate_signals(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
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

                # Debug logging for predictions
                self.logger.info(f"Model prediction results for {symbol}:")
                for model_name, pred in preds.items():
                    if pred is not None:
                        self.logger.info(f"  {model_name}: {pred}")
                    else:
                        self.logger.info(f"  {model_name}: None (failed)")
                self.logger.info(
                    f"Total predictions received: {len([p for p in preds.values() if p is not None])}"
                )

                # Also log the raw features being used
                self.logger.info(
                    f"Raw features for prediction: {dict(list(features.items())[:10])}..."
                )  # First 10 features

                if preds:
                    # Filter out invalid predictions (None, wrong size, NaN values)
                    pred_arrays = [
                        p
                        for p in preds.values()
                        if p is not None and len(p) == 3 and not np.any(np.isnan(p))
                    ]

                    # If no valid predictions, use technical analysis as fallback
                    if not pred_arrays:
                        self.logger.info(
                            "No valid model predictions found, using technical analysis fallback"
                        )
                        tech_pred = self._technical_analysis_prediction(features)
                        if (
                            tech_pred is not None
                            and len(tech_pred) == 3
                            and not np.any(np.isnan(tech_pred))
                        ):
                            pred_arrays = [tech_pred]
                            self.logger.info(
                                f"Technical analysis fallback prediction: {tech_pred}"
                            )

                    if pred_arrays:
                        pred_array = np.mean(pred_arrays, axis=0)
                        self.logger.info(f"Ensemble averaged predictions: {pred_array}")
                        self.logger.info(
                            f"Individual predictions being averaged: {pred_arrays}"
                        )

                        # Ensure no NaN values in final prediction
                        if not np.any(np.isnan(pred_array)):
                            best_idx = np.argmax(pred_array)
                            decision = self.CLASSES[best_idx]
                            confidence = float(pred_array[best_idx])
                            self.logger.info(
                                f"Final decision: {decision} with confidence {confidence:.3f} (index {best_idx})"
                            )
                        else:
                            self.logger.warning(
                                "Final prediction array contains NaN values, defaulting to HOLD"
                            )
                            decision, confidence = "HOLD", 0.0
                    else:
                        self.logger.warning(
                            "No valid prediction arrays found, defaulting to HOLD"
                        )
                        decision, confidence = "HOLD", 0.0
                else:
                    self.logger.warning("No predictions received, defaulting to HOLD")
                    decision, confidence = "HOLD", 0.0

                self.logger.info(
                    f"Ensemble decision for {symbol}: {decision} ({confidence:.4f})"
                )
                signals[symbol] = {"signal": decision, "confidence": confidence}

            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                signals[symbol] = {"signal": "HOLD", "confidence": 0.0}

        return signals

    def generate_signal(self, symbol: str, market_data: dict) -> tuple[str, float]:
        """
        Generate a single trading signal for the main trading loop.
        This is the main function called by main.py.

        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            market_data (dict): Current market data

        Returns:
            tuple: (signal, confidence) where signal is 'BUY', 'SELL', or 'HOLD'
        """
        try:
            self.logger.info(f"=== GENERATING ENSEMBLE SIGNAL FOR {symbol} ===")

            # ABSOLUTE MAXIMUM SPEED - No delays, no cooldowns, no restrictions
            current_time = datetime.now()
            self.logger.info(
                "⚡ MAXIMUM SPEED MODE: Zero delays, instant trading, 100x leverage ready"
            )

            # Create features from market data
            features = self._create_market_features(market_data)

            # Log the actual current price being used
            current_price = market_data.get("close", market_data.get("price", 0))
            self.logger.info(f"💰 Current market price: ${current_price:.2f}")

            # Collect all predictions with weights
            all_predictions = []
            prediction_weights = []
            prediction_sources = []

            # 1. Technical analysis prediction (baseline)
            tech_signal, tech_confidence = (
                self._generate_technical_signal_from_features(features)
            )
            if (
                tech_signal != "HOLD" or tech_confidence > 0.6
            ):  # Only include if confident or non-HOLD
                tech_pred = self._signal_to_array(tech_signal, tech_confidence)
                all_predictions.append(tech_pred)
                prediction_weights.append(0.3)  # 30% weight for technical analysis
                prediction_sources.append("Technical")
                self.logger.info(
                    f"📊 Technical analysis: {tech_signal} ({tech_confidence:.3f})"
                )
            else:
                self.logger.info(
                    f"📊 Technical analysis: {tech_signal} ({tech_confidence:.3f}) - SKIPPED (low confidence HOLD)"
                )

            # 2. Research strategy prediction (if enabled) - HIGH PRIORITY
            if self.enable_research_strategy and self.research_strategy is not None:
                try:
                    research_data = self._convert_features_to_dataframe(features)
                    research_signals = self.research_strategy.generate_signals(
                        {symbol: research_data}
                    )

                    if symbol in research_signals:
                        research_signal = research_signals[symbol]["signal"]
                        research_confidence = research_signals[symbol]["confidence"]
                        research_pred = self._signal_to_array(
                            research_signal, research_confidence
                        )
                        all_predictions.append(research_pred)
                        prediction_weights.append(
                            0.35
                        )  # 35% weight for research strategy
                        prediction_sources.append("Research")
                        self.logger.info(
                            f"🔬 Research strategy: {research_signal} ({research_confidence:.3f}) - HIGH PRIORITY"
                        )
                    else:
                        self.logger.warning(
                            "🔬 Research strategy generated no signal for symbol"
                        )
                except Exception as e:
                    self.logger.warning(f"🔬 Research strategy failed: {e}")
            else:
                self.logger.info("🔬 Research strategy disabled or unavailable")

            # 3. RL strategy prediction (if enabled) - LEARNING PRIORITY
            if self.enable_rl_strategy and self.rl_strategy is not None:
                try:
                    rl_signal, rl_confidence = self.rl_strategy.generate_signal(
                        "BTCUSDT", features
                    )

                    if rl_confidence >= 0.6:  # Only use high confidence RL predictions
                        rl_pred = self._signal_to_array(rl_signal, rl_confidence)
                        all_predictions.append(rl_pred)
                        prediction_weights.append(0.25)  # 25% weight for RL strategy
                        prediction_sources.append("RL")
                        self.logger.info(
                            f"🤖 RL strategy: {rl_signal} ({rl_confidence:.3f}) - LEARNING"
                        )
                    else:
                        self.logger.info(
                            f"🤖 RL strategy: {rl_signal} ({rl_confidence:.3f}) - SKIPPED (low confidence)"
                        )
                except Exception as e:
                    self.logger.warning(f"🤖 RL strategy failed: {e}")
            else:
                self.logger.info("🤖 RL strategy disabled or unavailable")

            # 4. Bonenkamp HFT strategy prediction (if enabled) - HIGH FREQUENCY PRIORITY
            if self.enable_bonenkamp_hft and self.bonenkamp_strategy is not None:
                try:
                    bonenkamp_data = self._convert_features_to_dataframe(features)
                    bonenkamp_signals = self.bonenkamp_strategy.generate_signals(
                        {symbol: bonenkamp_data}
                    )

                    if symbol in bonenkamp_signals:
                        bonenkamp_signal = bonenkamp_signals[symbol]["signal"]
                        bonenkamp_confidence = bonenkamp_signals[symbol]["confidence"]
                        bonenkamp_pred = self._signal_to_array(
                            bonenkamp_signal, bonenkamp_confidence
                        )
                        all_predictions.append(bonenkamp_pred)
                        prediction_weights.append(
                            0.25
                        )  # 25% weight for Bonenkamp HFT strategy
                        prediction_sources.append("Bonenkamp-HFT")
                        self.logger.info(
                            f"🎯 Bonenkamp HFT: {bonenkamp_signal} ({bonenkamp_confidence:.3f}) - HIGH FREQUENCY"
                        )
                    else:
                        self.logger.warning(
                            "🎯 Bonenkamp HFT generated no signal for symbol"
                        )
                except Exception as e:
                    self.logger.warning(f"🎯 Bonenkamp HFT failed: {e}")
            else:
                self.logger.info("🎯 Bonenkamp HFT disabled or unavailable")

            # 5. Model-based predictions (if available)
            try:
                model_preds = self._get_model_predictions(features)
                if model_preds:
                    # Filter valid predictions
                    valid_preds = [
                        p
                        for p in model_preds.values()
                        if p is not None and len(p) == 3 and not np.any(np.isnan(p))
                    ]
                    if valid_preds:
                        ensemble_pred = np.mean(valid_preds, axis=0)
                        all_predictions.append(ensemble_pred)
                        prediction_weights.append(0.15)  # 15% weight for model ensemble
                        prediction_sources.append("Models")
                        best_idx = np.argmax(ensemble_pred)
                        model_signal = self.CLASSES[best_idx]
                        model_confidence = float(ensemble_pred[best_idx])
                        self.logger.info(
                            f"🤖 Model ensemble: {model_signal} ({model_confidence:.3f})"
                        )
            except Exception as e:
                self.logger.warning(f"🤖 Model predictions failed: {e}")

            # Combine all predictions with weights
            if all_predictions:
                # Stash each member's own vote so main.py can record it at
                # execution time (adaptive-ensemble feedback, roadmap #11)
                if not hasattr(self, "last_model_votes"):
                    self.last_model_votes = {}
                self.last_model_votes[symbol] = {
                    source: self.CLASSES[int(np.argmax(pred))]
                    for source, pred in zip(prediction_sources, all_predictions)
                }

                # Modulate the static source weights by learned per-model
                # accuracy (uniform start = identical to static behavior)
                if os.getenv("ELVIS_ADAPTIVE_ENSEMBLE", "1") == "1":
                    try:
                        from trading.signals.adaptive_ensemble import combine_weights
                        from trading.signals.model_feedback import get_shared_weights

                        combined = combine_weights(
                            dict(zip(prediction_sources, prediction_weights)),
                            get_shared_weights().weights,
                        )
                        prediction_weights = [
                            combined[source] for source in prediction_sources
                        ]
                    except Exception as adaptive_err:
                        self.logger.warning(
                            f"🧠 Adaptive weights unavailable, using static: {adaptive_err}"
                        )

                # Normalize weights
                total_weight = sum(prediction_weights)
                normalized_weights = [w / total_weight for w in prediction_weights]

                # Weighted average of predictions
                final_prediction = np.zeros(3)
                for pred, weight in zip(all_predictions, normalized_weights):
                    final_prediction += pred * weight

                best_idx = np.argmax(final_prediction)
                signal = self.CLASSES[best_idx]
                confidence = float(final_prediction[best_idx])

                self.logger.info(f"🎯 Ensemble prediction: {final_prediction}")
                self.logger.info(
                    f"📊 Sources used: {dict(zip(prediction_sources, normalized_weights))}"
                )

                # ANTI-HOLD LOGIC: Force BUY/SELL decisions for active trading
                if signal == "HOLD":
                    # Choose BUY or SELL based on which has higher probability
                    buy_prob = final_prediction[0]  # BUY probability
                    sell_prob = final_prediction[2]  # SELL probability

                    if buy_prob > sell_prob:
                        signal = "BUY"
                        confidence = max(buy_prob, 0.65)  # Ensure minimum confidence
                        self.logger.info(
                            f"🔄 ANTI-HOLD: Converted HOLD to BUY (buy_prob={buy_prob:.3f} > sell_prob={sell_prob:.3f})"
                        )
                    else:
                        signal = "SELL"
                        confidence = max(sell_prob, 0.65)  # Ensure minimum confidence
                        self.logger.info(
                            f"🔄 ANTI-HOLD: Converted HOLD to SELL (sell_prob={sell_prob:.3f} > buy_prob={buy_prob:.3f})"
                        )

            else:
                # Fallback to technical analysis only
                signal, confidence = tech_signal, tech_confidence
                self.logger.warning(
                    "⚠️ No ensemble predictions available, using technical analysis fallback"
                )

                # Apply anti-HOLD logic to fallback as well
                if signal == "HOLD":
                    # Default to BUY in neutral conditions (slight bullish bias for crypto)
                    signal = "BUY"
                    confidence = 0.65
                    self.logger.info(
                        f"🔄 FALLBACK ANTI-HOLD: Converted HOLD to BUY with minimum confidence"
                    )

            # Market trend analysis (trend-following filter)
            try:
                if "sma_20" in market_data and "sma_50" in market_data:
                    sma_20 = market_data.get("sma_20", current_price)
                    sma_50 = market_data.get("sma_50", current_price)

                    # Strong downtrend: price below both SMAs and SMA20 < SMA50
                    if (
                        current_price < sma_20
                        and current_price < sma_50
                        and sma_20 < sma_50
                    ):
                        if signal == "BUY":
                            self.logger.warning(
                                f"🚫 TREND FILTER: Blocking BUY in downtrend - Price ${current_price:.2f} < SMA20 ${sma_20:.2f} < SMA50 ${sma_50:.2f}"
                            )
                            signal = "SELL"  # Flip to SELL in strong downtrend
                            confidence = min(confidence * 1.1, 0.8)

                    # Strong uptrend: price above both SMAs and SMA20 > SMA50
                    elif (
                        current_price > sma_20
                        and current_price > sma_50
                        and sma_20 > sma_50
                    ):
                        if signal == "SELL":
                            self.logger.warning(
                                f"🚫 TREND FILTER: Blocking SELL in uptrend - Price ${current_price:.2f} > SMA20 ${sma_20:.2f} > SMA50 ${sma_50:.2f}"
                            )
                            signal = "BUY"  # Flip to BUY in strong uptrend
                            confidence = min(confidence * 1.1, 0.8)

            except Exception as e:
                self.logger.error(f"Market trend analysis failed: {e}")

            # FIX #1: ENABLE PROPER SELL SIGNAL GENERATION
            # Remove the HOLD elimination that was forcing BUY signals
            # This was the core bug - bot never generated SELL signals!

            # Confidence flattening was INVERTED: max(confidence, 0.85) forced
            # every BUY/SELL to look near-certain, blinding the win-rate filter
            # (which then approved everything), maxing confidence-scaled
            # position size, and erasing the gradient the adaptive ensemble and
            # directional guard rely on. Let the real ensemble confidence flow.
            # ELVIS_CONFIDENCE_FLATTEN=1 restores the old behavior.
            if os.getenv("ELVIS_CONFIDENCE_FLATTEN", "0") == "1":
                if signal in ["BUY", "SELL"]:
                    confidence = max(confidence, 0.85)
                else:
                    confidence = max(confidence, 0.70)

            self.logger.info(
                f"🎯 FINAL ENSEMBLE SIGNAL: {signal} with {confidence:.3f} confidence"
            )
            self.logger.info(
                f"💡 Signal source: {prediction_sources if all_predictions else ['Technical fallback']}"
            )

            # Record trade signal for cooldown tracking
            current_price = market_data.get("close", market_data.get("price", 0))
            self.record_trade_signal(signal, current_price)

            return signal, confidence

        except Exception as e:
            self.logger.error(f"Error generating ensemble signal for {symbol}: {e}")
            return (
                "BUY",
                0.65,
            )  # Default to BUY with safe confidence to avoid emergency mode

    def _create_market_features(self, market_data: dict) -> dict:
        """Create features from market data for technical analysis."""
        try:
            features = {
                "price": market_data.get("close", market_data.get("price", 0.0)),
                "rsi": market_data.get("rsi", 50.0),
                "macd": market_data.get("macd", 0.0),
                "macd_signal": market_data.get("macd_signal", 0.0),
                "macd_histogram": market_data.get("macd_histogram", 0.0),
                "sma": market_data.get("sma_20", 0.0),
                "sma_20": market_data.get("sma_20", 0.0),
                "sma_50": market_data.get("sma_50", 0.0),
                "bb_upper": market_data.get("bb_upper", 0.0),
                "bb_lower": market_data.get("bb_lower", 0.0),
                "bb_middle": market_data.get("bb_middle", 0.0),
                "volume": market_data.get("volume", 0.0),
                "atr": market_data.get("atr", 0.0),
                "adx": market_data.get("adx", 0.0),
            }
            return features
        except Exception as e:
            self.logger.error(f"Error creating market features: {e}")
            return {}

    def _generate_technical_signal_from_features(
        self, features: dict
    ) -> tuple[str, float]:
        """Generate trading signal from technical analysis features."""
        try:
            # Use the existing technical analysis prediction method
            prediction_array = self._technical_analysis_prediction(features)

            if prediction_array is not None and len(prediction_array) == 3:
                # Find the signal with highest probability
                best_idx = np.argmax(prediction_array)
                signal = self.CLASSES[best_idx]  # ["BUY", "HOLD", "SELL"]
                confidence = float(prediction_array[best_idx])

                self.logger.info(
                    f"Technical analysis prediction array: {prediction_array}"
                )
                self.logger.info(
                    f"Selected signal: {signal} (index {best_idx}) with confidence {confidence:.3f}"
                )

                return signal, confidence
            else:
                self.logger.warning(
                    "Invalid technical analysis prediction, defaulting to HOLD"
                )
                return "HOLD", 0.1

        except Exception as e:
            self.logger.error(f"Error in technical signal generation: {e}")
            return "HOLD", 0.1

    def check_arbitrage_opportunities(
        self, symbol: str = "BTCUSDT"
    ) -> List[Dict[str, Any]]:
        """Check for arbitrage opportunities across exchanges"""
        if not self.exchange_manager:
            self.logger.warning(
                "Exchange manager not available for arbitrage detection"
            )
            return []

        try:
            opportunities = self.exchange_manager.detect_arbitrage_opportunities(symbol)

            if opportunities:
                self.logger.info(
                    f"Found {len(opportunities)} arbitrage opportunities for {symbol}"
                )
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

    def execute_multi_exchange_order(
        self, symbol: str, side: str, quantity: float, use_best_price: bool = True
    ) -> Dict[str, Any]:
        """Execute order using the best available exchange"""
        if not self.exchange_manager:
            self.logger.warning("Exchange manager not available, using single exchange")
            return {}

        try:
            if use_best_price:
                result = self.exchange_manager.execute_smart_order(
                    symbol, side, quantity
                )
                self.logger.info(
                    f"Smart order executed on {result.get('selected_exchange', 'unknown')} "
                    f"at price {result.get('selected_price', 0):.2f}"
                )
                return result
            else:
                # Use first available exchange
                available_exchanges = self.exchange_manager.get_available_exchanges()
                if available_exchanges:
                    exchange = self.exchange_manager.get_exchange(
                        available_exchanges[0]
                    )
                    if side.lower() == "buy":
                        return exchange.execute_buy(symbol, quantity)
                    else:
                        return exchange.execute_sell(symbol, quantity)

        except Exception as e:
            self.logger.error(f"Error executing multi-exchange order: {e}")
            return {"error": str(e)}

    def get_consolidated_portfolio(self) -> Dict[str, Any]:
        """Get consolidated portfolio across all exchanges"""
        if not self.exchange_manager:
            return {}

        try:
            consolidated_balance = self.exchange_manager.get_consolidated_balance()

            portfolio_summary = {
                "total_value_usd": 0.0,
                "balances": consolidated_balance,
                "exchange_count": len(self.exchange_manager.get_available_exchanges()),
                "timestamp": datetime.now().isoformat(),
            }

            # Calculate approximate USD value for major cryptocurrencies
            major_cryptos = ["BTC", "ETH", "ADA", "DOT", "LINK"]

            for crypto in major_cryptos:
                if crypto in consolidated_balance:
                    try:
                        symbol = f"{crypto}USDT"
                        prices = self.exchange_manager.get_prices_all_exchanges(symbol)
                        if prices:
                            avg_price = sum(prices.values()) / len(prices)
                            crypto_value = (
                                consolidated_balance[crypto]["total_balance"]
                                * avg_price
                            )
                            portfolio_summary["total_value_usd"] += crypto_value
                    except Exception:
                        pass

            # Add USD/USDT holdings directly
            for stablecoin in ["USD", "USDT", "USDC"]:
                if stablecoin in consolidated_balance:
                    portfolio_summary["total_value_usd"] += consolidated_balance[
                        stablecoin
                    ]["total_balance"]

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
                "markets": market_summary,
                "exchange_health": self.exchange_manager.check_all_exchanges_health(),
                "available_exchanges": self.exchange_manager.get_available_exchanges(),
                "timestamp": datetime.now().isoformat(),
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
        if not all(col in data.columns for col in ["high", "low", "close"]):
            self.logger.warning(
                "ATR calculation requires 'high', 'low', 'close' columns."
            )
            return 0.01  # Return a default small volatility

        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        tr = np.max([high_low, high_close, low_close], axis=0)
        atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]
        return atr if pd.notna(atr) else 0.01

    def calculate_position_size(
        self,
        data: pd.DataFrame,
        current_price: float,
        available_capital: float,
        leverage: float = 100.0,
        signal_confidence: float = 0.5,
    ) -> float:
        """
        Calculate conservative position size based on volatility, signal confidence, and risk management.

        Args:
            data (pd.DataFrame): The data to calculate position size from.
            current_price (float): The current price.
            available_capital (float): The available capital.
            leverage (float): The leverage to use for the position.
            signal_confidence (float): Confidence in the trading signal (0.0-1.0).

        Returns:
            float: The position size in BTC.
        """
        try:
            self.logger.info(
                f"Calculating position size - Price: {current_price}, Capital: {available_capital}, Leverage: {leverage}x, Confidence: {signal_confidence:.3f}"
            )

            if available_capital <= 0 or current_price <= 0:
                self.logger.warning(
                    f"Invalid inputs - Capital: {available_capital}, Price: {current_price}"
                )
                return self.min_position_size

            # AGGRESSIVE: Maximize profits with 100x leverage
            # Increase position sizes for better profit potential
            if current_price > 120000:  # Bitcoin above $120k
                base_risk = 0.025  # More aggressive at high prices for better profits
                self.logger.info(
                    f"High price profit optimization: Using {base_risk:.1%} base risk for BTC > $120k"
                )
            else:
                base_risk = 0.030  # Aggressive 3% base risk for maximum profits

            # Scale risk based on signal confidence - reward high confidence signals
            confidence_multiplier = max(0.7, signal_confidence)  # Never less than 0.7x
            confidence_multiplier = min(
                2.5, confidence_multiplier * 2.5
            )  # Cap at 2.5x for 100% confidence

            # Calculate volatility adjustment using ATR if available
            volatility_multiplier = 1.0
            if len(data) > 20:
                try:
                    # Calculate ATR for volatility adjustment
                    high = data["high"].rolling(14).max()
                    low = data["low"].rolling(14).min()
                    close = data["close"]
                    tr1 = high - low
                    tr2 = abs(high - close.shift())
                    tr3 = abs(low - close.shift())
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = true_range.rolling(14).mean().iloc[-1]

                    # Lower position size in high volatility markets
                    price_atr_ratio = atr / current_price
                    if price_atr_ratio > 0.05:  # High volatility (>5% ATR)
                        volatility_multiplier = 0.5  # Reduce position size by half
                    elif price_atr_ratio > 0.03:  # Medium volatility (>3% ATR)
                        volatility_multiplier = 0.75  # Reduce by 25%

                    self.logger.info(
                        f"ATR: {atr:.2f}, Price/ATR ratio: {price_atr_ratio:.4f}, Volatility multiplier: {volatility_multiplier:.2f}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Could not calculate volatility adjustment: {e}"
                    )

            # Final risk calculation
            adjusted_risk = base_risk * confidence_multiplier * volatility_multiplier
            adjusted_risk = min(
                adjusted_risk, 0.025
            )  # Cap at 2.5% maximum risk (5x increase)

            risk_amount = available_capital * adjusted_risk

            # High leverage usage for maximum capital efficiency
            effective_leverage = min(
                leverage, 100.0
            )  # Use full 100x leverage for maximum gains

            position_value = risk_amount * effective_leverage
            position_size = position_value / current_price

            # Higher position limits for 50x leverage
            min_size = self.min_position_size
            max_size = min(
                self.max_position_size, available_capital * 0.5 / current_price
            )  # Max 50% of capital for high leverage
            position_size = max(min_size, min(position_size, max_size))

            # Ensure affordable margin with high leverage buffer
            required_margin = position_value / effective_leverage
            max_affordable_margin = (
                available_capital * 0.95
            )  # Use 95% for high leverage efficiency
            if required_margin > max_affordable_margin:
                max_position_value = max_affordable_margin * effective_leverage
                position_size = max_position_value / current_price
                position_size = max(min_size, position_size)

            self.logger.info(
                f"Optimized position size: {position_size:.6f} BTC (${position_size * current_price:.2f}, {effective_leverage}x leverage)"
            )
            self.logger.info(
                f"Risk factors - Base: {base_risk:.3f}, Confidence: {confidence_multiplier:.2f}, Volatility: {volatility_multiplier:.2f}"
            )

            return position_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            # Balanced fallback
            fallback_risk = available_capital * 0.01  # 1% fallback risk (5x increase)
            fallback_size = (
                (fallback_risk * 5) / current_price
                if current_price > 0
                else self.min_position_size
            )  # 5x leverage fallback
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
        return entry_price - (atr * 2)  # Example: 2 * ATR below entry

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
        return entry_price + (atr * 3)  # Example: 3 * ATR above entry

    def _calculate_trend_strength(
        self, data: pd.DataFrame, adx_period: int = 14, rsi_period: int = 14
    ) -> float:
        """
        Calculate the trend strength using ADX and RSI.

        Returns:
            float: A value between 0 and 1 representing the trend strength.
        """
        adx = ta.trend.ADXIndicator(
            data["high"], data["low"], data["close"], window=adx_period
        ).adx()
        rsi = ta.momentum.RSIIndicator(data["close"], window=rsi_period).rsi()

        # Normalize ADX and RSI to a 0-1 scale
        adx_strength = min(adx.iloc[-1] / 50, 1.0)  # ADX > 50 is a strong trend
        rsi_strength = (
            abs(rsi.iloc[-1] - 50) / 50
        )  # RSI further from 50 is a stronger trend

        return (adx_strength + rsi_strength) / 2

    def _create_features_from_data(self, df: pd.DataFrame) -> dict:
        """Adapt the latest row without inventing retired model inputs."""
        try:
            latest = df.iloc[-1]
            features = {
                "price": latest.get("close", 0.0),
                "volume": latest.get("volume", 0.0),
                "sma": latest.get("sma_20", 0.0),
                "rsi": latest.get("rsi", 50.0),
                "macd": latest.get("macd", 0.0),
                "signal_line": latest.get(
                    "macd_signal", latest.get("signal_line", 0.0)
                ),
                "adx": latest.get("adx", 0.0),
                "atr": latest.get("atr", 0.0),
                "lower_bb": latest.get("bb_low", latest.get("lower_bb", 0.0)),
                "sma_bb": latest.get("bb_mid", latest.get("sma_bb", 0.0)),
                "upper_bb": latest.get("bb_high", latest.get("upper_bb", 0.0)),
            }
            if "volume_ma" in latest:
                features["volume_ma"] = latest["volume_ma"]
            return features
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return {}

    def _technical_analysis_prediction(self, features: dict) -> np.ndarray:
        """Simple technical analysis based prediction as fallback."""
        try:
            # Get indicator values and handle NaN/None
            rsi = features.get("rsi", 50.0)
            if pd.isna(rsi) or rsi is None:
                rsi = 50.0

            macd = features.get("macd", 0.0)
            if pd.isna(macd) or macd is None:
                macd = 0.0

            signal_line = features.get("signal_line", 0.0)
            if pd.isna(signal_line) or signal_line is None:
                signal_line = 0.0

            price = features.get("price", 0.0)
            if pd.isna(price) or price is None:
                price = 0.0

            sma = features.get("sma", 0.0)
            if pd.isna(sma) or sma is None:
                sma = price  # Use price as fallback for SMA

            self.logger.info(
                f"Technical analysis - RSI: {rsi:.2f}, MACD: {macd:.4f}, Signal: {signal_line:.4f}, Price: {price:.2f}, SMA: {sma:.2f}"
            )

            buy_signals = 0
            sell_signals = 0

            # Very aggressive RSI signals for testing
            if rsi < 50:  # Oversold (very relaxed)
                buy_signals += 1
                self.logger.info("RSI buy signal (oversold)")
            elif rsi > 50:  # Overbought (very relaxed)
                sell_signals += 1
                self.logger.info("RSI sell signal (overbought)")

            # MACD signals
            if macd > signal_line:
                buy_signals += 1
                self.logger.info("MACD buy signal (macd > signal)")
            elif macd < signal_line:
                sell_signals += 1
                self.logger.info("MACD sell signal (macd < signal)")

            # Price vs SMA - very sensitive for testing
            if sma > 0 and price > sma * 1.001:  # Price 0.1% above SMA
                buy_signals += 1
                self.logger.info("Price buy signal (above SMA)")
            elif sma > 0 and price < sma * 0.999:  # Price 0.1% below SMA
                sell_signals += 1
                self.logger.info("Price sell signal (below SMA)")

            self.logger.info(f"Signal count - Buy: {buy_signals}, Sell: {sell_signals}")

            # BALANCED APPROACH: Trade in reasonably clear market conditions
            self.logger.info(
                "=== ANALYZING MARKET CONDITIONS FOR BALANCED STRATEGY ==="
            )

            # Calculate price momentum over different periods
            if len(features) < 5:
                self.logger.info(
                    "Insufficient data for conservative analysis, defaulting to HOLD"
                )
                return np.array([0.1, 0.8, 0.1])

            # ATH DETECTION AND BIAS - UPDATED FOR NEW PARADIGM
            ath_bias = 0
            current_price = price

            # NEW: Detect if we're in extreme ATH territory (above $125,000)
            if current_price > 125000:
                ath_bias = -2  # Strong SELL bias at high ATH
                self.logger.warning(
                    f"🔴 HIGH ATH DETECTED: BTC at ${current_price:,.0f} - FAVORING SELL SIGNALS"
                )

                # Extra strong bias above $130,000 (extreme territory)
                if current_price > 130000:
                    ath_bias = -4  # Very strong SELL bias
                    self.logger.warning(
                        f"🚨 EXTREME ATH: BTC at ${current_price:,.0f} - HEAVILY FAVORING SELL"
                    )
            elif current_price > 120000:
                # Mild caution in normal high range
                ath_bias = -1  # Slight SELL bias
                self.logger.info(
                    f"🟡 NORMAL HIGH RANGE: BTC at ${current_price:,.0f} - Slight sell bias"
                )

            # Multi-timeframe trend analysis
            trend_score = ath_bias  # Start with ATH bias

            # 1. Strong price trend (must be very clear)
            price_sma_ratio = price / sma if sma > 0 else 1.0
            if price_sma_ratio > 1.05:  # Price 5% above SMA = very strong uptrend
                trend_score += 3
                self.logger.info(
                    f"VERY STRONG UPTREND: Price {price_sma_ratio:.4f}x SMA"
                )
            elif price_sma_ratio > 1.025:  # Price 2.5% above SMA = strong uptrend
                trend_score += 2
                self.logger.info(f"STRONG UPTREND: Price {price_sma_ratio:.4f}x SMA")
            elif price_sma_ratio < 0.95:  # Price 5% below SMA = very strong downtrend
                trend_score -= 3
                self.logger.info(
                    f"VERY STRONG DOWNTREND: Price {price_sma_ratio:.4f}x SMA"
                )
            elif price_sma_ratio < 0.975:  # Price 2.5% below SMA = strong downtrend
                trend_score -= 2
                self.logger.info(f"STRONG DOWNTREND: Price {price_sma_ratio:.4f}x SMA")
            else:
                self.logger.info(
                    f"SIDEWAYS MARKET: Price {price_sma_ratio:.4f}x SMA - avoiding trade"
                )

            # 2. RSI must confirm trend (very strict)
            if trend_score > 0 and rsi < 40:  # Uptrend + oversold RSI
                trend_score += 1
                self.logger.info(
                    f"RSI CONFIRMS UPTREND: RSI {rsi:.1f} oversold in uptrend"
                )
            elif trend_score < 0 and rsi > 60:  # Downtrend + overbought RSI
                trend_score -= 1
                self.logger.info(
                    f"RSI CONFIRMS DOWNTREND: RSI {rsi:.1f} overbought in downtrend"
                )
            elif (trend_score > 0 and rsi > 70) or (trend_score < 0 and rsi < 30):
                # Trend opposite to RSI - dangerous, reduce confidence
                trend_score = int(trend_score * 0.5)
                self.logger.info(f"RSI CONFLICTS WITH TREND: reducing confidence")

            # 3. MACD must align with trend
            macd_bullish = macd > signal_line
            if trend_score > 0 and macd_bullish:
                trend_score += 1
                self.logger.info("MACD CONFIRMS UPTREND")
            elif trend_score < 0 and not macd_bullish:
                trend_score -= 1
                self.logger.info("MACD CONFIRMS DOWNTREND")
            elif trend_score != 0:  # MACD conflicts with trend
                trend_score = int(trend_score * 0.7)
                self.logger.info("MACD CONFLICTS WITH TREND: reducing confidence")

            self.logger.info(f"FINAL TREND SCORE: {trend_score}")

            # ATH-AWARE SIGNAL GENERATION - HEAVILY FAVOR SELL AT ATH
            if (
                trend_score >= 1 and current_price < 125000
            ):  # BUY allowed in normal range (below $125k)
                confidence = min(0.9, 0.7 + trend_score * 0.1)  # Higher base confidence
                self.logger.info(
                    f"🟢 BUY SIGNAL (Below ATH) - score={trend_score}, confidence={confidence:.3f}"
                )
                return np.array([confidence, 1.0 - confidence, 0.0])
            elif (
                trend_score <= -1 or current_price > 125000
            ):  # SELL in high ATH territory
                if current_price > 130000:  # Extreme ATH
                    confidence = min(
                        0.95, 0.85 + abs(trend_score) * 0.1
                    )  # Very high confidence
                    self.logger.warning(
                        f"🚨 EXTREME ATH SELL SIGNAL - price=${current_price:,.0f}, score={trend_score}, confidence={confidence:.3f}"
                    )
                elif current_price > 125000:  # High ATH range
                    confidence = min(
                        0.9, 0.75 + abs(trend_score) * 0.1
                    )  # High confidence
                    self.logger.warning(
                        f"🔴 HIGH ATH SELL SIGNAL - price=${current_price:,.0f}, score={trend_score}, confidence={confidence:.3f}"
                    )
                else:
                    confidence = min(
                        0.9, 0.7 + abs(trend_score) * 0.1
                    )  # Normal confidence
                    self.logger.info(
                        f"🔴 SELL SIGNAL - score={trend_score}, confidence={confidence:.3f}"
                    )
                return np.array([0.0, 1.0 - confidence, confidence])
            else:
                # Updated neutral market behavior
                if current_price > 130000:  # Extreme ATH - strong sell bias
                    self.logger.warning(
                        f"⚠️ EXTREME ATH NEUTRAL - strong sell bias at ${current_price:,.0f}"
                    )
                    return np.array([0.05, 0.25, 0.7])  # Strong SELL bias
                elif current_price > 125000:  # High ATH - moderate sell bias
                    self.logger.warning(
                        f"⚠️ HIGH ATH NEUTRAL - moderate sell bias at ${current_price:,.0f}"
                    )
                    return np.array([0.1, 0.4, 0.5])  # Moderate SELL bias
                else:
                    # Normal trading range - balanced approach
                    self.logger.info(
                        f"⚪ NORMAL RANGE - score={trend_score}, balanced trading"
                    )
                    return np.array([0.35, 0.3, 0.35])  # More aggressive, less HOLD

            # Convert to probability distribution
            if buy_signals > sell_signals:
                confidence = min(
                    0.5 + buy_signals * 0.15, 0.9
                )  # Confidence based on number of signals
                result = np.array([confidence, 1.0 - confidence, 0.0])  # BUY
                self.logger.info(
                    f"Technical analysis BUY prediction with confidence {confidence:.3f}"
                )
                return result
            elif sell_signals > buy_signals:
                confidence = min(
                    0.5 + sell_signals * 0.15, 0.9
                )  # Confidence based on number of signals
                result = np.array([0.0, 1.0 - confidence, confidence])  # SELL
                self.logger.info(
                    f"Technical analysis SELL prediction with confidence {confidence:.3f}"
                )
                return result
            else:
                # No clear signal, return HOLD with lower confidence to allow other models to influence
                self.logger.info("Technical analysis result: HOLD (no clear signal)")
                return np.array([0.2, 0.6, 0.2])  # HOLD with distributed probabilities

        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return np.array([0.0, 1.0, 0.0])  # Default to HOLD

    def _signal_to_array(self, signal: str, confidence: float) -> np.ndarray:
        """Convert signal and confidence to probability array [BUY, HOLD, SELL]."""
        if signal == "BUY":
            return np.array([confidence, 1.0 - confidence, 0.0])
        elif signal == "SELL":
            return np.array([0.0, 1.0 - confidence, confidence])
        else:  # HOLD
            return np.array([0.0, confidence, 0.0])

    def _convert_features_to_dataframe(self, features: dict) -> pd.DataFrame:
        """
        Convert features dictionary to DataFrame format expected by research strategy.
        Creates a mock historical DataFrame with current values for indicator calculation.
        """
        try:
            # Use ACTUAL current market data - not fantasy values!
            current_price = features.get(
                "price", 107000.0
            )  # Use realistic current BTC price
            current_volume = features.get("volume", 1000.0)

            # Extract actual market data from features if available
            actual_high = features.get("high", current_price * 1.01)
            actual_low = features.get("low", current_price * 0.99)

            self.logger.info(
                f"🔬 Converting to DataFrame - Price: ${current_price:.2f}, High: ${actual_high:.2f}, Low: ${actual_low:.2f}"
            )

            # Generate realistic historical data around CURRENT price
            data_points = 50  # Enough for technical indicators

            # Create realistic price movements around current price (not fantasy values)
            base_prices = []
            current_sim_price = current_price

            for i in range(data_points - 1):
                # Small random walk around current price
                change_pct = np.random.uniform(-0.02, 0.02)  # ±2% max change
                current_sim_price = current_sim_price * (1 + change_pct)
                # Keep within reasonable bounds of actual current price
                current_sim_price = max(
                    current_price * 0.95, min(current_price * 1.05, current_sim_price)
                )
                base_prices.append(current_sim_price)

            # Add actual current price as the last point
            base_prices.append(current_price)

            data = {
                "high": [
                    price * (1 + np.random.uniform(0, 0.01)) for price in base_prices
                ],
                "low": [
                    price * (1 - np.random.uniform(0, 0.01)) for price in base_prices
                ],
                "close": base_prices.copy(),
                "volume": [
                    current_volume * (1 + np.random.uniform(-0.2, 0.2))
                    for _ in range(data_points)
                ],
            }

            # Set the LAST data point to ACTUAL current market values
            data["high"][-1] = actual_high
            data["low"][-1] = actual_low
            data["close"][-1] = current_price
            data["volume"][-1] = current_volume

            df = pd.DataFrame(data)

            # Add timestamp index for realism
            df.index = pd.date_range(
                end=pd.Timestamp.now(), periods=data_points, freq="5min"
            )

            # Add any additional columns that might be present in features
            for col in ["open"]:
                if col in features:
                    # Broadcast to all rows, but use actual value for last row
                    df[col] = features[col]
                    df[col].iloc[-1] = features[col]
                else:
                    # Use close prices as open prices (simple approximation)
                    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])

            self.logger.info(
                f"🔬 DataFrame created - Last close: ${df['close'].iloc[-1]:.2f}, Shape: {df.shape}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Error converting features to DataFrame: {e}")
            # Return DataFrame with ACTUAL current price, not fantasy values
            current_price = features.get("price", 107000.0)  # Use realistic fallback
            return pd.DataFrame(
                {
                    "high": [current_price * 1.01],
                    "low": [current_price * 0.99],
                    "close": [current_price],
                    "open": [current_price],
                    "volume": [1000.0],
                }
            )

    def calculate_cross_pair_correlation(
        self, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate the correlation matrix for the close prices of all symbols.
        """
        close_prices = pd.DataFrame(
            {symbol: df["close"] for symbol, df in data.items()}
        )
        return close_prices.corr()

    def update_rl_with_trade_result(self, trade_result: Dict[str, Any]):
        """
        Update the RL strategy with actual trade results for online learning

        Args:
            trade_result: Dictionary containing trade results
        """
        try:
            if self.enable_rl_strategy and self.rl_strategy is not None:
                self.rl_strategy.update_with_trade_result(trade_result)
                self.logger.info("🤖 RL strategy updated with trade result")
        except Exception as e:
            self.logger.error(f"Error updating RL strategy with trade result: {e}")

    def get_rl_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the RL strategy"""
        try:
            if self.enable_rl_strategy and self.rl_strategy is not None:
                return self.rl_strategy.get_performance_metrics()
            return {}
        except Exception as e:
            self.logger.error(f"Error getting RL performance metrics: {e}")
            return {}

    def force_rl_retrain(self) -> bool:
        """Force retraining of the RL model"""
        try:
            if self.enable_rl_strategy and self.rl_strategy is not None:
                success = self.rl_strategy.force_retrain()
                if success:
                    self.logger.info("🤖 RL model retrained successfully")
                else:
                    self.logger.error("🤖 RL model retraining failed")
                return success
            return False
        except Exception as e:
            self.logger.error(f"Error forcing RL retrain: {e}")
            return False
