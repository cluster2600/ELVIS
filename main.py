# isort: skip_file  (import order is load-bearing: the libomp guard below MUST
# run before any ML import — do not let isort reorder this file)
import argparse
import logging
import math
import threading
import signal
import sys
import time
import curses
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal

# ponytail: torch, sklearn and skimage each bundle their own libomp.dylib; loading
# them together segfaults during RL training on macOS. Must be set before any
# ML import. Proper fix: single OpenMP runtime in the venv.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load environment variables first
from dotenv import load_dotenv

load_dotenv()

# Set Vault environment variables only if not already set
if not os.getenv("VAULT_ADDR"):
    os.environ["VAULT_ADDR"] = "http://127.0.0.1:8200"
if not os.getenv("VAULT_TOKEN"):
    # Vault token must be supplied externally when Vault-backed secret loading is used.
    logging.getLogger(__name__).warning(
        "VAULT_TOKEN is not set; Vault authentication will be unavailable."
    )

from core.bootstrap import bootstrap_application
from core.di import container
from core.events import event_bus, SystemEvent
from utils.console_dashboard import ConsoleDashboard
from trading.utils.trade_history_api import app as trade_history_app
from trading.cooldown.trade_cooldown_manager import TradeCooldownManager
from trading.analysis.technical_indicators import add_technical_indicators
from trading.application.order_service import OrderService
from trading.application.rsi_gate_policy import RsiGatePolicy
from trading.application.signal_policy import SignalPolicyPipeline
from trading.data.market_frames import enrich_symbol_frames
from trading.domain.orders import OrderIntent, OrderSide, OrderType, SubmissionReport
from trading.domain.signals import Signal, SignalAction
from trading.execution.legacy_paper_adapter import LegacyPaperExecutionAdapter
import pandas as pd


def start_trade_history_server():
    """
    Start the Trade History Flask server in a separate thread.
    """
    # Secure by default: expose API only locally unless explicitly overridden.
    api_host = os.getenv("TRADE_HISTORY_API_HOST", "127.0.0.1")
    api_port = int(os.getenv("TRADE_HISTORY_API_PORT", "5050"))
    trade_history_app.run(host=api_host, port=api_port)


# Global shutdown flag
shutdown_requested = False
trading_thread = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully - close all positions and exit."""
    global shutdown_requested
    logger = container.get_optional("logger", logging.getLogger(__name__))
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")

    # Set shutdown flag instead of immediate exit
    shutdown_requested = True

    # 🚨 CRITICAL: Close all open positions before shutdown
    try:
        logger.info("🔄 Closing all open positions before shutdown...")

        # Get the executor from DI container
        exchange_manager = container.get_optional("exchange_manager")
        if exchange_manager and hasattr(exchange_manager, "executors"):
            for exchange_name, executor in exchange_manager.executors.items():
                if hasattr(executor, "close_all_positions"):
                    logger.info(f"🔄 Closing positions on {exchange_name}...")
                    results = executor.close_all_positions(
                        reason=f"Shutdown signal {signum}"
                    )

                    if results["closed"]:
                        logger.info(
                            f"✅ {exchange_name}: Closed {len(results['closed'])} positions, P&L: ${results['total_pnl']:.2f}"
                        )
                    if results["errors"]:
                        logger.warning(
                            f"⚠️ {exchange_name}: {len(results['errors'])} errors occurred"
                        )
        else:
            logger.warning("⚠️ No executors found - positions may remain open")

        # 🔄 RESET P&L: Create a new trading session reset point
        try:
            logger.info("🔄 Resetting P&L for next session...")
            from utils.paper_trade_db import reset_trading_session

            reset_trading_session()
            logger.info("✅ P&L reset complete - next session will start fresh")
        except Exception as e:
            logger.warning(f"⚠️ Could not reset P&L: {e}")

    except Exception as e:
        logger.error(f"❌ Error closing positions during shutdown: {e}")
        logger.error("⚠️ Some positions may remain open!")

    # Publish shutdown event
    event_bus.publish(
        SystemEvent(
            system_type="shutdown",
            component="main",
            status="ok",
            message=f"Shutdown signal {signum} received - positions closed",
            source="signal_handler",
        )
    )

    # Wait for trading thread to finish if it exists
    if trading_thread and trading_thread.is_alive():
        logger.info("⏳ Waiting for trading operations to complete...")
        trading_thread.join(
            timeout=10
        )  # Reduced timeout since positions are already closed

    logger.info("✅ Graceful shutdown complete - all positions closed")
    sys.exit(0)


def risk_management_loop(risk_manager, logger):
    """Continuously manage position risk in a separate thread."""
    while not shutdown_requested:
        try:
            risk_manager.manage_positions()
        except Exception as e:
            logger.error(f"Error in risk management loop: {e}")

        # Check for shutdown during sleep
        for i in range(5):
            if shutdown_requested:
                logger.info("Shutdown requested, exiting risk management loop...")
                return
            time.sleep(1)

    logger.info("Risk management loop finished due to shutdown request")


def _log_strategy_mode(strategy_mode: str, logger: logging.Logger):
    if strategy_mode == "research":
        logger.info("🔬 Research-based strategy active in paper mode")
        logger.info("📊 Binary classification: BUY/SELL only (no HOLD signals)")
        logger.info("🎯 Historical research metrics are not performance promises")
    elif strategy_mode == "balanced":
        logger.info("🎯 Balanced strategy active - EMERGENCY MODE")
        logger.info("⚠️ Reduced trading frequency: 2 trades/hour max")
        logger.info("💰 Higher profit targets: $1.00 per trade")
        logger.info("⏱️ Cooldown enforcement: 10 minutes between trades")


def _train_research_strategy_if_needed(
    active_strategy, price_fetcher, logger: logging.Logger
):
    if not hasattr(active_strategy, "is_trained") or active_strategy.is_trained:
        return

    logger.info("🧠 Research model not found or loaded. Initiating pre-training...")
    try:
        logger.info("Fetching historical data for initial training...")
        initial_data = price_fetcher.get_historical_klines("BTCUSDT", "5m", limit=2100)

        if (
            initial_data is not None
            and not initial_data.empty
            and len(initial_data) > 200
        ):
            logger.info(f"✅ Fetched {len(initial_data)} records for initial training.")
            active_strategy.train_model(initial_data)
        else:
            logger.error(
                "❌ Could not fetch sufficient initial data for training. Bot will not trade until trained."
            )
    except Exception as e:
        logger.error(f"❌ Error during initial model training: {e}", exc_info=True)


def _strategy_loop_sleep_seconds(
    active_strategy, default_seconds: float = 1.0
) -> float:
    """Resolve the per-iteration loop pace from the strategy.

    The research strategy documents a 5-minute trading frequency via
    ``trading_frequency_minutes``. When that attribute is set (and positive),
    pace the loop accordingly; otherwise fall back to ``default_seconds`` so
    strategies without an opinion keep the previous behaviour.
    """
    freq_minutes = getattr(active_strategy, "trading_frequency_minutes", None)
    try:
        if freq_minutes is not None and float(freq_minutes) > 0:
            return float(freq_minutes) * 60.0
    except TypeError, ValueError:
        pass
    return default_seconds


def _retrain_strategy_if_due(active_strategy, price_fetcher, logger: logging.Logger):
    """Retrain the strategy once if it reports it is due.

    Calls ``active_strategy.should_retrain()`` when that method exists. When it
    returns truthy, fresh historical data is fetched and handed to
    ``train_model`` (mirroring the initial-training path). Everything is guarded
    so a retrain failure is logged but never interrupts the trading loop.
    """
    should_retrain = getattr(active_strategy, "should_retrain", None)
    if not callable(should_retrain):
        return False
    try:
        if not should_retrain():
            return False
        logger.info("🔁 Strategy reports retraining is due. Retraining model...")
        training_data = price_fetcher.get_historical_klines("BTCUSDT", "5m", limit=2100)
        if (
            training_data is not None
            and not training_data.empty
            and len(training_data) > 200
        ):
            active_strategy.train_model(training_data)
            logger.info("✅ Strategy retrained successfully.")
            return True
        logger.warning(
            "⚠️ Skipping retrain: insufficient historical data fetched for training."
        )
        return False
    except Exception as e:  # non-fatal: never break the trading loop
        logger.error(f"⚠️ Non-fatal error during strategy retraining: {e}")
        return False


def _close_position_by_id(position, exit_price, realized_pnl, logger):
    """Close ONE open position by its id and book the realized PnL.

    Exits (stop-loss / take-profit / trailing) must NOT route through
    executor.place_order: that re-runs the open/netting logic, which — when
    it can't net cleanly — leaves the position open OR spawns a new opposite
    one. The position then re-triggers its exit every cycle (the take-profit
    that fired identically every 2s while 20 duplicate positions piled up).
    close_position locks the exact row and commits its realized trade and
    deletion together. A missing/already-closed row returns False, so later
    exit rules are short-circuited only after the close transaction commits.
    """
    from utils.paper_trade_db import close_position

    try:
        pos_id = position[0]
        qty = abs(float(position[4]))
        fee = abs(float(exit_price)) * qty * 0.0004
        return (
            close_position(
                pos_id,
                float(exit_price),
                float(realized_pnl),
                fee,
            )
            is True
        )
    except Exception as exc:
        logger.error(f"Close-by-id failed for position {position[0]}: {exc}")
        return False


def _protection_floor(mode: str, current_balance: float, baseline_state: dict):
    """Emergency-stop floor for the portfolio-protection check.

    Returns ``(floor, baseline, pct)``. Paper mode measures against the
    configured ``INITIAL_USDT_BALANCE`` deposit; live mode measures against
    the FIRST balance observed this session (persisted in
    ``baseline_state``) — real money must never be judged against the
    fictional paper deposit. ``ELVIS_PROTECT_FLOOR_PCT`` (default 0.7 = the
    original lose-more-than-30% rule) scales the floor in both modes.
    """
    from config.config import PAPER_TRADING_CONFIG

    pct = float(os.getenv("ELVIS_PROTECT_FLOOR_PCT", "0.7"))
    if mode == "paper":
        baseline = float(PAPER_TRADING_CONFIG.get("INITIAL_USDT_BALANCE", 100.0))
    else:
        baseline = baseline_state.setdefault("baseline", float(current_balance))
    return baseline * pct, baseline, pct


def _stop_loss_threshold(entry_price: float, quantity: float) -> float:
    """Negative USD stop-loss threshold as a percent of position notional.

    ELVIS_SL_PCT (default 0.005 = 0.5%) of entry*quantity. Replaces the
    FOURTH $1000-era absolute (a hardcoded -$15 that micro-notional
    positions could mathematically never hit, so losers sat open forever
    and the feedback loop starved of closed trades). A malformed env value
    falls back to the default instead of skipping the stop-loss check.
    """
    try:
        pct = float(os.getenv("ELVIS_SL_PCT", "0.005"))
    except ValueError:
        pct = 0.005
    return -abs(entry_price * quantity * pct)


def _record_model_votes(strategy, symbol, side, logger: logging.Logger):
    """Adaptive-ensemble feedback (#11): store each member's vote at entry.

    Guarded so a feedback failure never interrupts the trading loop.
    """
    if os.getenv("ELVIS_ADAPTIVE_ENSEMBLE", "1") != "1":
        return
    try:
        from trading.signals.model_feedback import record_entry

        votes = getattr(strategy, "last_model_votes", {}).get(symbol, {})
        if votes:
            record_entry(symbol, side, votes)
    except Exception as exc:
        logger.error(f"⚠️ Model-feedback record error: {exc}")


def _observe_rsi_policy_shadow(
    *,
    signal_symbol: str,
    legacy_action: str,
    legacy_confidence: float,
    reference_price: float,
    strategy_id: str,
    rsi: object,
    legacy_rsi_reason: str | None,
    logger: logging.Logger,
) -> None:
    """Compare the authoritative RSI gate with a side-effect-free candidate."""
    try:
        decision = Signal(
            decision_id=uuid.uuid4().hex,
            symbol=signal_symbol,
            action=SignalAction(legacy_action),
            confidence=legacy_confidence,
            reference_price=reference_price,
            observed_at=datetime.now(timezone.utc),
            strategy_id=strategy_id,
            reasons=("shadow.rsi.input",),
        )
        policy = RsiGatePolicy(rsi)
        candidate = SignalPolicyPipeline((policy,)).evaluate(decision)

        legacy_vetoed = legacy_rsi_reason is not None
        observed_legacy_action = (
            SignalAction.HOLD.value if legacy_vetoed else decision.action.value
        )
        observed_legacy_confidence = 0.0 if legacy_vetoed else decision.confidence
        candidate_reasons = candidate.reasons[len(decision.reasons) :]
        action_match = candidate.action.value == observed_legacy_action
        confidence_match = candidate.confidence == observed_legacy_confidence
        matched = action_match and confidence_match

        observed_rsi = None
        if isinstance(policy.rsi, (int, float)) and not isinstance(policy.rsi, bool):
            numeric_rsi = float(policy.rsi)
            if math.isfinite(numeric_rsi):
                observed_rsi = numeric_rsi

        extra = {
            "event_type": "signal_policy_shadow",
            "migration_slice": "M6b2",
            "migration_mode": "shadow",
            "stage": "roadmap_filters.rsi",
            "policy_id": policy.policy_id,
            "shadow_evaluation_id": decision.decision_id,
            "signal_symbol": decision.symbol,
            "strategy_id": decision.strategy_id,
            "rsi": observed_rsi,
            "legacy_action": observed_legacy_action,
            "legacy_confidence": observed_legacy_confidence,
            "candidate_action": candidate.action.value,
            "candidate_confidence": candidate.confidence,
            "legacy_reason": legacy_rsi_reason,
            "candidate_reasons": candidate_reasons,
            "action_match": action_match,
            "confidence_match": confidence_match,
            "matched": matched,
        }
        log = logger.info if matched else logger.warning
        log(
            "RSI policy shadow %s for %s: legacy=%s/%.3f candidate=%s/%.3f",
            "match" if matched else "divergence",
            decision.symbol,
            observed_legacy_action,
            observed_legacy_confidence,
            candidate.action.value,
            candidate.confidence,
            extra=extra,
        )
    except Exception as exc:
        try:
            logger.warning(
                "RSI policy shadow unavailable (%s)",
                type(exc).__name__,
                extra={
                    "event_type": "signal_policy_shadow_error",
                    "migration_slice": "M6b2",
                    "migration_mode": "shadow",
                    "stage": "roadmap_filters.rsi",
                    "policy_id": "rsi-gate",
                    "signal_symbol": signal_symbol,
                    "error_type": type(exc).__name__,
                },
            )
        except Exception:
            pass


def _observe_take_profit_regime_shadow(
    *,
    signal_symbol: str,
    quality_regime: object,
    candidate_regime: object,
    logger: logging.Logger,
) -> None:
    """Compare the legacy TP fallback with a side-effect-free regime candidate."""
    supported_regimes = {"TRENDING", "REVERSAL", "RANGING", "CHOPPY"}
    try:
        normalized_quality = quality_regime if isinstance(quality_regime, str) else None
        normalized_legacy = (
            normalized_quality.upper() if normalized_quality is not None else ""
        )
        legacy_effective_regime = (
            normalized_legacy if normalized_legacy in supported_regimes else "RANGING"
        )
        normalized_candidate = (
            candidate_regime
            if isinstance(candidate_regime, str)
            and candidate_regime in supported_regimes
            else None
        )
        candidate_available = normalized_candidate is not None
        matched = normalized_candidate == legacy_effective_regime
        extra = {
            "event_type": "take_profit_regime_shadow",
            "migration_slice": "M7g",
            "migration_mode": "shadow",
            "stage": "pretrade.take_profit_regime",
            "shadow_evaluation_id": uuid.uuid4().hex,
            "signal_symbol": signal_symbol,
            "quality_regime": normalized_quality,
            "legacy_effective_regime": legacy_effective_regime,
            "candidate_regime": normalized_candidate,
            "candidate_available": candidate_available,
            "matched": matched,
        }
    except Exception as exc:
        try:
            logger.warning(
                "Take-profit regime shadow unavailable (%s)",
                type(exc).__name__,
                extra={
                    "event_type": "take_profit_regime_shadow_error",
                    "migration_slice": "M7g",
                    "migration_mode": "shadow",
                    "stage": "pretrade.take_profit_regime",
                    "signal_symbol": signal_symbol,
                    "error_type": type(exc).__name__,
                },
            )
        except Exception:
            pass
        return

    try:
        log = logger.info if matched else logger.warning
        log(
            "Take-profit regime shadow %s for %s: legacy=%s candidate=%s",
            "match" if matched else "divergence",
            signal_symbol,
            legacy_effective_regime,
            normalized_candidate or "unavailable",
            extra=extra,
        )
    except Exception:
        pass


def _validated_active_fee_profile(candidate_regime: object) -> str | None:
    """Return a current produced TP profile or fail closed with ``None``."""
    if type(candidate_regime) is str and candidate_regime in {
        "TRENDING",
        "RANGING",
        "CHOPPY",
    }:
        return candidate_regime
    return None


def _legacy_order_intent(
    *,
    symbol: str,
    signal: str,
    confidence: float,
    current_price: float,
    position_size: float,
    leverage: int,
    strategy_id: str,
) -> OrderIntent:
    """Validate the legacy decision before it reaches the paper adapter."""
    decision_id = uuid.uuid4().hex
    observed_at = datetime.now(timezone.utc)
    decision = Signal(
        decision_id=decision_id,
        symbol=symbol,
        action=SignalAction(signal),
        confidence=confidence,
        reference_price=current_price,
        observed_at=observed_at,
        strategy_id=strategy_id,
        reasons=("legacy signal filters approved",),
    )
    return OrderIntent(
        client_order_id=f"ELV-{decision_id}",
        decision_id=decision.decision_id,
        symbol=decision.symbol,
        side=OrderSide(decision.action.value),
        quantity=Decimal(str(position_size)),
        order_type=OrderType.MARKET,
        reference_price=Decimal(str(decision.reference_price)),
        leverage=leverage,
        created_at=observed_at,
    )


def _record_acknowledged_legacy_order(
    report: SubmissionReport,
    intent: OrderIntent,
    *,
    confidence: float,
    strategy,
    cooldown_manager,
    logger: logging.Logger,
) -> bool:
    """Record legacy feedback once, and only after a proven acknowledgment."""
    if not report.acknowledged:
        return False
    if report.client_order_id != intent.client_order_id:
        logger.error("Refusing to record an acknowledgment for another order")
        return False

    _record_model_votes(strategy, intent.symbol, intent.side.value, logger)
    try:
        cooldown_manager.record_trade(
            intent.symbol,
            intent.side.value,
            float(intent.quantity),
            confidence,
        )
    except Exception as exc:
        logger.error(
            f"⚠️ Cooldown record failed after acknowledged order "
            f"{intent.client_order_id}: {type(exc).__name__}"
        )
    return True


def _score_model_votes(logger: logging.Logger):
    """Adaptive-ensemble feedback (#11): score votes of closed positions.

    Runs once per loop cycle; EMA-updates the persisted adaptive weights.
    """
    if os.getenv("ELVIS_ADAPTIVE_ENSEMBLE", "1") != "1":
        return
    try:
        from trading.signals.model_feedback import score_closed_trades

        scored = score_closed_trades()
        if scored:
            logger.info(f"🧠 Adaptive ensemble: scored {scored} model votes")
    except Exception as exc:
        logger.error(f"⚠️ Model-feedback scoring error: {exc}")


def main(mode: str, log_level: str):
    """
    Main entry point for the trading bot using dependency injection.

    Args:
        mode (str): Trading mode. Only 'paper' is currently executable.
        log_level (str): Logging level string.
    """
    if mode != "paper":
        raise RuntimeError(
            "ELVIS currently supports paper trading only; live execution is disabled"
        )

    # Bootstrap the application
    bootstrapper = bootstrap_application(mode, log_level)
    logger = container.get("logger")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start Trade History Server in background
        threading.Thread(target=start_trade_history_server, daemon=True).start()
        api_host = os.getenv("TRADE_HISTORY_API_HOST", "127.0.0.1")
        api_port = os.getenv("TRADE_HISTORY_API_PORT", "5050")
        logger.info(f"Started Trade History Server on {api_host}:{api_port}...")

        # Get dependencies from container
        strategy_manager = container.get("strategy_manager")
        risk_manager = container.get("risk_manager")
        price_fetcher = container.get("price_fetcher")
        executor = container.get("executor")
        legacy_execution = LegacyPaperExecutionAdapter(executor, runtime_mode=mode)
        order_service = OrderService(legacy_execution)
        llm_advisor = container.get("llm_advisor")

        # Get the active strategy (ensemble or research-based)
        active_strategy = container.get("strategy")
        strategy_mode = os.getenv("STRATEGY_MODE", "ensemble")
        logger.info(
            f"🎯 Active strategy: {type(active_strategy).__name__} (mode: {strategy_mode})"
        )

        _log_strategy_mode(strategy_mode, logger)
        if strategy_mode == "research":
            _train_research_strategy_if_needed(active_strategy, price_fetcher, logger)

        # Initialize the console dashboard
        price_fetcher = container.get("price_fetcher")
        dashboard = ConsoleDashboard(
            config={
                "portfolio_value": 10520.30,
                "unrealized_pnl": 150.10,
                "realized_pnl": 450.20,
                "open_positions": [],
                "recent_trades": [],
                "leverage": 100,  # Set 100x leverage for dashboard display
                "risk_manager": risk_manager,
                "performance_monitor": container.get("performance_monitor"),
                "trade_analyzer": container.get("trade_analyzer"),
                "system_monitor": container.get("system_monitor"),
            },
            logger=logger,
            price_fetcher=price_fetcher,
        )

        # Register dashboard in container for other components to use
        container.register_singleton("dashboard", lambda: dashboard)

        # Start risk management loop
        threading.Thread(
            target=risk_management_loop, args=(risk_manager, logger), daemon=True
        ).start()
        logger.info("Started Risk Management loop...")

        # Publish system ready event
        event_bus.publish(
            SystemEvent(
                system_type="startup",
                component="main",
                status="ok",
                message="Trading system started successfully",
                source="main",
                metrics={"mode": mode, "symbol": "BTCUSDT"},
            )
        )

        # Main trading loop
        logger.info(f"Starting trading bot in {mode} mode...")

        # Get executor for trade execution
        executor = container.get("executor")

        # The main loop will now be managed by the strategy manager
        def trading_loop():
            # Get executor for trade execution
            executor = container.get("executor")
            while not shutdown_requested:
                try:
                    # Get market data with detailed logging
                    logger.info("=== TRADING LOOP ITERATION START ===")

                    # Retrain hook: once per iteration, retrain if the strategy
                    # reports it is due (non-fatal, guarded inside the helper).
                    _retrain_strategy_if_due(active_strategy, price_fetcher, logger)

                    logger.info("Attempting to fetch market data...")
                    try:
                        # Multi-symbol trading: BTCUSDT + BNBUSDT (both USDT pairs for balance consistency)
                        from config.config import SYMBOLS_CONFIG

                        symbols_to_trade = SYMBOLS_CONFIG.get(
                            "PRIMARY_SYMBOLS", ["BTCUSDT", "BNBUSDT"]
                        )
                        all_data = {}

                        for trading_symbol in symbols_to_trade:
                            try:
                                symbol_data = price_fetcher.get_historical_klines(
                                    trading_symbol, "1m"
                                )
                                if not symbol_data.empty:
                                    all_data[trading_symbol] = symbol_data
                                    logger.debug(
                                        f"✅ {trading_symbol}: {len(symbol_data)} records"
                                    )
                                else:
                                    logger.warning(f"⚠️ No data for {trading_symbol}")
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Error fetching {trading_symbol}: {e}"
                                )

                        # Use BTCUSDT as primary for the dashboard.
                        data = all_data.get("BTCUSDT", pd.DataFrame())
                        logger.debug(
                            f"Fetched data shape: {data.shape if not data.empty else 'EMPTY'}"
                        )
                        if not data.empty:
                            logger.debug(f"Data columns: {list(data.columns)}")
                            # Check if 'close' column exists before accessing it
                            if "close" in data.columns:
                                logger.debug(
                                    f"Latest close price: {data.iloc[-1]['close']}"
                                )
                            else:
                                logger.warning(
                                    f"'close' column missing from data. Available columns: {list(data.columns)}"
                                )
                                # Try to use a different column or create mock data
                                if (
                                    len(data.columns) >= 5
                                ):  # Assume standard OHLCV order
                                    data.columns = [
                                        "open_time",
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "volume",
                                    ] + list(data.columns[6:])
                                    logger.debug(
                                        f"Reassigned column names. Latest close price: {data.iloc[-1]['close']}"
                                    )
                                else:
                                    logger.warning(
                                        "Insufficient columns, falling back to mock data"
                                    )
                                    data = pd.DataFrame()  # Force fallback to mock data
                        else:
                            # EMERGENCY: Get real Bitcoin price for ATH detection
                            logger.error(
                                "🚨 No market data available - getting real BTC price for emergency ATH detection"
                            )
                            try:
                                import requests

                                response = requests.get(
                                    "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                                    timeout=5,
                                )
                                if response.status_code == 200:
                                    real_price = float(response.json()["price"])
                                    logger.error(
                                        f"🚨 REAL BTC PRICE: ${real_price:,.2f}"
                                    )

                                    # EMERGENCY: If BTC is above $135k, STOP TRADING (Updated for new paradigm)
                                    if real_price > 135000:
                                        logger.error(
                                            f"🚨 EMERGENCY: BTC at ${real_price:,.2f} > $135k - EXTREME ATH!"
                                        )
                                        logger.error(
                                            "🚨 STOPPING ALL TRADING - MANUAL INTERVENTION REQUIRED"
                                        )
                                        return  # Exit trading loop

                                    # Use real price for data
                                    mock_data = {
                                        "open": [real_price] * 50,
                                        "high": [real_price * 1.001] * 50,
                                        "low": [real_price * 0.999] * 50,
                                        "close": [real_price] * 50,
                                        "volume": [1000] * 50,
                                    }
                                else:
                                    # Fallback with warning
                                    logger.error(
                                        "🚨 Could not get real price - using emergency mock data"
                                    )
                                    mock_data = {
                                        "open": [97000] * 50,
                                        "high": [97200] * 50,
                                        "low": [96800] * 50,
                                        "close": [97000] * 50,
                                        "volume": [1000] * 50,
                                    }
                            except Exception as price_error:
                                logger.error(
                                    f"🚨 Emergency price fetch failed: {price_error}"
                                )
                                mock_data = {
                                    "open": [97000] * 50,
                                    "high": [97200] * 50,
                                    "low": [96800] * 50,
                                    "close": [97000] * 50,
                                    "volume": [1000] * 50,
                                }

                            data = pd.DataFrame(mock_data)
                            logger.info(
                                f"Emergency data created with BTC price: ${data.iloc[-1]['close']:,.2f}"
                            )
                    except Exception as e:
                        logger.error(f"Error fetching market data: {e}")
                        # Fallback to mock data
                        import numpy as np

                        mock_data = {
                            "open": np.random.normal(97000, 200, 50),
                            "high": np.random.normal(97200, 200, 50),
                            "low": np.random.normal(96800, 200, 50),
                            "close": np.random.normal(97000, 200, 50),
                            "volume": np.random.normal(1000, 100, 50),
                        }
                        data = pd.DataFrame(mock_data)
                        logger.info("Using fallback mock data")
                        if "close" in data.columns:
                            logger.info(
                                f"Fallback data latest close: {data.iloc[-1]['close']:.2f}"
                            )
                        else:
                            logger.error(
                                "Fallback mock data missing 'close' column - this should not happen"
                            )

                    if not data.empty:
                        # Enrich every fetched symbol independently. Keep
                        # emergency fallback data outside all_data so it can
                        # inform the dashboard without becoming tradeable.
                        logger.debug(
                            "Adding technical indicators for each available symbol..."
                        )
                        all_data = enrich_symbol_frames(all_data, logger)
                        if "BTCUSDT" in all_data:
                            data = all_data["BTCUSDT"]
                        else:
                            data = add_technical_indicators(
                                data.copy(deep=True), logger
                            )
                        logger.debug(f"Data with indicators shape: {data.shape}")
                        logger.debug(f"Available columns: {list(data.columns)}")

                        # Update dashboard config with real OHLC data
                        try:
                            if "dashboard" in locals() and hasattr(dashboard, "config"):
                                # Ensure OHLC data is properly converted to numeric types
                                ohlc_subset = (
                                    data[["open", "high", "low", "close"]]
                                    .tail(40)
                                    .copy()
                                )

                                # Force conversion to numeric types and handle any conversion errors
                                for col in ["open", "high", "low", "close"]:
                                    ohlc_subset[col] = pd.to_numeric(
                                        ohlc_subset[col], errors="coerce"
                                    )

                                # Drop any rows with NaN values after conversion
                                ohlc_subset = ohlc_subset.dropna()

                                # Only update if we have valid data
                                if not ohlc_subset.empty:
                                    dashboard.config["ohlc_data"] = ohlc_subset
                                    dashboard.config["current_price"] = float(
                                        data.iloc[-1]["close"]
                                    )
                                    dashboard.config["indicators"] = {
                                        "rsi": (
                                            float(data.iloc[-1].get("rsi", 50.0))
                                            if pd.notna(data.iloc[-1].get("rsi"))
                                            else 50.0
                                        ),
                                        "macd": (
                                            float(data.iloc[-1].get("macd", 0.0))
                                            if pd.notna(data.iloc[-1].get("macd"))
                                            else 0.0
                                        ),
                                        "sma_20": (
                                            float(
                                                data.iloc[-1].get(
                                                    "sma_20", data.iloc[-1]["close"]
                                                )
                                            )
                                            if pd.notna(data.iloc[-1].get("sma_20"))
                                            else float(data.iloc[-1]["close"])
                                        ),
                                    }

                                    # Update open positions from paper trading database
                                    try:
                                        from utils.paper_trade_db import (
                                            get_open_positions,
                                        )

                                        open_positions_raw = get_open_positions()

                                        # Convert to dashboard format
                                        open_positions = []
                                        for pos in open_positions_raw:
                                            # Correct schema: (id, symbol, side, entry_price, quantity, leverage, entry_time)
                                            if len(pos) >= 7:
                                                try:
                                                    symbol = pos[1]
                                                    side = pos[2]
                                                    entry_price = float(pos[3])
                                                    quantity = float(pos[4])
                                                    leverage = int(pos[5])
                                                    entry_time = pos[6]

                                                    # Get current price for this symbol
                                                    try:
                                                        if price_fetcher:
                                                            current_price = price_fetcher.get_current_price(
                                                                symbol
                                                            )
                                                            if current_price is None:
                                                                # Use executor's mock price for proper P&L calculation
                                                                try:
                                                                    current_price = executor._get_mock_price(
                                                                        symbol
                                                                    )
                                                                    logger.debug(
                                                                        f"Dashboard using fallback price for {symbol}: ${current_price:.2f}"
                                                                    )
                                                                except:
                                                                    current_price = (
                                                                        entry_price
                                                                    )
                                                            else:
                                                                current_price = float(
                                                                    current_price
                                                                )
                                                        else:
                                                            # No price_fetcher, use executor's mock price
                                                            try:
                                                                current_price = executor._get_mock_price(
                                                                    symbol
                                                                )
                                                            except:
                                                                current_price = (
                                                                    entry_price
                                                                )
                                                    except Exception as price_err:
                                                        logger.warning(
                                                            f"Could not fetch current price for {symbol}: {price_err}"
                                                        )
                                                        # Fallback to executor's mock price
                                                        try:
                                                            current_price = executor._get_mock_price(
                                                                symbol
                                                            )
                                                        except:
                                                            current_price = entry_price

                                                    # Calculate comprehensive P&L
                                                    if hasattr(
                                                        executor,
                                                        "calculate_open_position_pnl",
                                                    ):
                                                        pnl_detail = executor.calculate_open_position_pnl(
                                                            symbol,
                                                            side,
                                                            current_price,
                                                            entry_price,
                                                            quantity,
                                                            leverage,
                                                            entry_time,
                                                        )
                                                        net_pnl = pnl_detail.get(
                                                            "net_pnl", 0.0
                                                        )
                                                        gross_pnl = pnl_detail.get(
                                                            "gross_pnl", 0.0
                                                        )
                                                        fees_info = {
                                                            "total_fees": pnl_detail.get(
                                                                "ongoing_costs", 0.0
                                                            )
                                                            + pnl_detail.get(
                                                                "estimated_exit_fee",
                                                                0.0,
                                                            ),
                                                            "funding_fee": pnl_detail.get(
                                                                "funding_fee", 0.0
                                                            ),
                                                            "borrowing_cost": pnl_detail.get(
                                                                "borrowing_cost", 0.0
                                                            ),
                                                            "hours_held": pnl_detail.get(
                                                                "hours_held", 0.0
                                                            ),
                                                        }
                                                    else:
                                                        # Fallback to simple calculation
                                                        pnl_factor = (
                                                            1
                                                            if side.upper() == "BUY"
                                                            else -1
                                                        )
                                                        net_pnl = (
                                                            (
                                                                current_price
                                                                - entry_price
                                                            )
                                                            * quantity
                                                            * pnl_factor
                                                        )
                                                        gross_pnl = net_pnl
                                                        fees_info = {"total_fees": 0}

                                                    open_positions.append(
                                                        {
                                                            "symbol": symbol,
                                                            "size": quantity,
                                                            "entry_price": entry_price,
                                                            "pnl": net_pnl,
                                                            "gross_pnl": gross_pnl,
                                                            "fees_info": fees_info,
                                                            "leverage": leverage,
                                                            "entry_time": entry_time,
                                                        }
                                                    )
                                                except (
                                                    ValueError,
                                                    TypeError,
                                                    IndexError,
                                                ) as e:
                                                    logger.error(
                                                        f"Error processing position tuple {pos}: {e}"
                                                    )
                                                    continue
                                            else:
                                                logger.warning(
                                                    f"Skipping malformed position tuple: {pos}"
                                                )

                                        dashboard.config["open_positions"] = (
                                            open_positions
                                        )

                                        # Update portfolio value and PnL with live data
                                        try:
                                            # Get dynamic balance from executor
                                            balance_info = (
                                                executor.get_balance()
                                                if executor
                                                else {"USDT": 10000.0, "BTC": 0.0}
                                            )
                                            usdt_balance = balance_info.get(
                                                "USDT", 10000.0
                                            )
                                            btc_balance = balance_info.get("BTC", 0.0)

                                            # Calculate total portfolio value (USDT + BTC value in USDT)
                                            current_btc_price = (
                                                float(data.iloc[-1]["close"])
                                                if "close" in data.columns
                                                else 97000.0
                                            )
                                            btc_value_in_usdt = (
                                                btc_balance * current_btc_price
                                            )
                                            total_portfolio_value = (
                                                usdt_balance + btc_value_in_usdt
                                            )

                                            dashboard.config["portfolio_value"] = (
                                                total_portfolio_value
                                            )

                                            # Add real-time price data
                                            dashboard.config["current_btc_price"] = (
                                                current_btc_price
                                            )

                                            # Add real-time leverage from executor
                                            if executor and hasattr(
                                                executor, "default_leverage"
                                            ):
                                                dashboard.config["leverage"] = (
                                                    executor.default_leverage
                                                )
                                            else:
                                                dashboard.config["leverage"] = (
                                                    100  # Default to 100x leverage
                                                )

                                            # Calculate total unrealized PnL with live prices
                                            total_unrealized_pnl = sum(
                                                pos["pnl"] for pos in open_positions
                                            )
                                            dashboard.config["unrealized_pnl"] = (
                                                total_unrealized_pnl
                                            )

                                            # Calculate realized PnL from trade history (exclude TEST trades)
                                            try:
                                                from utils.paper_trade_db import (
                                                    get_all_trades,
                                                )

                                                all_trades = get_all_trades(
                                                    limit=1000, exclude_test=True
                                                )
                                                total_realized_pnl = sum(
                                                    float(trade[6])
                                                    for trade in all_trades
                                                    if len(trade) >= 7
                                                )
                                                dashboard.config["realized_pnl"] = (
                                                    total_realized_pnl
                                                )
                                            except Exception:
                                                dashboard.config["realized_pnl"] = 0.0

                                            # Log the updated values
                                            logger.info(
                                                f"Portfolio Update - USDT: ${usdt_balance:.2f}, BTC: {btc_balance:.6f}, "
                                                f"BTC Value: ${btc_value_in_usdt:.2f}, Total: ${total_portfolio_value:.2f}, "
                                                f"Unrealized PnL: ${total_unrealized_pnl:.2f}, Realized PnL: ${dashboard.config['realized_pnl']:.2f}"
                                            )

                                        except Exception as e:
                                            logger.warning(
                                                f"Could not update portfolio value: {e}"
                                            )
                                            # Fallback to static values
                                            dashboard.config["portfolio_value"] = (
                                                10000.0
                                            )

                                        # Update recent trades from database (exclude TEST trades)
                                        try:
                                            from utils.paper_trade_db import (
                                                get_all_trades,
                                            )

                                            recent_trades_raw = get_all_trades(
                                                limit=10, exclude_test=True
                                            )
                                            recent_trades = []
                                            for trade in recent_trades_raw:
                                                # trade is tuple: (id, timestamp, symbol, side, price, quantity, pnl, fee)
                                                if len(trade) >= 8:
                                                    recent_trades.append(
                                                        {
                                                            "symbol": trade[2],
                                                            "side": trade[3],
                                                            "price": float(trade[4]),
                                                            "quantity": float(trade[5]),
                                                            "pnl": float(trade[6]),
                                                            "timestamp": trade[1],
                                                        }
                                                    )
                                            dashboard.config["recent_trades"] = (
                                                recent_trades
                                            )
                                        except Exception as e:
                                            logger.debug(
                                                f"Could not update recent trades: {e}"
                                            )

                                        logger.debug(
                                            f"Updated dashboard with {len(open_positions)} open positions and {len(dashboard.config.get('recent_trades', []))} recent trades"
                                        )

                                    except Exception as e:
                                        logger.warning(
                                            f"Could not update open positions: {e}"
                                        )

                                    logger.debug(
                                        f"Updated dashboard with OHLC data types: {ohlc_subset.dtypes.to_dict()}"
                                    )
                                else:
                                    logger.warning(
                                        "OHLC data conversion resulted in empty DataFrame"
                                    )
                        except Exception as e:
                            logger.warning(f"Could not update dashboard config: {e}")

                        # Log key indicator values
                        latest = data.iloc[-1]
                        # Format indicator values safely
                        rsi_val = latest.get("rsi", "N/A")
                        macd_val = latest.get("macd", "N/A")
                        sma_val = latest.get("sma_20", "N/A")

                        rsi_str = (
                            f"{rsi_val:.2f}"
                            if isinstance(rsi_val, (int, float))
                            else str(rsi_val)
                        )
                        macd_str = (
                            f"{macd_val:.4f}"
                            if isinstance(macd_val, (int, float))
                            else str(macd_val)
                        )
                        sma_str = (
                            f"{sma_val:.2f}"
                            if isinstance(sma_val, (int, float))
                            else str(sma_val)
                        )

                        logger.info(
                            f"Latest indicators - RSI: {rsi_str}, MACD: {macd_str}, SMA: {sma_str}"
                        )

                        # 💰 AGGRESSIVE POSITION MANAGEMENT: Check positions EVERY iteration for profit-taking
                        try:
                            exchange_manager = container.get_optional(
                                "exchange_manager"
                            )
                            if exchange_manager and hasattr(
                                exchange_manager, "executors"
                            ):
                                for (
                                    exchange_name,
                                    executor,
                                ) in exchange_manager.executors.items():
                                    if hasattr(executor, "check_and_manage_positions"):
                                        executor.check_and_manage_positions()
                                        logger.debug(
                                            f"🔍 Aggressive position check completed for {exchange_name}"
                                        )
                        except Exception as pos_mgmt_e:
                            logger.debug(f"Position management error: {pos_mgmt_e}")

                        # 🤖 PERIODIC LLM MARKET ANALYSIS (every 10th iteration)
                        if hasattr(trading_loop, "iteration_count"):
                            trading_loop.iteration_count += 1
                        else:
                            trading_loop.iteration_count = 1

                        if llm_advisor and trading_loop.iteration_count % 10 == 0:
                            # market_data is populated per-symbol later in the loop;
                            # default to empty here so this periodic block can't
                            # NameError before any symbol has been processed.
                            market_data = {}
                            try:
                                sentiment_analysis = (
                                    llm_advisor.analyze_market_sentiment(market_data)
                                )
                                sentiment = sentiment_analysis.get(
                                    "sentiment", "Neutral"
                                )
                                sentiment_confidence = sentiment_analysis.get(
                                    "confidence", 0
                                )
                                risk_level = sentiment_analysis.get(
                                    "risk_level", "Medium"
                                )
                                analysis = sentiment_analysis.get("analysis", "")

                                logger.info(
                                    f"🧠 LLM Market Sentiment: {sentiment} ({sentiment_confidence:.1%} confidence)"
                                )
                                logger.info(f"📊 Risk Level: {risk_level}")
                                if analysis:
                                    logger.info(f"💡 LLM Analysis: {analysis[:100]}...")

                                # 🤖 Enhanced LLM Analysis - Performance & Trends
                                if (
                                    trading_loop.iteration_count % 20 == 0
                                ):  # Every 20 iterations for comprehensive analysis
                                    try:
                                        # Get recent trading data
                                        recent_trades = []
                                        portfolio_data = {}
                                        open_positions = []

                                        try:
                                            # Get recent trades from API
                                            import requests

                                            response = requests.get(
                                                "http://localhost:5050/trades",
                                                timeout=2,
                                            )
                                            if response.status_code == 200:
                                                recent_trades = response.json()[
                                                    -10:
                                                ]  # Last 10 trades

                                            # Get portfolio data
                                            balance_response = requests.get(
                                                "http://localhost:5050/balance",
                                                timeout=2,
                                            )
                                            if balance_response.status_code == 200:
                                                balance_data = balance_response.json()
                                                usdt_balance = balance_data.get(
                                                    "USDT", 1000
                                                )
                                                bnb_balance = balance_data.get(
                                                    "BNB", 1.67
                                                )
                                                btc_balance = balance_data.get(
                                                    "BTC", 0.0086
                                                )

                                                # Calculate portfolio value with current prices
                                                bnb_price = market_data.get(
                                                    "bnb_price", 600
                                                )
                                                btc_price = market_data.get(
                                                    "btc_price", 117000
                                                )
                                                total_value = (
                                                    usdt_balance
                                                    + (bnb_balance * bnb_price)
                                                    + (btc_balance * btc_price)
                                                )

                                                portfolio_data = {
                                                    "total_value": total_value,
                                                    "realized_pnl": sum(
                                                        float(t.get("pnl", 0))
                                                        for t in recent_trades
                                                    ),
                                                    "unrealized_pnl": 0,  # Will be calculated from open positions
                                                }

                                            # Get open positions
                                            positions_response = requests.get(
                                                "http://localhost:5050/open_positions",
                                                timeout=2,
                                            )
                                            if positions_response.status_code == 200:
                                                open_positions = (
                                                    positions_response.json()
                                                )

                                        except Exception as data_e:
                                            logger.debug(
                                                f"Could not fetch trading data for LLM analysis: {data_e}"
                                            )

                                        # 📊 Trading Performance Analysis
                                        if recent_trades:
                                            performance_analysis = (
                                                llm_advisor.analyze_trading_performance(
                                                    recent_trades,
                                                    portfolio_data,
                                                    open_positions,
                                                )
                                            )

                                            perf_score = performance_analysis.get(
                                                "performance_score", "N/A"
                                            )
                                            win_rate = performance_analysis.get(
                                                "win_rate", "N/A"
                                            )
                                            risk_assessment = performance_analysis.get(
                                                "risk_assessment", "UNKNOWN"
                                            )

                                            logger.info(
                                                f"🎯 LLM Performance Analysis - Score: {perf_score}/10, Win Rate: {win_rate}%, Risk: {risk_assessment}"
                                            )

                                            # Log key recommendations
                                            recommendations = performance_analysis.get(
                                                "recommendations", []
                                            )
                                            for i, rec in enumerate(
                                                recommendations[:3], 1
                                            ):
                                                logger.info(
                                                    f"💡 LLM Recommendation {i}: {rec}"
                                                )

                                        # 📈 Market Trend Analysis
                                        if (
                                            hasattr(data, "index") and len(data) > 10
                                        ):  # Ensure we have enough price data
                                            try:
                                                trend_analysis = (
                                                    llm_advisor.analyze_market_trends(
                                                        data
                                                    )
                                                )

                                                trend = trend_analysis.get(
                                                    "trend", "UNKNOWN"
                                                )
                                                momentum = trend_analysis.get(
                                                    "momentum", "UNKNOWN"
                                                )
                                                volatility = trend_analysis.get(
                                                    "volatility", "N/A"
                                                )

                                                logger.info(
                                                    f"📈 LLM Trend Analysis - Trend: {trend}, Momentum: {momentum}, Volatility: {volatility}%"
                                                )

                                                # 🔮 Generate Comprehensive Trading Insights
                                                if (
                                                    recent_trades
                                                    and trend_analysis.get("analysis")
                                                ):
                                                    insights_report = llm_advisor.generate_trading_insights(
                                                        performance_analysis,
                                                        trend_analysis,
                                                    )

                                                    # Log the insights report (truncated for console)
                                                    insights_lines = (
                                                        insights_report.split("\n")
                                                    )
                                                    logger.info(
                                                        "🤖 LLM COMPREHENSIVE TRADING INSIGHTS:"
                                                    )
                                                    for line in insights_lines[
                                                        :5
                                                    ]:  # Show first 5 lines
                                                        if line.strip():
                                                            logger.info(
                                                                f"   {line.strip()}"
                                                            )

                                                    # Store insights for dashboard display
                                                    if (
                                                        "dashboard" in locals()
                                                        and hasattr(dashboard, "config")
                                                    ):
                                                        dashboard.config[
                                                            "llm_insights"
                                                        ] = {
                                                            "performance": performance_analysis,
                                                            "trends": trend_analysis,
                                                            "report": insights_report,
                                                            "timestamp": datetime.now(),
                                                        }

                                            except Exception as trend_e:
                                                logger.debug(
                                                    f"LLM trend analysis error: {trend_e}"
                                                )

                                    except Exception as analysis_e:
                                        logger.warning(
                                            f"LLM comprehensive analysis error: {analysis_e}"
                                        )
                                        logger.debug(
                                            "LLM analysis error details:", exc_info=True
                                        )

                                # Store for dashboard
                                if "dashboard" in locals() and hasattr(
                                    dashboard, "config"
                                ):
                                    dashboard.config["llm_sentiment"] = {
                                        "sentiment": sentiment,
                                        "confidence": sentiment_confidence,
                                        "risk_level": risk_level,
                                        "analysis": analysis,
                                        "timestamp": sentiment_analysis.get(
                                            "timestamp"
                                        ),
                                    }

                            except Exception as e:
                                logger.debug(f"LLM sentiment analysis failed: {e}")

                        # BALANCED STARTER STRATEGY - Only execute if not main strategy.
                        # Gated OFF by default: it trades independently with a
                        # hardcoded $1000 notional and its own bookkeeping,
                        # bypassing the winrate filter, roadmap gates, Kelly
                        # cap and cooldowns entirely.
                        if (
                            strategy_mode != "balanced"
                            and os.getenv("ELVIS_BALANCED_STARTER", "0") == "1"
                        ):
                            try:
                                from trading.strategies.balanced_starter import (
                                    BalancedStarterStrategy,
                                )

                                # Initialize balanced starter if not exists
                                if not hasattr(
                                    container.get("strategy"), "balanced_starter"
                                ):
                                    container.get("strategy").balanced_starter = (
                                        BalancedStarterStrategy(logger)
                                    )
                                    logger.info(
                                        "🎯 INITIALIZED BALANCED STARTER STRATEGY"
                                    )

                                # Execute balanced starter strategy
                                balanced_result = container.get(
                                    "strategy"
                                ).balanced_starter.execute_strategy(
                                    data, executor.get_account_balance()
                                )

                                if balanced_result["action"] in [
                                    "INITIALIZED",
                                    "ADAPTED",
                                ]:
                                    logger.info(
                                        f"🎯 BALANCED STARTER: {balanced_result['action']} at ${balanced_result.get('price', 0):,.2f}"
                                    )
                                    if "market_bias" in balanced_result:
                                        logger.info(
                                            f"📊 Market Bias: {balanced_result['market_bias']:.3f}"
                                        )
                                    if "positions" in balanced_result:
                                        pos = balanced_result["positions"]
                                        logger.info(
                                            f"📊 Position Distribution: {pos['long_count']} LONG, {pos['short_count']} SHORT"
                                        )

                            except Exception as e:
                                logger.error(f"Error in balanced starter strategy: {e}")
                        else:
                            logger.info(
                                "🎯 BALANCED STARTER: Running as main strategy, skipping secondary execution"
                            )

                        # EMERGENCY PORTFOLIO PROTECTION - Check balance before trading
                        try:
                            current_balance = executor.get_account_balance()
                            # Mode-aware floor (see _protection_floor): paper
                            # measures against the configured deposit — the old
                            # hardcoded $700 floor from the $1000 world
                            # emergency-stopped every run once the honest $100
                            # deposit landed — while live measures against the
                            # session's first observed real balance.
                            if not hasattr(trading_loop, "_protection_state"):
                                trading_loop._protection_state = {}
                            _protect_floor, _baseline, _pct = _protection_floor(
                                mode, current_balance, trading_loop._protection_state
                            )
                            if current_balance < _protect_floor:
                                logger.error(
                                    f"🚨 PORTFOLIO PROTECTION: Balance dropped to ${current_balance:.2f}"
                                    f" (floor ${_protect_floor:.2f} = {_pct:.0%}"
                                    f" of ${_baseline:.2f} {mode} baseline)"
                                )
                                logger.error(
                                    "🚨 EMERGENCY SHUTDOWN - Portfolio protection activated"
                                )
                                logger.error(
                                    "🚨 Manual intervention required - losses too high"
                                )
                                return  # Exit trading loop
                        except Exception as balance_error:
                            logger.error(f"Could not check balance: {balance_error}")

                        # Initialize trade cooldown manager for fewer, bigger trades strategy
                        if not hasattr(trading_loop, "cooldown_manager"):
                            trading_loop.cooldown_manager = TradeCooldownManager(
                                logger, cooldown_minutes=15
                            )
                            logger.info(
                                "🕐 Trade Cooldown Manager initialized for FEWER, BIGGER trades"
                            )

                        # Use the active strategy (ensemble or research-based)
                        logger.info(f"Using strategy: {type(active_strategy).__name__}")

                        # Generate signals using the NEW generate_signal method (singular) with anti-HOLD logic
                        if hasattr(active_strategy, "generate_signal"):
                            # Get current market data for signal generation with error handling
                            try:
                                current_price = float(data.iloc[-1]["close"])
                                logger.debug(
                                    f"Current price extracted: {current_price} (type: {type(current_price)})"
                                )

                                market_data = {
                                    "close": current_price,
                                    "price": current_price,
                                    "high": float(
                                        data.iloc[-1].get("high", current_price)
                                    ),
                                    "low": float(
                                        data.iloc[-1].get("low", current_price)
                                    ),
                                    "volume": float(data.iloc[-1].get("volume", 1000)),
                                    "rsi": (
                                        float(data.iloc[-1].get("rsi", 50.0))
                                        if pd.notna(data.iloc[-1].get("rsi"))
                                        else 50.0
                                    ),
                                    "macd": (
                                        float(data.iloc[-1].get("macd", 0.0))
                                        if pd.notna(data.iloc[-1].get("macd"))
                                        else 0.0
                                    ),
                                    "macd_signal": (
                                        float(data.iloc[-1].get("signal_line", 0.0))
                                        if pd.notna(data.iloc[-1].get("signal_line"))
                                        else 0.0
                                    ),
                                    "sma_20": (
                                        float(
                                            data.iloc[-1].get("sma_20", current_price)
                                        )
                                        if pd.notna(data.iloc[-1].get("sma_20"))
                                        else current_price
                                    ),
                                    "sma_50": (
                                        float(
                                            data.iloc[-1].get("sma_50", current_price)
                                        )
                                        if pd.notna(data.iloc[-1].get("sma_50"))
                                        else current_price
                                    ),
                                    "atr": (
                                        float(
                                            data.iloc[-1].get(
                                                "atr", current_price * 0.02
                                            )
                                        )
                                        if pd.notna(data.iloc[-1].get("atr"))
                                        else current_price * 0.02
                                    ),
                                    "adx": (
                                        float(data.iloc[-1].get("adx", 25.0))
                                        if pd.notna(data.iloc[-1].get("adx"))
                                        else 25.0
                                    ),
                                    "bb_upper": (
                                        float(
                                            data.iloc[-1].get(
                                                "upper_bb", current_price * 1.02
                                            )
                                        )
                                        if pd.notna(data.iloc[-1].get("upper_bb"))
                                        else current_price * 1.02
                                    ),
                                    "bb_lower": (
                                        float(
                                            data.iloc[-1].get(
                                                "lower_bb", current_price * 0.98
                                            )
                                        )
                                        if pd.notna(data.iloc[-1].get("lower_bb"))
                                        else current_price * 0.98
                                    ),
                                    "bb_middle": (
                                        float(
                                            data.iloc[-1].get("sma_bb", current_price)
                                        )
                                        if pd.notna(data.iloc[-1].get("sma_bb"))
                                        else current_price
                                    ),
                                }
                            except Exception as e:
                                logger.error(f"❌ Error processing market data: {e}")
                                logger.error(
                                    f"   Data types - close: {type(data.iloc[-1]['close'])}, value: {data.iloc[-1]['close']}"
                                )
                                continue

                            # Process ALL symbols from symbols_to_trade
                            logger.info(
                                f"🔄 Processing {len(symbols_to_trade)} trading symbols: {symbols_to_trade}"
                            )

                            for symbol in symbols_to_trade:
                                try:
                                    # Get data for this specific symbol
                                    symbol_data = all_data.get(symbol)
                                    if symbol_data is None or symbol_data.empty:
                                        logger.warning(
                                            f"⚠️ No data available for {symbol}, skipping"
                                        )
                                        continue

                                    symbol_history = symbol_data.tail(100).copy(
                                        deep=True
                                    )
                                    filter_result = None
                                    regime_result = None
                                    take_profit_regime = None

                                    # Extract current price and market data for this symbol
                                    current_price = float(symbol_data.iloc[-1]["close"])

                                    # Preserve the per-symbol observation used
                                    # by the legacy gate before the strategy's
                                    # neutral fallback is applied.
                                    _raw_filter_rsi = symbol_data.iloc[-1].get("rsi")
                                    _filter_rsi = (
                                        float(_raw_filter_rsi)
                                        if pd.notna(_raw_filter_rsi)
                                        else None
                                    )

                                    # Create market data specific to this symbol
                                    market_data = {
                                        "close": current_price,
                                        "price": current_price,
                                        "high": float(
                                            symbol_data.iloc[-1].get(
                                                "high", current_price
                                            )
                                        ),
                                        "low": float(
                                            symbol_data.iloc[-1].get(
                                                "low", current_price
                                            )
                                        ),
                                        "volume": float(
                                            symbol_data.iloc[-1].get("volume", 1000)
                                        ),
                                        "rsi": (
                                            _filter_rsi
                                            if _filter_rsi is not None
                                            else 50.0
                                        ),
                                        "macd": (
                                            float(symbol_data.iloc[-1].get("macd", 0.0))
                                            if pd.notna(
                                                symbol_data.iloc[-1].get("macd")
                                            )
                                            else 0.0
                                        ),
                                        "macd_signal": (
                                            float(
                                                symbol_data.iloc[-1].get(
                                                    "signal_line", 0.0
                                                )
                                            )
                                            if pd.notna(
                                                symbol_data.iloc[-1].get("signal_line")
                                            )
                                            else 0.0
                                        ),
                                        "sma_20": (
                                            float(
                                                symbol_data.iloc[-1].get(
                                                    "sma_20", current_price
                                                )
                                            )
                                            if pd.notna(
                                                symbol_data.iloc[-1].get("sma_20")
                                            )
                                            else current_price
                                        ),
                                        "sma_50": (
                                            float(
                                                symbol_data.iloc[-1].get(
                                                    "sma_50", current_price
                                                )
                                            )
                                            if pd.notna(
                                                symbol_data.iloc[-1].get("sma_50")
                                            )
                                            else current_price
                                        ),
                                        "atr": (
                                            float(
                                                symbol_data.iloc[-1].get(
                                                    "atr", current_price * 0.02
                                                )
                                            )
                                            if pd.notna(symbol_data.iloc[-1].get("atr"))
                                            else current_price * 0.02
                                        ),
                                        "adx": (
                                            float(symbol_data.iloc[-1].get("adx", 25.0))
                                            if pd.notna(symbol_data.iloc[-1].get("adx"))
                                            else 25.0
                                        ),
                                        "bb_upper": (
                                            float(
                                                symbol_data.iloc[-1].get(
                                                    "upper_bb", current_price * 1.02
                                                )
                                            )
                                            if pd.notna(
                                                symbol_data.iloc[-1].get("upper_bb")
                                            )
                                            else current_price * 1.02
                                        ),
                                        "bb_lower": (
                                            float(
                                                symbol_data.iloc[-1].get(
                                                    "lower_bb", current_price * 0.98
                                                )
                                            )
                                            if pd.notna(
                                                symbol_data.iloc[-1].get("lower_bb")
                                            )
                                            else current_price * 0.98
                                        ),
                                        "bb_middle": (
                                            float(
                                                symbol_data.iloc[-1].get(
                                                    "sma_bb", current_price
                                                )
                                            )
                                            if pd.notna(
                                                symbol_data.iloc[-1].get("sma_bb")
                                            )
                                            else current_price
                                        ),
                                    }

                                    logger.info(
                                        f"🎯 Processing {symbol}: Price=${current_price:.2f}, RSI={market_data['rsi']:.1f}"
                                    )

                                    # Generate signal for this symbol
                                    signal, confidence = (
                                        active_strategy.generate_signal(
                                            symbol, market_data
                                        )
                                    )

                                    logger.info(
                                        f"🎉 {symbol} RAW SIGNAL: {signal} with confidence {confidence:.3f}"
                                    )

                                    # 🎯 HIGH WIN RATE FILTERING SYSTEM
                                    if (
                                        signal in ["BUY", "SELL"]
                                        and confidence >= 0.6
                                        and os.getenv("ELVIS_WINRATE_FILTER", "1")
                                        == "1"
                                    ):
                                        try:
                                            from trading.analysis.high_winrate_filter import (
                                                HighWinRateFilter,
                                            )
                                            from trading.analysis.market_regime_detector import (
                                                MarketRegimeDetector,
                                            )

                                            # Initialize filters (cached for performance)
                                            if not hasattr(main, "_winrate_filter"):
                                                main._winrate_filter = (
                                                    HighWinRateFilter(logger)
                                                )
                                                main._regime_detector = (
                                                    MarketRegimeDetector(logger)
                                                )

                                            # Analyze signal quality
                                            filter_result = main._winrate_filter.analyze_signal_quality(
                                                symbol, market_data, symbol_history
                                            )
                                            candidate_take_profit_regime = None
                                            market_regime_result = filter_result.get(
                                                "market_regime"
                                            )
                                            if isinstance(market_regime_result, dict):
                                                candidate_take_profit_regime = (
                                                    market_regime_result.get(
                                                        "take_profit_regime"
                                                    )
                                                )

                                            # Analyze market regime
                                            regime_result = main._regime_detector.detect_current_regime(
                                                symbol_history
                                            )

                                            # Cache regime per symbol for the
                                            # dynamic-TP / fee-gate stages below
                                            if not hasattr(main, "_last_regime"):
                                                main._last_regime = {}
                                            main._last_regime[symbol] = regime_result[
                                                "regime"
                                            ]["class"]

                                            # Apply filtering logic
                                            original_signal = signal
                                            original_confidence = confidence

                                            if (
                                                filter_result["trade_approved"]
                                                and regime_result["regime"][
                                                    "should_trade"
                                                ]
                                            ):
                                                # HIGH QUALITY SIGNAL - Enhance confidence
                                                signal_quality = filter_result[
                                                    "confidence"
                                                ]
                                                regime_quality = regime_result[
                                                    "regime"
                                                ]["confidence"]

                                                # Combine confidences with weights
                                                enhanced_confidence = (
                                                    confidence * 0.4
                                                    + signal_quality * 0.4
                                                    + regime_quality * 0.2
                                                )
                                                confidence = min(
                                                    enhanced_confidence, 0.95
                                                )  # Cap at 95%

                                                logger.info(
                                                    f"✅ {symbol} SIGNAL APPROVED: {signal} | Enhanced Confidence: {confidence:.3f}"
                                                )
                                                logger.info(
                                                    f"   📊 Quality Score: {signal_quality:.2%} | Regime Score: {regime_quality:.2%}"
                                                )
                                                logger.info(
                                                    f"   🎯 Regime: {regime_result['regime']['class']} | Action: {regime_result['trading_recommendation']['action']}"
                                                )

                                            else:
                                                # REJECT SIGNAL - Set to HOLD
                                                rejection_reason = filter_result.get(
                                                    "rejection_reason", "Unknown"
                                                )
                                                if not regime_result["regime"][
                                                    "should_trade"
                                                ]:
                                                    rejection_reason = f"Unfavorable regime: {regime_result['regime']['class']}"

                                                signal = "HOLD"
                                                confidence = 0.3

                                                logger.warning(
                                                    f"❌ {symbol} SIGNAL REJECTED: {original_signal} -> HOLD"
                                                )
                                                logger.warning(
                                                    f"   🚫 Reason: {rejection_reason}"
                                                )
                                                logger.warning(
                                                    f"   📉 Quality: {filter_result['confidence']:.2%} | Regime: {regime_result['regime']['class']}"
                                                )

                                            # Publish the candidate only after
                                            # the complete analysis/filtering
                                            # block succeeds.
                                            take_profit_regime = (
                                                candidate_take_profit_regime
                                            )
                                            if (
                                                os.getenv(
                                                    "ELVIS_TP_REGIME_MODE", "legacy"
                                                )
                                                == "shadow"
                                            ):
                                                _observe_take_profit_regime_shadow(
                                                    signal_symbol=symbol,
                                                    quality_regime=regime_result[
                                                        "regime"
                                                    ]["class"],
                                                    candidate_regime=take_profit_regime,
                                                    logger=logger,
                                                )

                                        except Exception as filter_error:
                                            logger.error(
                                                f"⚠️ High win rate filter error: {filter_error}"
                                            )
                                            logger.info(
                                                f"Continuing with original signal: {signal}"
                                            )

                                    # 🧰 PROFITABILITY-ROADMAP SIGNAL GATES
                                    # (RSI/momentum/BB-squeeze/hours/MACD, order
                                    # flow, MTF — each env-toggled)
                                    if signal in ["BUY", "SELL"]:
                                        try:
                                            if (
                                                os.getenv("ELVIS_ROADMAP_FILTERS", "1")
                                                == "1"
                                            ):
                                                from trading.signals.filters import (
                                                    apply_signal_filters,
                                                )

                                                _roadmap_input_signal = signal
                                                _roadmap_input_confidence = confidence
                                                signal, confidence, _gate_reasons = (
                                                    apply_signal_filters(
                                                        signal,
                                                        confidence,
                                                        symbol_history,
                                                        rsi=_filter_rsi,
                                                    )
                                                )
                                                if (
                                                    os.getenv(
                                                        "ELVIS_RSI_POLICY_MODE",
                                                        "legacy",
                                                    )
                                                    == "shadow"
                                                ):
                                                    _legacy_rsi_reason = next(
                                                        (
                                                            reason
                                                            for reason in _gate_reasons
                                                            if reason.startswith(
                                                                "rsi_gate:"
                                                            )
                                                        ),
                                                        None,
                                                    )
                                                    _observe_rsi_policy_shadow(
                                                        signal_symbol=symbol,
                                                        legacy_action=_roadmap_input_signal,
                                                        legacy_confidence=_roadmap_input_confidence,
                                                        reference_price=current_price,
                                                        strategy_id=type(
                                                            active_strategy
                                                        ).__name__,
                                                        rsi=_filter_rsi,
                                                        legacy_rsi_reason=_legacy_rsi_reason,
                                                        logger=logger,
                                                    )
                                                for _reason in _gate_reasons:
                                                    logger.warning(
                                                        f"🚧 {symbol} filter: {_reason}"
                                                    )
                                            if (
                                                signal in ["BUY", "SELL"]
                                                and os.getenv("ELVIS_ORDER_FLOW", "0")
                                                == "1"
                                            ):
                                                from trading.signals.order_flow import (
                                                    confirm_signal_with_flow,
                                                )

                                                _flow_ok, _imbalance = (
                                                    confirm_signal_with_flow(
                                                        signal,
                                                        executor.get_order_book(symbol),
                                                    )
                                                )
                                                if not _flow_ok:
                                                    logger.warning(
                                                        f"🚧 {symbol} order flow contradicts {signal} "
                                                        f"(imbalance {_imbalance:+.2f}) -> HOLD"
                                                    )
                                                    signal, confidence = "HOLD", 0.0
                                            if (
                                                signal in ["BUY", "SELL"]
                                                and os.getenv("ELVIS_MTF", "0") == "1"
                                            ):
                                                from trading.signals.mtf import (
                                                    MTFAnalyzer,
                                                )

                                                if not hasattr(main, "_mtf"):
                                                    main._mtf = MTFAnalyzer(
                                                        price_fetcher.get_historical_klines
                                                    )
                                                _mtf_result = main._mtf.get_signal(
                                                    symbol
                                                )
                                                if _mtf_result["aligned"] != signal:
                                                    logger.warning(
                                                        f"🚧 {symbol} MTF misalignment {_mtf_result} -> HOLD"
                                                    )
                                                    signal, confidence = "HOLD", 0.0
                                        except Exception as gate_error:
                                            logger.error(
                                                f"⚠️ Roadmap signal gates error: {gate_error}"
                                            )

                                    # 📉 DIRECTIONAL PERFORMANCE GUARD — raise
                                    # the confidence bar for a side that's been
                                    # losing (learned from realized PnL; the
                                    # honest book showed SELL winning 3% of its
                                    # last 95 trades). Self-relaxing.
                                    if (
                                        signal in ["BUY", "SELL"]
                                        and os.getenv("ELVIS_DIRECTIONAL_GUARD", "1")
                                        == "1"
                                    ):
                                        try:
                                            from trading.signals.directional_guard import (
                                                required_confidence,
                                            )

                                            _floor = required_confidence(
                                                signal, symbol=symbol
                                            )
                                            if confidence < _floor:
                                                logger.warning(
                                                    f"📉 {symbol} {signal} throttled: "
                                                    f"confidence {confidence:.2f} < "
                                                    f"{_floor:.2f} bar (this side has "
                                                    f"been losing) -> HOLD"
                                                )
                                                signal, confidence = "HOLD", 0.0
                                        except Exception as dg_error:
                                            logger.error(
                                                f"⚠️ Directional guard error: {dg_error}"
                                            )

                                    logger.info(
                                        f"🎯 {symbol} FINAL SIGNAL: {signal} with confidence {confidence:.3f}"
                                    )

                                    # 🤖 LLM SIGNAL ENHANCEMENT
                                    if llm_advisor and signal in ["BUY", "SELL"]:
                                        try:
                                            llm_analysis = (
                                                llm_advisor.enhance_trading_signal(
                                                    signal, confidence, market_data
                                                )
                                            )

                                            # Apply LLM confidence adjustment
                                            original_confidence = confidence
                                            confidence = llm_analysis.get(
                                                "adjusted_confidence", confidence
                                            )
                                            validation = llm_analysis.get(
                                                "validation", "CAUTION"
                                            )

                                            logger.info(
                                                f"🧠 LLM Enhancement: {validation} | Confidence {original_confidence:.3f} → {confidence:.3f}"
                                            )

                                            if validation == "REJECT":
                                                logger.warning(
                                                    f"🚫 LLM REJECTED signal: {llm_analysis.get('reasoning', 'No reason given')}"
                                                )
                                                signal = "HOLD"
                                                confidence = 0.1
                                            elif validation == "CAUTION":
                                                logger.info(
                                                    f"⚠️ LLM CAUTION: {llm_analysis.get('recommendation', 'Proceed with caution')}"
                                                )
                                            elif validation == "CONFIRM":
                                                logger.info(
                                                    f"✅ LLM CONFIRMS signal: {llm_analysis.get('recommendation', 'Signal validated')}"
                                                )

                                            # Log risk factors
                                            risk_factors = llm_analysis.get(
                                                "risk_factors", []
                                            )
                                            if risk_factors:
                                                logger.info(
                                                    f"⚠️ LLM Risk Factors: {', '.join(risk_factors)}"
                                                )

                                        except Exception as e:
                                            logger.warning(
                                                f"LLM enhancement failed: {e}"
                                            )

                                    # EMERGENCY ATH FIX: Allow HOLD during extreme market conditions
                                    if signal == "HOLD":
                                        # During extreme ATH periods, HOLD is often the correct decision
                                        if (
                                            current_price > 130000
                                        ):  # Updated: Extreme ATH territory now at $130k
                                            logger.info(
                                                f"💎 HOLD signal during EXTREME ATH (${current_price:,.0f}) - SMART DECISION"
                                            )
                                        elif current_price > 125000:  # Normal ATH range
                                            logger.info(
                                                f"💎 HOLD signal in ATH range (${current_price:,.0f}) - Cautious approach"
                                            )
                                        else:
                                            logger.info(
                                                f"⏸️ HOLD signal - Waiting for better opportunity"
                                            )

                                    logger.info(
                                        f"Final signal for {symbol}: {signal} (confidence: {confidence:.3f})"
                                    )

                                    # Execute trades based on signals for this symbol
                                    if (
                                        signal in ["BUY", "SELL"] and confidence >= 0.60
                                    ):  # Lowered threshold to allow more trades
                                        logger.info(
                                            f"🎯 High confidence {symbol} signal: {signal} with {confidence:.3f} confidence"
                                        )

                                        try:
                                            # Get current account balance
                                            available_balance = (
                                                executor.get_account_balance()
                                            )
                                            logger.info(
                                                f"💰 Available balance: ${available_balance}"
                                            )

                                            # 🕐 CHECK TRADE COOLDOWN FIRST - FEWER TRADES STRATEGY
                                            cooldown_check = (
                                                trading_loop.cooldown_manager.can_trade(
                                                    symbol, max_daily_trades=8
                                                )
                                            )

                                            if not cooldown_check["can_trade"]:
                                                logger.warning(
                                                    f"🚫 COOLDOWN ACTIVE: {cooldown_check['reason']}"
                                                )
                                                if cooldown_check.get(
                                                    "minutes_remaining"
                                                ):
                                                    logger.warning(
                                                        f"   ⏰ Time remaining: {cooldown_check['minutes_remaining']:.1f} minutes"
                                                    )
                                                continue  # Skip this symbol due to cooldown

                                            logger.info(
                                                f"✅ COOLDOWN CHECK PASSED: {cooldown_check['daily_trades_remaining']} trades remaining today"
                                            )

                                            # 🎯 BIGGER TRADES STRATEGY - Larger positions, fewer trades
                                            base_risk_per_trade = 75.0  # Base $75 risk per trade (increased)

                                            # Scale position size based on confidence (85% = 0.5x, 95% = 2.0x)
                                            confidence_multiplier = 0.5 + (
                                                confidence - 0.85
                                            ) * (
                                                1.5 / 0.10
                                            )  # Steeper scale
                                            confidence_multiplier = max(
                                                0.5, min(2.0, confidence_multiplier)
                                            )  # Cap between 0.5x - 2.0x

                                            # Scale based on regime quality if filter was used
                                            regime_multiplier = 1.0
                                            if (
                                                filter_result is not None
                                                and regime_result is not None
                                            ):
                                                if (
                                                    regime_result[
                                                        "trading_recommendation"
                                                    ]["action"]
                                                    == "trade_aggressively"
                                                ):
                                                    regime_multiplier = 1.2
                                                elif (
                                                    regime_result[
                                                        "trading_recommendation"
                                                    ]["action"]
                                                    == "trade_conservatively"
                                                ):
                                                    regime_multiplier = 0.7
                                                elif (
                                                    regime_result[
                                                        "trading_recommendation"
                                                    ]["action"]
                                                    == "avoid_trading"
                                                ):
                                                    regime_multiplier = 0.3

                                            # Calculate final position sizing
                                            adjusted_risk = (
                                                base_risk_per_trade
                                                * confidence_multiplier
                                                * regime_multiplier
                                            )
                                            adjusted_risk = max(
                                                25.0, min(150.0, adjusted_risk)
                                            )  # Risk between $25-$150

                                            position_size = (
                                                adjusted_risk / current_price
                                            )  # Convert to crypto units

                                            logger.info(
                                                f"💰 ADAPTIVE SIZING: Base=${base_risk_per_trade:.0f} | Conf={confidence:.2%}({confidence_multiplier:.2f}x) | Regime=({regime_multiplier:.2f}x) | Final=${adjusted_risk:.0f}"
                                            )

                                            # 📐 ROADMAP SIZING: volume-scaled +
                                            # Kelly-capped (env-toggled)
                                            try:
                                                if (
                                                    os.getenv(
                                                        "ELVIS_VOLUME_SIZING", "1"
                                                    )
                                                    == "1"
                                                ):
                                                    from trading.risk.position_sizing import (
                                                        volume_multiplier,
                                                    )

                                                    _vol_mult = volume_multiplier(
                                                        symbol_history
                                                    )
                                                    if _vol_mult != 1.0:
                                                        position_size *= _vol_mult
                                                        logger.info(
                                                            f"📐 Volume sizing {_vol_mult:.2f}x -> {position_size:.6f}"
                                                        )
                                                if (
                                                    os.getenv("ELVIS_KELLY_SIZING", "0")
                                                    == "1"
                                                ):
                                                    from trading.risk.position_sizing import (
                                                        kelly_from_trades,
                                                    )
                                                    from utils.paper_trade_db import (
                                                        get_all_trades,
                                                    )

                                                    # get_all_trades rows:
                                                    # (id, timestamp, symbol,
                                                    #  side, price, quantity,
                                                    #  pnl, fee) -> pnl at [6].
                                                    # Column order is pinned by
                                                    # tests/test_roadmap_wiring.py
                                                    _PNL_IDX = 6
                                                    _SIDE_IDX = 3
                                                    _kelly_f = kelly_from_trades(
                                                        [
                                                            {"pnl": t[_PNL_IDX]}
                                                            for t in get_all_trades(
                                                                limit=200
                                                            )
                                                            if t[_SIDE_IDX]
                                                            in ("BUY", "SELL")
                                                        ]
                                                    )
                                                    _kelly_cap = (
                                                        float(available_balance)
                                                        * _kelly_f
                                                        / float(current_price)
                                                    )
                                                    if 0 < _kelly_cap < position_size:
                                                        position_size = _kelly_cap
                                                        logger.info(
                                                            f"📐 Kelly cap f*={_kelly_f:.3f} -> {position_size:.6f}"
                                                        )
                                            except Exception as sizing_error:
                                                logger.error(
                                                    f"⚠️ Roadmap sizing error: {sizing_error}"
                                                )

                                            # 💸 FEE GATE: skip trades whose
                                            # regime-expected move can't beat
                                            # all-in costs (env-toggled)
                                            if os.getenv("ELVIS_FEE_GATE", "1") == "1":
                                                try:
                                                    from trading.execution.exits import (
                                                        dynamic_take_profit,
                                                    )
                                                    from trading.fees.fee_gate import (
                                                        is_trade_viable,
                                                    )

                                                    _fee_regime_mode = os.getenv(
                                                        "ELVIS_TP_REGIME_MODE",
                                                        "legacy",
                                                    )
                                                    if _fee_regime_mode == "active":
                                                        _fee_regime = _validated_active_fee_profile(
                                                            take_profit_regime
                                                        )
                                                    else:
                                                        _fee_regime = getattr(
                                                            main,
                                                            "_last_regime",
                                                            {},
                                                        ).get(symbol, "RANGING")
                                                        if _fee_regime is None:
                                                            _fee_regime = "RANGING"
                                                    if _fee_regime is None:
                                                        logger.warning(
                                                            f"💸 {symbol} {signal} skipped: "
                                                            "current take-profit regime unavailable"
                                                        )
                                                        signal = "HOLD"
                                                    else:
                                                        _tp_price = dynamic_take_profit(
                                                            _fee_regime,
                                                            current_price,
                                                            side=signal,
                                                        )
                                                        _viable, _net, _costs = (
                                                            is_trade_viable(
                                                                current_price,
                                                                _tp_price,
                                                                position_size,
                                                                side=signal,
                                                            )
                                                        )
                                                        if not _viable:
                                                            logger.warning(
                                                                f"💸 {symbol} {signal} skipped by fee gate: "
                                                                f"net ${_net:.4f} after ${_costs['total']:.4f} fees "
                                                                f"({_fee_regime} TP target)"
                                                            )
                                                            signal = "HOLD"
                                                except Exception as fee_error:
                                                    logger.error(
                                                        f"⚠️ Fee gate error: {fee_error}"
                                                    )
                                                    signal = "HOLD"

                                            if signal in ("BUY", "SELL"):
                                                logger.info(
                                                    f"🎯 [{signal}] Submitting {symbol} paper order - Size: {position_size:.6f}, Price: ${current_price:.2f}"
                                                )
                                                intent = _legacy_order_intent(
                                                    symbol=symbol,
                                                    signal=signal,
                                                    confidence=confidence,
                                                    current_price=current_price,
                                                    position_size=position_size,
                                                    leverage=legacy_execution.default_leverage,
                                                    strategy_id=type(
                                                        active_strategy
                                                    ).__name__,
                                                )
                                                report = order_service.submit(intent)

                                                if _record_acknowledged_legacy_order(
                                                    report,
                                                    intent,
                                                    confidence=confidence,
                                                    strategy=active_strategy,
                                                    cooldown_manager=trading_loop.cooldown_manager,
                                                    logger=logger,
                                                ):
                                                    logger.info(
                                                        f"✅ [ACKNOWLEDGED] {symbol} {signal} paper order {report.venue_order_id}"
                                                    )
                                                elif report.requires_reconciliation:
                                                    logger.error(
                                                        f"⚠️ [AMBIGUOUS] {symbol} {signal} submission requires reconciliation; no automatic retry"
                                                    )
                                                else:
                                                    logger.warning(
                                                        f"🚫 [NOT SUBMITTED] {symbol} {signal}: {report.status.value} ({report.reason})"
                                                    )

                                        except Exception as e:
                                            logger.error(
                                                f"❌ Trading execution error for {symbol}: {e}"
                                            )

                                    else:
                                        logger.info(
                                            f"📊 {symbol} Signal: {signal} | Confidence: {confidence:.3f} | Action: HOLD (below 0.60 execution threshold)"
                                        )

                                except Exception as e:
                                    logger.error(f"❌ Error processing {symbol}: {e}")
                                    continue  # Continue with next symbol

                            # END OF SYMBOL LOOP
                            logger.info("✅ Completed processing all trading symbols")

                            # Score closed positions' model votes once per cycle
                            _score_model_votes(logger)

                            # CRITICAL: Check for stop losses on open positions FIRST
                            try:
                                from utils.paper_trade_db import get_open_positions

                                open_positions = get_open_positions()

                                for position in open_positions:
                                    try:
                                        # position format: (id, symbol, side, entry_price, quantity, leverage, entry_time)
                                        if len(position) < 5:
                                            continue

                                        # Debug: log the actual position tuple to see the structure
                                        logger.debug(f"Position tuple: {position}")
                                        logger.debug(
                                            f"Position length: {len(position)}"
                                        )

                                        pos_symbol = position[1]
                                        side = position[2]  # BUY/SELL

                                        # Add validation before float conversion
                                        try:
                                            entry_price = float(position[3])
                                        except (ValueError, TypeError) as e:
                                            logger.error(
                                                f"Could not convert entry_price '{position[3]}' to float: {e}"
                                            )
                                            logger.error(
                                                f"Full position tuple: {position}"
                                            )
                                            continue

                                        try:
                                            quantity = float(position[4])
                                        except (ValueError, TypeError) as e:
                                            logger.error(
                                                f"Could not convert quantity '{position[4]}' to float: {e}"
                                            )
                                            logger.error(
                                                f"Full position tuple: {position}"
                                            )
                                            continue

                                        if quantity == 0 or entry_price <= 0:
                                            continue

                                        # Get current price for this position.
                                        # ALWAYS fetch per pos_symbol: the old
                                        # `pos_symbol == symbol` shortcut paired
                                        # the leftover loop symbol with a price
                                        # computed from the BTC dataframe, so
                                        # BNB positions were closed against BTC
                                        # prices (fake -$553 "losses"). And a
                                        # mock-price fallback here once booked a
                                        # fictional +$816 close at $116,500 —
                                        # if no live price exists, SKIP the
                                        # position this cycle; fiction never
                                        # enters the books.
                                        position_current_price = (
                                            price_fetcher.get_current_price(pos_symbol)
                                        )
                                        if not position_current_price:
                                            logger.warning(
                                                f"No live price for {pos_symbol}; skipping exit checks this cycle"
                                            )
                                            continue

                                        # Calculate P&L percentage
                                        if side.upper() == "BUY":  # LONG position
                                            pnl_pct = (
                                                (position_current_price - entry_price)
                                                / entry_price
                                            ) * 100
                                        else:  # SHORT position
                                            pnl_pct = (
                                                (entry_price - position_current_price)
                                                / entry_price
                                            ) * 100

                                        # 🎯 STOP LOSS as % of position
                                        # notional (see _stop_loss_threshold)
                                        stop_loss_threshold_usd = _stop_loss_threshold(
                                            entry_price, quantity
                                        )

                                        # Calculate absolute dollar loss
                                        if side.upper() == "BUY":  # LONG position
                                            absolute_loss = (
                                                position_current_price - entry_price
                                            ) * quantity
                                        else:  # SHORT position
                                            absolute_loss = (
                                                entry_price - position_current_price
                                            ) * quantity

                                        if absolute_loss < stop_loss_threshold_usd:
                                            logger.warning(
                                                f"🛑 STOP LOSS triggered for {pos_symbol}: ${abs(absolute_loss):.2f} loss (limit: ${abs(stop_loss_threshold_usd):.2f})"
                                            )
                                            try:
                                                success = _close_position_by_id(
                                                    position,
                                                    position_current_price,
                                                    absolute_loss,
                                                    logger,
                                                )
                                                if success:
                                                    logger.info(
                                                        f"🛑 Stop loss closed {pos_symbol} (pnl ${absolute_loss:.2f})"
                                                    )
                                                    continue  # position closed
                                                else:
                                                    logger.error(
                                                        f"❌ Stop loss execution failed for {pos_symbol}"
                                                    )
                                            except Exception as e:
                                                logger.error(
                                                    f"❌ Stop loss execution error: {e}"
                                                )

                                        # 📉 TRAILING STOP (env-toggled): ratchets
                                        # with the trend, closes on 2% giveback
                                        if os.getenv("ELVIS_TRAILING_STOP", "1") == "1":
                                            try:
                                                from trading.execution.exits import (
                                                    TrailingStop,
                                                )

                                                if not hasattr(main, "_trailing"):
                                                    main._trailing = TrailingStop(
                                                        trail_pct=float(
                                                            os.getenv(
                                                                "ELVIS_TRAIL_PCT",
                                                                "0.02",
                                                            )
                                                        )
                                                    )
                                                _trail_close, _trail_stop = (
                                                    main._trailing.update(
                                                        str(position[0]),
                                                        side.upper(),
                                                        entry_price,
                                                        position_current_price,
                                                    )
                                                )
                                                if _trail_close:
                                                    logger.warning(
                                                        f"📉 TRAILING STOP hit for {pos_symbol} @ ${_trail_stop:.2f} "
                                                        f"(price ${position_current_price:.2f})"
                                                    )
                                                    _trail_pnl = (
                                                        (
                                                            position_current_price
                                                            - entry_price
                                                        )
                                                        if side.upper() == "BUY"
                                                        else (
                                                            entry_price
                                                            - position_current_price
                                                        )
                                                    ) * abs(quantity)
                                                    if _close_position_by_id(
                                                        position,
                                                        position_current_price,
                                                        _trail_pnl,
                                                        logger,
                                                    ):
                                                        logger.info(
                                                            f"📉 Trailing stop closed {side.upper()} "
                                                            f"{abs(quantity):.6f} {pos_symbol}"
                                                        )
                                                        main._trailing.clear(
                                                            str(position[0])
                                                        )
                                                        continue  # position closed
                                            except Exception as trail_error:
                                                logger.error(
                                                    f"⚠️ Trailing stop error: {trail_error}"
                                                )

                                        # MICRO-SCALPING TAKE PROFIT: 10 cents profit target
                                        # Calculate absolute profit in USD
                                        if side.upper() == "BUY":  # LONG position
                                            absolute_profit = (
                                                position_current_price - entry_price
                                            ) * quantity
                                        else:  # SHORT position
                                            absolute_profit = (
                                                entry_price - position_current_price
                                            ) * quantity

                                        # 🎯 HIGH WIN RATE OPTIMIZED PROFIT TARGETS
                                        take_profit_threshold_usd = 8.0  # $8.00 profit target (better win rate ratio)

                                        # 🎯 DYNAMIC TP (env-toggled): regime-aware
                                        # target replaces the fixed $8 when the
                                        # symbol's regime is known
                                        if os.getenv("ELVIS_DYNAMIC_TP", "1") == "1":
                                            try:
                                                from trading.execution.exits import (
                                                    dynamic_take_profit,
                                                )

                                                _tp_regime = getattr(
                                                    main, "_last_regime", {}
                                                ).get(pos_symbol)
                                                if _tp_regime:
                                                    _tp_target_price = (
                                                        dynamic_take_profit(
                                                            _tp_regime,
                                                            entry_price,
                                                            side=side.upper(),
                                                        )
                                                    )
                                                    take_profit_threshold_usd = (
                                                        abs(
                                                            _tp_target_price
                                                            - entry_price
                                                        )
                                                        * quantity
                                                    )
                                                    logger.debug(
                                                        f"🎯 Dynamic TP {pos_symbol}: {_tp_regime} -> "
                                                        f"${take_profit_threshold_usd:.2f}"
                                                    )
                                            except Exception as tp_error:
                                                logger.error(
                                                    f"⚠️ Dynamic TP error: {tp_error}"
                                                )

                                        if absolute_profit >= take_profit_threshold_usd:
                                            logger.info(
                                                f"💰 TAKE PROFIT triggered for {pos_symbol}: ${absolute_profit:.4f} profit (${take_profit_threshold_usd} target)"
                                            )
                                            try:
                                                success = _close_position_by_id(
                                                    position,
                                                    position_current_price,
                                                    absolute_profit,
                                                    logger,
                                                )
                                                if success:
                                                    logger.info(
                                                        f"💰 Take profit closed {side.upper()} {abs(quantity):.6f} {pos_symbol}"
                                                    )
                                                    continue  # position closed
                                                else:
                                                    logger.error(
                                                        f"❌ Take profit execution failed for {pos_symbol}"
                                                    )
                                            except Exception as e:
                                                logger.error(
                                                    f"❌ Take profit execution error: {e}"
                                                )

                                    except Exception as e:
                                        logger.error(
                                            f"❌ Position management error: {e}"
                                        )

                            except Exception as e:
                                logger.error(f"❌ Position management loop error: {e}")

                            # MAXIMUM SPEED TRADING - No delays whatsoever
                            current_time = time.time()

                            # Execute trades based on signals with MUCH HIGHER threshold to stop overtrading
                            if (
                                signal in ["BUY", "SELL"] and confidence >= 0.60
                            ):  # FIXED: Lowered threshold for more trades
                                logger.info(
                                    f"🎯 High confidence signal detected: {signal} with {confidence:.3f} confidence"
                                )

                            # Multi-symbol trading is now handled by the main loop above
                            logger.info(
                                "✅ Multi-symbol trading handled by main loop - BNBUSDT included"
                            )

                            # The legacy single-symbol branch reused the last
                            # multi-symbol decision and could place it twice.
                            # It is permanently fail-closed while its remaining
                            # sizing code is removed in the risk-planning slice.
                            signal = "HOLD"

                            # Continue with execution logic
                            current_price = data.iloc[-1]["close"]

                            # DYNAMIC RISK MANAGEMENT with x100 LEVERAGE ENFORCEMENT
                            available_balance = executor.get_account_balance()
                            base_leverage = getattr(
                                executor, "default_leverage", 100
                            )  # Default to x100

                            # Get risk manager for dynamic calculations
                            risk_manager = container.get_optional("risk_manager")

                            try:
                                # ENFORCE MINIMUM x50 LEVERAGE
                                if risk_manager and hasattr(
                                    risk_manager, "enforce_minimum_leverage"
                                ):
                                    leverage = risk_manager.enforce_minimum_leverage(
                                        base_leverage
                                    )
                                    # Use dynamic position sizing
                                    if hasattr(
                                        risk_manager, "calculate_dynamic_position_size"
                                    ):
                                        position_size = risk_manager.calculate_dynamic_position_size(
                                            available_balance,
                                            current_price,
                                            confidence,
                                            leverage,
                                        )
                                    else:
                                        # Fallback position size calculation
                                        position_size = min(
                                            0.001,
                                            float(available_balance)
                                            / float(current_price)
                                            * 0.05,
                                        )
                                    logger.info(
                                        f"⚡ DYNAMIC: Leverage {leverage}x, Dynamic sizing used"
                                    )
                                else:
                                    # Fallback to strategy calculation with enforced leverage
                                    leverage = max(
                                        base_leverage, 100.0
                                    )  # Ensure minimum x100
                                    balance_info = executor.get_balance()
                                    position_size = (
                                        active_strategy.calculate_position_size(
                                            data,
                                            current_price,
                                            available_capital=available_balance,
                                            leverage=leverage,
                                            signal_confidence=confidence,
                                        )
                                    )
                                    logger.info(
                                        f"⚡ ENFORCED: Leverage {leverage}x, Strategy sizing used"
                                    )
                            except Exception as e:
                                logger.error(f"Error in position size calculation: {e}")
                                # Emergency fallback position size with proper type conversion
                                leverage = max(
                                    base_leverage, 100.0
                                )  # Set leverage for emergency fallback
                                try:
                                    safe_balance = (
                                        float(available_balance)
                                        if available_balance is not None
                                        else 1000.0
                                    )
                                    safe_price = (
                                        float(current_price)
                                        if current_price is not None
                                        else 97000.0
                                    )
                                    position_size = min(
                                        0.001, safe_balance / safe_price * 0.05
                                    )  # REDUCED from 0.1 to 0.05
                                    logger.warning(
                                        f"🚨 Using emergency position size: {position_size:.6f} with leverage {leverage}x"
                                    )
                                except Exception as calc_error:
                                    logger.error(
                                        f"❌ Emergency calculation failed: {calc_error}"
                                    )
                                    logger.error(
                                        f"   available_balance: {available_balance} (type: {type(available_balance)})"
                                    )
                                    logger.error(
                                        f"   current_price: {current_price} (type: {type(current_price)})"
                                    )
                                    position_size = 0.001  # Ultra-safe fallback
                                    if "leverage" not in locals():
                                        leverage = 100.0  # Default leverage
                                    logger.warning(
                                        f"🚨 Using ultra-safe fallback position size: {position_size:.6f} with leverage {leverage}x"
                                    )

                            # Position size calculated - ready for execution

                            # Emergency check - force minimum position size if zero
                            if position_size <= 0:
                                if "leverage" not in locals():
                                    leverage = max(
                                        base_leverage, 100.0
                                    )  # Ensure leverage is defined
                                try:
                                    safe_balance = (
                                        float(available_balance)
                                        if available_balance is not None
                                        else 1000.0
                                    )
                                    safe_price = (
                                        float(current_price)
                                        if current_price is not None
                                        else 97000.0
                                    )
                                    position_size = min(
                                        0.001, safe_balance / safe_price * 0.05
                                    )  # REDUCED
                                    logger.warning(
                                        f"🚨 Position size was zero, forced to: {position_size:.6f}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"❌ Zero position size fix failed: {e}"
                                    )
                                    position_size = 0.001  # Ultra-safe fallback
                                    logger.warning(
                                        f"🚨 Using ultra-safe position size: {position_size:.6f}"
                                    )

                            logger.info(
                                f"🎯 DYNAMIC x50+ EXECUTION: {signal} order - Price: ${current_price:.2f}, Size: {position_size:.6f}, Balance: ${available_balance:.2f}, Leverage: {leverage}x"
                            )

                            # DUPLICATE TRADE PREVENTION
                            trade_key = f"{signal}_{symbol}_{current_price:.2f}_{position_size:.6f}"
                            current_time_ms = int(time.time() * 1000)

                            # Skip if same trade executed within last 30 seconds
                            if hasattr(trading_loop, "recent_trades"):
                                if (
                                    trading_loop.recent_trades.get(trade_key, 0) + 30000
                                    > current_time_ms
                                ):
                                    logger.warning(
                                        f"🚫 DUPLICATE PREVENTED: {trade_key}"
                                    )
                                    continue
                            else:
                                trading_loop.recent_trades = {}

                            # Execute order with new method
                            logger.info(
                                f"🎯 ORDER ATTEMPT: {signal} | Size: {position_size:.6f} | Price: ${current_price:.2f}"
                            )

                            if signal == "BUY":
                                logger.error(
                                    "Legacy duplicate BUY path is retired; order not submitted"
                                )
                                order_result = None
                                if order_result:
                                    logger.info(
                                        f"🎉 [SUCCESS] BUY order executed: {position_size:.6f} {symbol} at ${current_price:.2f}"
                                    )

                                    # 📝 RECORD TRADE IN COOLDOWN MANAGER
                                    trading_loop.cooldown_manager.record_trade(
                                        symbol, "BUY", position_size, confidence
                                    )

                                    # 💰 IMMEDIATE PROFIT CHECK: Check for profit-taking opportunities after every trade
                                    try:
                                        if hasattr(
                                            executor, "check_and_manage_positions"
                                        ):
                                            executor.check_and_manage_positions()
                                            logger.debug(
                                                f"🔍 Position management check completed for {symbol}"
                                            )
                                    except Exception as pos_e:
                                        logger.debug(
                                            f"Position management check error: {pos_e}"
                                        )

                                    # 🔥 FIX: Add position to risk manager for monitoring
                                    try:
                                        position_data = {
                                            "symbol": symbol,
                                            "side": "BUY",
                                            "entry_price": current_price,
                                            "quantity": position_size,
                                            "leverage": leverage,
                                            "timestamp": time.time(),
                                        }
                                        risk_manager.add_position(symbol, position_data)
                                        logger.info(
                                            f"✅ Position added to risk manager: BUY {position_size:.6f} {symbol}"
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"❌ Failed to add BUY position to risk manager: {e}"
                                        )

                                    # Record trade to prevent duplicates
                                    trading_loop.recent_trades[trade_key] = (
                                        current_time_ms
                                    )
                                    # Small delay to ensure trade completion
                                    time.sleep(0.5)
                                else:
                                    logger.error(
                                        f"❌ [FAIL] Failed to execute BUY order for {symbol} - Size: {position_size:.6f}, Price: ${current_price:.2f}"
                                    )
                            elif signal == "SELL":
                                logger.error(
                                    "Legacy duplicate SELL path is retired; order not submitted"
                                )
                                order_result = None
                                if order_result:
                                    logger.info(
                                        f"🎉 [SUCCESS] SELL order executed: {position_size:.6f} {symbol} at ${current_price:.2f}"
                                    )

                                    # 📝 RECORD TRADE IN COOLDOWN MANAGER
                                    trading_loop.cooldown_manager.record_trade(
                                        symbol, "SELL", position_size, confidence
                                    )

                                    # 💰 IMMEDIATE PROFIT CHECK: Check for profit-taking opportunities after every trade
                                    try:
                                        if hasattr(
                                            executor, "check_and_manage_positions"
                                        ):
                                            executor.check_and_manage_positions()
                                            logger.debug(
                                                f"🔍 Position management check completed for {symbol}"
                                            )
                                    except Exception as pos_e:
                                        logger.debug(
                                            f"Position management check error: {pos_e}"
                                        )

                                    # 🔥 FIX: Add position to risk manager for monitoring
                                    try:
                                        position_data = {
                                            "symbol": symbol,
                                            "side": "SELL",
                                            "entry_price": current_price,
                                            "quantity": position_size,
                                            "leverage": leverage,
                                            "timestamp": time.time(),
                                        }
                                        risk_manager.add_position(symbol, position_data)
                                        logger.info(
                                            f"✅ Position added to risk manager: SELL {position_size:.6f} {symbol}"
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"❌ Failed to add SELL position to risk manager: {e}"
                                        )

                                    # Record trade to prevent duplicates
                                    trading_loop.recent_trades[trade_key] = (
                                        current_time_ms
                                    )
                                    # Small delay to ensure trade completion
                                    time.sleep(0.5)
                                else:
                                    logger.error(
                                        f"❌ [FAIL] Failed to execute SELL order for {symbol} - Size: {position_size:.6f}, Price: ${current_price:.2f}"
                                    )
                            else:
                                logger.info(
                                    f"📊 Signal: {signal} | Confidence: {confidence:.3f} | Action: HOLD (below 60% threshold)"
                                )
                        else:
                            # Fallback for strategies without the new generate_signal method
                            logger.warning(
                                f"Strategy {type(active_strategy).__name__} doesn't have generate_signal method - using old approach"
                            )
                            if hasattr(active_strategy, "generate_signals"):
                                signal_data = {"BTCUSDT": data}
                                signals = active_strategy.generate_signals(signal_data)

                                for symbol, signal_info in signals.items():
                                    signal = signal_info.get("signal", "HOLD")
                                    confidence = signal_info.get("confidence", 0.0)

                                    if signal in ["BUY", "SELL"] and confidence >= 0.60:
                                        current_price = data.iloc[-1]["close"]
                                        available_balance = (
                                            executor.get_account_balance()
                                        )
                                        position_size = (
                                            active_strategy.calculate_position_size(
                                                data, current_price, available_balance
                                            )
                                        )

                                        logger.error(
                                            "Strategy %s produced %s through the retired "
                                            "generate_signals API; order not submitted",
                                            type(active_strategy).__name__,
                                            signal,
                                        )
                    else:
                        logger.error("Data is empty even after mock data creation!")

                    logger.debug("=== TRADING LOOP ITERATION END ===\n")

                    # Pace the loop using the strategy's documented trading
                    # frequency (e.g. 5-minute intervals) when set; otherwise
                    # keep the previous minimal 1-second pause. The sleep is
                    # broken into 1-second slices so shutdown stays responsive.
                    if shutdown_requested:
                        logger.info("Shutdown requested, exiting trading loop...")
                        return
                    loop_sleep_seconds = _strategy_loop_sleep_seconds(
                        active_strategy, default_seconds=1.0
                    )
                    logger.info(
                        f"⏱️ Next iteration in {loop_sleep_seconds:.0f}s "
                        "(strategy-paced)"
                    )
                    slept = 0.0
                    while slept < loop_sleep_seconds:
                        if shutdown_requested:
                            logger.info(
                                "Shutdown requested during pacing sleep, exiting..."
                            )
                            return
                        time.sleep(min(1.0, loop_sleep_seconds - slept))
                        slept += 1.0

                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    import traceback

                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    # Check for shutdown before sleeping in error handling
                    for i in range(60):
                        if shutdown_requested:
                            logger.info(
                                "Shutdown requested during error sleep, exiting trading loop..."
                            )
                            return
                        time.sleep(1)

            logger.info("Trading loop finished due to shutdown request")

        # Start trading thread as non-daemon so it can complete before shutdown
        global trading_thread
        trading_thread = threading.Thread(target=trading_loop, daemon=False)
        trading_thread.start()

        # The dashboard's run method is blocking, so it should be the last call
        import sys

        # Check if we can run a curses dashboard
        # Modified to be less restrictive - only require TERM to be set
        if not os.getenv("TERM"):
            logger.info("TERM not set - running in headless mode")
            logger.info("Check logs for trading activity")
            # Keep the main thread alive in headless mode
            signal.pause()
        else:
            try:
                logger.info("Starting console dashboard...")
                logger.info(
                    f"Dashboard has price_fetcher: {dashboard.price_fetcher is not None}"
                )
                logger.info(
                    "🎯 Look for Market Depth in the RIGHT PANE (columns 94-120)"
                )

                # Temporarily disable console logging to prevent screen jumping
                root_logger = logging.getLogger()
                console_handlers = [
                    h
                    for h in root_logger.handlers
                    if isinstance(h, logging.StreamHandler)
                ]
                for handler in console_handlers:
                    root_logger.removeHandler(handler)

                curses.wrapper(dashboard.run)

                # Re-enable console logging after dashboard exits
                for handler in console_handlers:
                    root_logger.addHandler(handler)
            except curses.error as e:
                logger.warning(f"Curses terminal UI not available: {e}")
                logger.info(
                    "Running in headless mode - check logs for trading activity"
                )
                # Keep the main thread alive in headless mode
                signal.pause()

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")

        # Publish error event
        event_bus.publish(
            SystemEvent(
                system_type="error",
                component="main",
                status="critical",
                message=f"Unexpected error: {str(e)}",
                source="main",
            )
        )
    finally:
        # Cleanup
        bootstrapper.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ELVIS Trading Bot with Dependency Injection"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper"],
        help="Trading mode (paper only)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    args = parser.parse_args()

    # Start the main bot
    main(mode=args.mode, log_level=args.log_level.upper())
