#!/usr/bin/env python3.14
"""
Native ELVIS Console Dashboard - Exact Replica
Mimics the exact layout and appearance of the native console dashboard
"""

# Make the repo root importable no matter where this script is run from
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import curses
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

import requests


class NativeConsoleDashboard:
    """Replica of the native ELVIS console dashboard with exact layout"""

    def __init__(self):
        self.stdscr = None
        self.running = False
        self.animation_frame = 0

    def safe_addstr(self, y, x, text, attr=0, max_w=None):
        """Safely add string to screen with bounds checking.

        ``max_w`` clips to a pane-relative width so long content can never
        punch through a pane border (the screen-edge clip alone can't know
        where the pane ends).
        """
        try:
            if max_w is not None and len(text) > max_w:
                text = text[: max(0, max_w)]
            max_y, max_x = self.stdscr.getmaxyx()
            if 0 <= y < max_y and 0 <= x < max_x:
                # Truncate text if it would exceed screen width
                available_width = max_x - x - 1
                if len(text) > available_width:
                    text = text[:available_width]
                self.stdscr.addstr(y, x, text, attr)
        except curses.error:
            pass

    def safe_addch(self, y, x, char, attr=0):
        """Safely add character to screen with bounds checking"""
        try:
            max_y, max_x = self.stdscr.getmaxyx()
            if 0 <= y < max_y and 0 <= x < max_x:
                self.stdscr.addch(y, x, char, attr)
        except curses.error:
            pass

    def get_api_data(self, endpoint):
        """Fetch data from ELVIS API.

        Sends the X-API-Key header (env API_KEY) required since the API
        hardening, and returns None for any non-OK/error payload so callers'
        ``or []`` / ``or {}`` fallbacks kick in. Previously an error dict
        leaked through and panes iterated its keys, crashing with
        "'str' object has no attribute 'get'".
        """
        try:
            headers = {}
            api_key = self._dashboard_api_key()
            if api_key:
                headers["X-API-Key"] = api_key
            response = requests.get(
                f"http://localhost:5050{endpoint}", timeout=2, headers=headers
            )
            if not response.ok:
                return None
            data = response.json()
            if isinstance(data, dict) and "error" in data:
                return None
            return data
        except Exception:
            return None

    def _dashboard_api_key(self):
        """X-API-Key for the trade-history API: env API_KEY first (dev/CI),
        then OpenBao at secrets/dashboard field api_key — same resolution the
        server itself uses, so vault-backed setups need no env plumbing."""
        if not hasattr(self, "_api_key_resolved"):
            try:
                from utils.secrets_manager import resolve_dashboard_api_key

                self._api_key_resolved = resolve_dashboard_api_key()
            except Exception:
                # NOT leftover duplication: this outer guard covers the case
                # where utils.secrets_manager itself can't import (stripped
                # TUI-only env without the full secrets stack) — the env var
                # is then the only possible source.
                self._api_key_resolved = os.getenv("API_KEY")
        return self._api_key_resolved

    def _get_json_cached(self, url, params=None, ttl=5.0):
        """GET a JSON endpoint through a small TTL cache.

        The render loop runs every second; without this, several serial
        network calls per frame (prices, depth) could exceed the refresh
        interval under latency. Failures cache as None (retried after ttl).
        """
        key = (url, tuple(sorted((params or {}).items())))
        now = time.time()
        if not hasattr(self, "_net_cache"):
            self._net_cache = {}
        hit = self._net_cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        try:
            resp = requests.get(url, params=params, timeout=2)
            data = resp.json() if resp.ok else None
        except Exception:
            data = None
        self._net_cache[key] = (now, data)
        return data

    def _ticker_price(self, symbol, default):
        """Current price for symbol via the cached public ticker."""
        data = self._get_json_cached(
            "https://api.binance.com/api/v3/ticker/price", {"symbol": symbol}
        )
        try:
            return float(data["price"]) if data else default
        except KeyError, TypeError, ValueError:
            return default

    def _system_statuses(self, ttl=10.0):
        """Real Redis/Postgres connectivity, cached for ttl seconds."""
        now = time.time()
        cached = getattr(self, "_status_cache", None)
        if cached and now - cached[0] < ttl:
            return cached[1]
        statuses = {}
        try:
            import socket

            with socket.create_connection(("localhost", 6379), timeout=0.5):
                statuses["redis"] = True
        except Exception:
            statuses["redis"] = False
        try:
            from utils.paper_trade_db import get_conn

            conn = get_conn()
            statuses["postgres"] = conn is not None
            if conn is not None:
                conn.close()
        except Exception:
            statuses["postgres"] = False
        self._status_cache = (now, statuses)
        return statuses

    def _draw_box(self, start_y: int, start_x: int, end_y: int, end_x: int):
        """Draw a rectangular box using ASCII characters"""
        try:
            self.safe_addch(start_y, start_x, "+")
            self.safe_addch(start_y, end_x, "+")
            self.safe_addch(end_y, start_x, "+")
            self.safe_addch(end_y, end_x, "+")
            for x in range(start_x + 1, end_x):
                self.safe_addch(start_y, x, "-")
                self.safe_addch(end_y, x, "-")
            for y in range(start_y + 1, end_y):
                self.safe_addch(y, start_x, "|")
                self.safe_addch(y, end_x, "|")
        except curses.error:
            pass

    # Four dance frames for the King — silhouette contributed by the owner
    # (raised G hand, pompadour ,,)_ swoosh, mid-twist body). The note by his
    # hand and the leg stance shimmy once per refresh tick (1s).
    ELVIS_FRAMES = [
        [
            " G     __       ",
            " \\\\   ,,)_      ",
            "  \\'-\\( /       ",
            "    \\ | ,\\      ",
            "    \\|_/\\\\      ",
            "    / _ '.D     ",
            "  /_\\   |_\\     ",
        ],
        [
            " G ♪   __       ",
            " \\\\   ,,)_      ",
            "  \\'-\\( /       ",
            "    \\ | ,\\      ",
            "    \\|_/\\\\      ",
            "    / _ '.D     ",
            "   /_\\ |_\\      ",
        ],
        [
            " G     __       ",
            " \\\\   ,,)_      ",
            "  \\'-\\( /       ",
            "    \\ | ,\\      ",
            "    \\|_/\\\\      ",
            "    / _ '.D     ",
            "   |_\\  /_\\     ",
        ],
        [
            " G ♫   __       ",
            " \\\\   ,,)_      ",
            "  \\'-\\( /       ",
            "    \\ | ,\\      ",
            "    \\|_/\\\\      ",
            "    / _ '.D     ",
            "  /_\\    |_\\    ",
        ],
    ]

    # Captions cycle with the dance — the King has things to say.
    ELVIS_CAPTIONS = [
        "uh-huh-huh!      ",
        "* hip swing *    ",
        "~ jailhouse rock ~",
        "thank ya v. much ",
    ]

    def _draw_header(self):
        """Draw the kawaii ELVIS header: logo, dancing King, twinkling stars"""
        try:
            max_y, max_x = self.stdscr.getmaxyx()
            logo = [
                "███████╗██╗     ██╗   ██╗██╗███████╗",
                "██╔════╝██║     ██║   ██║██║██╔════╝",
                "█████╗  ██║     ██║   ██║██║███████╗",
                "██╔══╝  ██║     ╚██╗ ██╔╝██║╚════██║",
                "███████╗███████╗ ╚████╔╝ ██║███████║",
                "╚══════╝╚══════╝  ╚═══╝  ╚═╝╚══════╝",
            ]
            start_y = 1
            # Logo: alternating magenta/cyan rows, kawaii candy-stripe
            for i, line in enumerate(logo):
                x = (max_x - len(line)) // 2
                color = curses.color_pair(5 if i % 2 == 0 else 4)
                self.safe_addstr(start_y + i, x, line, color | curses.A_BOLD)

            # Dancing Elvis on the left, one dance step per second
            step = self.animation_frame % len(self.ELVIS_FRAMES)
            for i, line in enumerate(self.ELVIS_FRAMES[step]):
                self.safe_addstr(
                    start_y + i, 6, line, curses.color_pair(3) | curses.A_BOLD
                )

            # Twinkling stars on the right, phase-shifted by frame
            twinkle = ["✧ ･ﾟ", "･ﾟ ✧", "ﾟ✧ ･", " ✧･ﾟ"]
            for i in range(6):
                spark = twinkle[(self.animation_frame + i) % len(twinkle)]
                self.safe_addstr(start_y + i, max_x - 14, spark, curses.color_pair(5))

            # Ribbon under the logo carries the King's cycling caption
            ribbon = f"･ﾟ✧ {self.ELVIS_CAPTIONS[step].strip()} ✧ﾟ･"
            self.safe_addstr(
                start_y + 6,
                max(24, (max_x - len(ribbon)) // 2),
                ribbon,
                curses.color_pair(5) | curses.A_BOLD,
            )
        except curses.error:
            pass

    def _draw_info_pane(self, start_y: int, start_x: int, limit_y: int = None):
        """Draw the left pane with general info, PnL, and system status.

        ``limit_y`` is the pane's bottom border row; content never crosses it.
        """
        y = start_y
        if limit_y is None:
            limit_y = self.stdscr.getmaxyx()[0] - 2

        # Time and Status
        current_time = datetime.now()
        self.safe_addstr(
            y,
            start_x,
            f"Time: {current_time.strftime('%H:%M:%S')}",
            curses.color_pair(4),
        )
        self.safe_addstr(
            y + 1,
            start_x,
            f"Date: {current_time.strftime('%Y-%m-%d')}",
            curses.color_pair(6),
        )
        self.safe_addstr(
            y + 2,
            start_x,
            "Status: PAPER COMPATIBILITY",
            curses.color_pair(1) | curses.A_BOLD,
        )

        # Portfolio section
        y += 3
        self.safe_addstr(
            y, start_x, "--- Portfolio ---", curses.color_pair(3) | curses.A_BOLD
        )

        # Get live trade data and balance
        trades = self.get_api_data("/trades") or []
        balance_data = self.get_api_data("/balance") or {}

        # FIXED: Calculate P&L from recent trades only (not all historical data)
        realized_pnl = 0.0
        trade_count = 0

        # Get recent P&L directly from database to match balance calculation
        try:
            import psycopg2

            from utils.paper_trade_db import get_conn

            # Use the same connection method as the rest of the system
            conn = get_conn()

            if conn:
                with conn.cursor() as c:
                    # Get the latest reset timestamp
                    c.execute("""
                        SELECT reset_timestamp FROM trading_session_resets 
                        ORDER BY reset_timestamp DESC LIMIT 1
                    """)
                    reset_result = c.fetchone()

                    if reset_result:
                        reset_timestamp = reset_result[0]
                        # Get P&L from trades after reset only
                        c.execute(
                            """
                            SELECT COALESCE(SUM(pnl), 0), COUNT(*) FROM trades 
                            WHERE timestamp >= %s
                            AND pnl BETWEEN -100 AND 100
                        """,
                            (reset_timestamp,),
                        )
                        result = c.fetchone()
                        if result:
                            realized_pnl = (
                                float(result[0]) if result[0] is not None else 0.0
                            )
                            trade_count = int(result[1]) if result[1] is not None else 0
                    else:
                        # No reset found, default to 0 (fresh start)
                        realized_pnl = 0.0
                        trade_count = 0
                conn.close()
            else:
                # Connection failed, use defaults
                realized_pnl = 0.0
                trade_count = 0

            # Cap P&L to reasonable range
            realized_pnl = max(-1000.0, min(1000.0, realized_pnl))

            # P&L calculation complete - values updated

        except Exception as e:
            # If database fails, P&L = 0 (fresh start)
            realized_pnl = 0.0
            trade_count = 0

        # Get actual balances or use defaults - FIXED to show P&L-adjusted balances
        usdt_balance = balance_data.get("USDT", 1000.0)  # USDT (≈ USD)
        bnb_balance = balance_data.get("BNB", 1.67)  # BNB amount

        # Get current BNB price in USDT for accurate USD equivalent calculation
        bnb_price_usdt = self._ticker_price("BNBUSDT", 600.0)

        # Calculate USD equivalent value (USDT ≈ USD)
        bnb_usd_value = bnb_balance * bnb_price_usdt

        # Get BTC balance - paper trading now includes BTC
        btc_balance = balance_data.get("BTC", 0.008583)  # BTC amount (≈$1000 worth)

        # Get current BTC price in USDT for accurate USD equivalent calculation
        btc_price_usdt = self._ticker_price("BTCUSDT", 116500.0)

        # Calculate USD equivalent value (USDT ≈ USD)
        btc_usd_value = btc_balance * btc_price_usdt

        # FIXED: Calculate TRUE portfolio value including realized P&L and all assets
        initial_portfolio = (
            usdt_balance + bnb_usd_value + btc_usd_value
        )  # Initial balances ($3000 total)
        true_portfolio_value = (
            initial_portfolio + realized_pnl
        )  # Add P&L (negative reduces value)

        # Calculate unrealized P&L from open positions
        unrealized_pnl = 0.0
        open_positions = self.get_api_data("/open_positions") or []

        for pos in open_positions:
            try:
                symbol = pos.get("symbol", "")
                side = pos.get("side", "BUY")
                entry_price = float(pos.get("entry_price", 0))
                quantity = float(pos.get("quantity", 0))

                if quantity == 0 or entry_price == 0:
                    continue

                # Get current price for P&L calculation
                current_price = self._ticker_price(symbol, entry_price)

                # Calculate unrealized P&L
                if side.upper() == "BUY":
                    position_pnl = (current_price - entry_price) * quantity
                else:  # SELL/SHORT
                    position_pnl = (entry_price - current_price) * quantity

                unrealized_pnl += position_pnl
            except:
                continue

        y += 1
        # Portfolio breakdown; values compact enough to stay inside the
        # 38-col pane (per-asset USD detail lives in Total Value)
        pane_val_w = 24  # cols available from start_x+10 to the pane border
        self.safe_addstr(y, start_x, "💰 USDT:", curses.color_pair(6))
        self.safe_addstr(
            y,
            start_x + 10,
            f"${usdt_balance:,.2f}",
            curses.color_pair(2),
            max_w=pane_val_w,
        )
        y += 1

        self.safe_addstr(y, start_x, "💎 BNB:", curses.color_pair(6))
        self.safe_addstr(
            y,
            start_x + 10,
            f"{bnb_balance:.4f} @ ${bnb_price_usdt:,.2f}",
            curses.color_pair(2),
            max_w=pane_val_w,
        )
        y += 1

        self.safe_addstr(y, start_x, "₿ BTC:", curses.color_pair(6))
        self.safe_addstr(
            y,
            start_x + 10,
            f"{btc_balance:.6f} @ ${btc_price_usdt:,.0f}",
            curses.color_pair(2),
            max_w=pane_val_w,
        )
        y += 1

        # Total portfolio value
        portfolio_color = (
            curses.color_pair(1)
            if true_portfolio_value >= 3000.0
            else curses.color_pair(2)
        )
        self.safe_addstr(
            y, start_x, "💼 Total Value:", curses.color_pair(6) | curses.A_BOLD
        )
        self.safe_addstr(
            y,
            start_x + 15,
            f"${true_portfolio_value:,.2f} USD",
            portfolio_color | curses.A_BOLD,
        )
        y += 1

        # Unrealized P&L (from open positions)
        unrealized_color = (
            curses.color_pair(1) if unrealized_pnl >= 0 else curses.color_pair(2)
        )
        unrealized_sign = "+" if unrealized_pnl >= 0 else ""
        self.safe_addstr(y, start_x, "📊 Unrealized P&L:", curses.color_pair(6))
        self.safe_addstr(
            y,
            start_x + 19,
            f"{unrealized_sign}${unrealized_pnl:.2f} USD",
            unrealized_color,
        )
        y += 1

        # Realized P&L (from completed trades)
        pnl_color = curses.color_pair(1) if realized_pnl >= 0 else curses.color_pair(2)
        pnl_sign = "+" if realized_pnl >= 0 else ""
        self.safe_addstr(y, start_x, "💸 Realized P&L:", curses.color_pair(6))
        self.safe_addstr(
            y,
            start_x + 19,
            f"{pnl_sign}${realized_pnl:.2f} USD",
            pnl_color | curses.A_BOLD,
        )

        y += 1
        self.safe_addstr(
            y, start_x, f"Total Trades: {trade_count}", curses.color_pair(6)
        )

        y += 1
        win_rate = 0.0
        if trade_count > 0:
            winning_trades = sum(
                1 for trade in trades if float(trade.get("pnl", 0)) > 0
            )
            win_rate = (winning_trades / trade_count) * 100
        self.safe_addstr(y, start_x, f"Win Rate: {win_rate:.1f}%", curses.color_pair(3))

        # Mood kaomoji: the account's feelings about its own P&L
        y += 1
        if realized_pnl > 0:
            mood, mood_color = "mood: \\(^o^)/ yatta!", curses.color_pair(1)
        elif realized_pnl < 0:
            mood, mood_color = "mood: (T_T) ganbatte...", curses.color_pair(2)
        else:
            mood, mood_color = "mood: (-_-)zzz flat", curses.color_pair(6)
        self.safe_addstr(y, start_x, mood, mood_color | curses.A_BOLD)

        # Open Positions section
        y += 2
        self.safe_addstr(
            y, start_x, "--- Open Positions ---", curses.color_pair(3) | curses.A_BOLD
        )

        # Get open positions
        open_positions = self.get_api_data("/open_positions") or []
        y += 1

        if open_positions:
            for i, pos in enumerate(open_positions[:4]):  # Show up to 4 open positions
                if y + i >= self.stdscr.getmaxyx()[0] - 8:
                    break

                try:
                    symbol = pos.get("symbol", "N/A")[:8]
                    side = pos.get("side", "BUY")[:4]
                    entry_price = float(pos.get("entry_price", 0))
                    quantity = float(pos.get("quantity", 0))
                    leverage = float(pos.get("leverage", 1))

                    # Skip positions with zero quantity or price
                    if quantity == 0 or entry_price == 0:
                        continue

                    # Get current BTC price for P&L calculation
                    current_price = entry_price  # Default fallback
                    try:
                        # Try to get live price from Binance
                        response = requests.get(
                            f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
                            timeout=2,
                        )
                        if response.status_code == 200:
                            current_price = float(response.json()["price"])
                    except:
                        pass

                    # Calculate unrealized P&L based on position side
                    if side.upper() == "BUY":
                        unrealized_pnl = (current_price - entry_price) * quantity
                    else:  # SELL/SHORT
                        unrealized_pnl = (entry_price - current_price) * quantity

                    pnl_color = (
                        curses.color_pair(1)
                        if unrealized_pnl >= 0
                        else curses.color_pair(2)
                    )
                    pnl_sign = "+" if unrealized_pnl >= 0 else ""

                    # Show position info with side and leverage
                    pos_str = f"{symbol} {side} {quantity:.4f} {pnl_sign}${unrealized_pnl:.2f}"
                    self.safe_addstr(y + i, start_x, pos_str, pnl_color)
                except:
                    continue
            y += len(open_positions[:4])
        else:
            self.safe_addstr(y, start_x, "No open positions", curses.color_pair(6))
            y += 1

        # Recent Trades section
        self.safe_addstr(
            y, start_x, "--- Recent Trades ---", curses.color_pair(3) | curses.A_BOLD
        )

        recent_trades = trades[:4]  # Show fewer trades to make room for positions
        y += 1

        for i, trade in enumerate(recent_trades):
            if y + i >= limit_y:
                break

            try:
                symbol = trade.get("symbol", "N/A")[:8]
                side = trade.get("side", "N/A")[:4]
                pnl = float(trade.get("pnl", 0))
                quantity = float(trade.get("quantity", 0))

                pnl_color = curses.color_pair(1) if pnl >= 0 else curses.color_pair(2)
                pnl_sign = "+" if pnl >= 0 else ""

                trade_str = f"{symbol} {side} {pnl_sign}${pnl:.4f}"
                self.safe_addstr(y + i, start_x, trade_str, pnl_color)
            except:
                continue

        # System Info section (compact, two lines)
        y += len(recent_trades) + 1
        if y + 2 >= limit_y:
            return
        self.safe_addstr(
            y, start_x, "--- System Info ---", curses.color_pair(3) | curses.A_BOLD
        )

        y += 1
        # Check API health
        health = self.get_api_data("/health")
        api_status = (
            "HEALTHY" if health and health.get("status") == "healthy" else "OFFLINE"
        )
        api_color = (
            curses.color_pair(1) if api_status == "HEALTHY" else curses.color_pair(2)
        )
        self.safe_addstr(
            y, start_x, f"API: {api_status} · Ensemble", api_color | curses.A_BOLD
        )

        y += 1
        self.safe_addstr(y, start_x, "Mode: Paper · 5min HFT", curses.color_pair(4))

    def get_ohlc_data(self):
        """Generate OHLC candlestick data like the original console dashboard"""
        try:
            # Try to get real data from Binance
            response = requests.get(
                "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=40",
                timeout=3,
            )
            data = response.json()

            ohlc_data = []
            for candle in data:
                ohlc_data.append(
                    {
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    }
                )
            return ohlc_data
        except:
            # Fallback to mock data like original
            import random

            import numpy as np

            np.random.seed(42)  # Consistent mock data like original

            current_price = 67234.56
            base_price = current_price
            mock_data = []

            for i in range(40):
                # Create realistic OHLC candles exactly like original
                open_price = float(base_price)
                change_pct = np.random.normal(0, 0.002)  # 0.2% volatility
                close_price = float(open_price * (1 + change_pct))

                # High and low based on volatility
                volatility = abs(change_pct) + np.random.uniform(0.001, 0.005)
                high_price = float(max(open_price, close_price) * (1 + volatility / 2))
                low_price = float(min(open_price, close_price) * (1 - volatility / 2))

                mock_data.append(
                    {
                        "open": float(open_price),
                        "high": float(high_price),
                        "low": float(low_price),
                        "close": float(close_price),
                        "volume": float(np.random.uniform(100, 1000)),
                    }
                )
                base_price = close_price

            return mock_data

    def _draw_chart_pane(self, start_y: int, start_x: int, height: int, width: int):
        """Draw candlestick chart pane exactly like the original console dashboard"""
        self.safe_addstr(
            start_y,
            start_x,
            "--- BTC/USDT Candlestick Chart (1m) ---",
            curses.color_pair(3) | curses.A_BOLD,
        )

        # Get OHLC data
        ohlc_data = self.get_ohlc_data()
        if len(ohlc_data) < 2:
            self.safe_addstr(
                start_y + 2,
                start_x,
                "Waiting for candlestick data...",
                curses.color_pair(3),
            )
            return

        # Chart dimensions: reserve 12 cols on the right for the in-pane
        # price scale (it used to be drawn PAST the pane border, bleeding
        # into the market-depth pane)
        chart_height = height - 5
        chart_width = min(max(10, width - 14), len(ohlc_data))

        # Use the most recent candles that fit
        candles = (
            ohlc_data[-chart_width:] if len(ohlc_data) >= chart_width else ohlc_data
        )

        # Calculate price range for scaling
        all_highs = [c["high"] for c in candles]
        all_lows = [c["low"] for c in candles]
        min_price = min(all_lows)
        max_price = max(all_highs)
        price_range = max_price - min_price if max_price > min_price else 1.0

        # Draw candlesticks exactly like original
        for i, candle in enumerate(candles):
            if i >= chart_width:
                break

            open_price = candle["open"]
            high_price = candle["high"]
            low_price = candle["low"]
            close_price = candle["close"]

            # Calculate Y positions (inverted because screen coordinates)
            def price_to_y(price):
                normalized = (price - min_price) / price_range
                return int(
                    start_y + 2 + chart_height - 1 - (normalized * (chart_height - 1))
                )

            open_y = price_to_y(open_price)
            high_y = price_to_y(high_price)
            low_y = price_to_y(low_price)
            close_y = price_to_y(close_price)

            # Ensure positions are within bounds
            open_y = max(start_y + 2, min(open_y, start_y + chart_height))
            high_y = max(start_y + 2, min(high_y, start_y + chart_height))
            low_y = max(start_y + 2, min(low_y, start_y + chart_height))
            close_y = max(start_y + 2, min(close_y, start_y + chart_height))

            candle_x = start_x + 2 + i

            # Determine candle color (green for up, red for down)
            is_bullish = close_price >= open_price
            candle_color = curses.color_pair(1) if is_bullish else curses.color_pair(2)

            # Draw the wick (high-low line) exactly like original
            for y in range(min(high_y, low_y), max(high_y, low_y) + 1):
                self.safe_addch(y, candle_x, "|", curses.color_pair(6))

            # Draw the body (open-close rectangle) exactly like original
            body_top = min(open_y, close_y)
            body_bottom = max(open_y, close_y)

            if body_top == body_bottom:  # Doji (open == close)
                self.safe_addch(body_top, candle_x, "-", candle_color | curses.A_BOLD)
            else:
                # Draw body with different characters for bullish/bearish
                body_char = (
                    "█" if is_bullish else "▓"
                )  # Solid for bullish, lighter for bearish
                for y in range(body_top, body_bottom + 1):
                    self.safe_addch(
                        y, candle_x, body_char, candle_color | curses.A_BOLD
                    )

            # Mark open and close levels exactly like original
            self.safe_addch(open_y, candle_x, "○", candle_color)  # Open
            self.safe_addch(
                close_y, candle_x, "●", candle_color | curses.A_BOLD
            )  # Close

        # Draw price scale inside the pane's right edge
        scale_x = start_x + width - 10
        num_levels = min(8, chart_height // 2)
        for i in range(num_levels):
            if num_levels > 1:
                scale_price = max_price - (i / (num_levels - 1)) * price_range
            else:
                scale_price = (max_price + min_price) / 2
            scale_y = start_y + 2 + int(i * (chart_height - 1) / max(1, num_levels - 1))
            scale_y = max(start_y + 2, min(scale_y, start_y + chart_height - 1))
            self.safe_addstr(
                scale_y, scale_x, f"${scale_price:.0f}", curses.color_pair(6), max_w=9
            )

        # Draw volume bars at the bottom exactly like original
        volume_y = start_y + chart_height + 1
        self.safe_addstr(volume_y - 1, start_x, "Volume:", curses.color_pair(6))

        volumes = [c["volume"] for c in candles]
        max_vol = max(volumes) if volumes else 1
        if max_vol > 0:
            for i, candle in enumerate(candles):
                if i >= chart_width:
                    break
                volume = candle["volume"]
                vol_height = max(1, min(3, int((volume / max_vol) * 3)))
                candle_x = start_x + 2 + i
                for h in range(vol_height):
                    self.safe_addch(volume_y + h, candle_x, "▁", curses.color_pair(4))

        # Technical indicators at the bottom exactly like original
        current_price = candles[-1]["close"] if candles else 67234.56
        closes = [c["close"] for c in candles]

        # Calculate RSI like original
        rsi_val = 50.0
        if len(closes) >= 14:
            gains = []
            losses = []
            for i in range(1, len(closes)):
                change = closes[i] - closes[i - 1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            if len(gains) >= 14:
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi_val = 100 - (100 / (1 + rs))

        # Calculate SMA like original
        sma_val = current_price
        if len(closes) >= 20:
            sma_val = sum(closes[-20:]) / 20

        # Display indicators exactly like original
        indicator_y = start_y + height - 2
        rsi_color = (
            curses.color_pair(1) if 30 <= rsi_val <= 70 else curses.color_pair(2)
        )
        self.safe_addstr(indicator_y, start_x, f"RSI(14): {rsi_val:.1f}", rsi_color)
        self.safe_addstr(
            indicator_y, start_x + 15, f"SMA(20): ${sma_val:.0f}", curses.color_pair(6)
        )

        # Current price display
        price_change = (
            ((current_price - candles[0]["close"]) / candles[0]["close"] * 100)
            if len(candles) > 1
            else 0
        )
        price_color = (
            curses.color_pair(1) if price_change >= 0 else curses.color_pair(2)
        )
        change_sign = "+" if price_change >= 0 else ""

        self.safe_addstr(
            indicator_y + 1,
            start_x,
            f"Price: ${current_price:.2f} ({change_sign}{price_change:.2f}%)",
            price_color,
        )

    def _draw_market_depth_pane(
        self, start_y: int, start_x: int, height: int, width: int
    ):
        """Draw the right pane: live market depth + system status"""
        y = start_y
        limit_y = start_y + height

        self.safe_addstr(
            y, start_x, "--- Market Depth ---", curses.color_pair(3) | curses.A_BOLD
        )

        # Real public order book (same source paper trading uses); falls
        # back to placeholders if the fetch fails
        asks = [("--", "--")] * 4
        bids = [("--", "--")] * 4
        try:
            book = self._get_json_cached(
                "https://api.binance.com/api/v3/depth",
                {"symbol": "BTCUSDT", "limit": 5},
                ttl=3.0,
            )
            self._binance_ok = book is not None
            book = book or {}
            asks = [
                (f"{float(p):,.2f}", f"{float(q):.3f}")
                for p, q in reversed(book.get("asks", [])[:4])
            ] or asks
            bids = [
                (f"{float(p):,.2f}", f"{float(q):.3f}")
                for p, q in book.get("bids", [])[:4]
            ] or bids
        except Exception:
            pass

        y += 1
        self.safe_addstr(y, start_x, "      ASKS", curses.color_pair(2) | curses.A_BOLD)
        y += 1
        for i, (price, size) in enumerate(asks):
            if y + i >= limit_y:
                break
            self.safe_addstr(
                y + i, start_x, f"{price:>12} {size}", curses.color_pair(2)
            )

        y += len(asks)
        self.safe_addstr(y, start_x, "      BIDS", curses.color_pair(1) | curses.A_BOLD)
        y += 1
        for i, (price, size) in enumerate(bids):
            if y + i >= limit_y:
                break
            self.safe_addstr(
                y + i, start_x, f"{price:>12} {size}", curses.color_pair(1)
            )

        # API Status
        y += len(bids) + 1
        self.safe_addstr(
            y, start_x, "--- API Status ---", curses.color_pair(3) | curses.A_BOLD
        )

        y += 1
        health = self.get_api_data("/health")
        status_text = "✓ Connected" if health else "✗ Disconnected"
        status_color = curses.color_pair(1) if health else curses.color_pair(2)
        self.safe_addstr(y, start_x, f"ELVIS API: {status_text}", status_color)

        statuses = self._system_statuses()
        rows = [
            ("Binance", getattr(self, "_binance_ok", False)),
            ("Redis", statuses.get("redis", False)),
            ("PostgreSQL", statuses.get("postgres", False)),
        ]
        for name, ok in rows:
            y += 1
            if y >= limit_y:
                return
            mark, color = ("✓ Connected", 1) if ok else ("✗ Down", 2)
            self.safe_addstr(y, start_x, f"{name}: {mark}", curses.color_pair(color))

        # Position Sizing (bounded like the ladders above)
        y += 2
        if y + 3 >= limit_y:
            return
        self.safe_addstr(
            y, start_x, "--- Position Sizing ---", curses.color_pair(3) | curses.A_BOLD
        )

        y += 1
        self.safe_addstr(y, start_x, "Risk per Trade: 2%", curses.color_pair(6))

        y += 1
        self.safe_addstr(y, start_x, "Max Position: $200", curses.color_pair(6))

        y += 1
        self.safe_addstr(y, start_x, "Leverage: 10x", curses.color_pair(6))

    def _draw_console_messages(
        self, start_y: int, start_x: int, height: int, width: int
    ):
        """Draw console messages at the bottom"""
        y = start_y

        self.safe_addstr(
            y, start_x, "--- Console Messages ---", curses.color_pair(3) | curses.A_BOLD
        )

        # Recent log messages
        messages = [
            f"{datetime.now().strftime('%H:%M:%S')} - INFO - ELVIS trading system started successfully",
            f"{datetime.now().strftime('%H:%M:%S')} - INFO - Bonenkamp HFT strategy loaded",
            f"{datetime.now().strftime('%H:%M:%S')} - INFO - Paper trading mode active: $1000 USDT + $1000 BNB",
            f"{datetime.now().strftime('%H:%M:%S')} - INFO - Ensemble strategy initialized",
            f"{datetime.now().strftime('%H:%M:%S')} - INFO - Market data feed connected",
        ]

        for i, message in enumerate(messages[: height - 2]):
            if y + i + 1 >= start_y + height:
                break
            # Truncate message to fit width
            if len(message) > width - 2:
                message = message[: width - 5] + "..."
            self.safe_addstr(y + i + 1, start_x, message, curses.color_pair(6))

    def _draw_frame(self):
        """Draw a single frame of the dashboard UI"""
        try:
            self.stdscr.clear()
            max_y, max_x = self.stdscr.getmaxyx()

            if max_y < 40 or max_x < 120:
                self.stdscr.addstr(
                    0, 0, "Terminal too small, resize to at least 120x40"
                )
                self.stdscr.refresh()
                return

            # Main layout boxes
            self._draw_box(0, 0, max_y - 1, max_x - 1)  # Main border
            self._draw_header()

            # Define layout sections (even wider chart pane)
            left_pane_width = 38  # Further reduced to make chart even wider
            right_pane_width = 45  # Further reduced to make chart even wider
            chart_pane_width = max(
                35, max_x - left_pane_width - right_pane_width - 3
            )  # Much wider minimum chart

            chart_pane_x = left_pane_width + 1
            right_pane_x = chart_pane_x + chart_pane_width + 1

            # Three panes on top, a full-width console strip in its own box
            # below (it used to be painted straight across all three panes)
            bottom_split = max_y - 8

            # Draw panes
            self._draw_box(8, 1, bottom_split, left_pane_width)  # Left pane
            self._draw_box(
                8, chart_pane_x, bottom_split, chart_pane_x + chart_pane_width
            )  # Chart pane
            self._draw_box(8, right_pane_x, bottom_split, max_x - 2)  # Right pane

            # Draw content in panes
            self._draw_info_pane(9, 3, limit_y=bottom_split)
            self._draw_chart_pane(
                9, chart_pane_x + 1, bottom_split - 11, chart_pane_width - 1
            )

            market_depth_height = bottom_split - 10
            if market_depth_height > 10:
                self._draw_market_depth_pane(
                    9, right_pane_x + 2, market_depth_height, right_pane_width - 2
                )

            # Console messages strip
            self._draw_box(bottom_split + 1, 1, max_y - 2, max_x - 2)
            self._draw_console_messages(
                bottom_split + 2, 3, max_y - 2 - (bottom_split + 2), max_x - 8
            )

            self.animation_frame = (self.animation_frame + 1) % 10
            self.stdscr.refresh()

        except Exception as e:
            # Error handling
            self.safe_addstr(0, 0, f"Dashboard Error: {str(e)}", curses.color_pair(2))
            self.stdscr.refresh()

    def run(self):
        """Main dashboard loop with 1-second refresh"""

        def main(stdscr):
            self.stdscr = stdscr
            self.running = True

            # Initialize colors
            curses.start_color()
            curses.init_pair(
                1, curses.COLOR_GREEN, curses.COLOR_BLACK
            )  # Green for profits
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)  # Red for losses
            curses.init_pair(
                3, curses.COLOR_YELLOW, curses.COLOR_BLACK
            )  # Yellow for headers
            curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Cyan for info
            curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Magenta
            curses.init_pair(
                6, curses.COLOR_WHITE, curses.COLOR_BLACK
            )  # White for normal text

            # Hide cursor
            curses.curs_set(0)

            # Set nodelay for non-blocking input
            stdscr.nodelay(True)

            try:
                while self.running:
                    # Check for user input
                    try:
                        key = stdscr.getch()
                        if key == ord("q") or key == 27:  # 'q' or ESC to quit
                            break
                    except:
                        pass

                    # Draw frame
                    self._draw_frame()

                    # Sleep for 1 second (1-second refresh as requested)
                    time.sleep(1.0)

            except KeyboardInterrupt:
                pass
            finally:
                self.running = False

        # Redirect fd 2 (stderr) to a log file while curses owns the screen:
        # Vault/DB warnings from imported modules otherwise print straight
        # into the frame and corrupt the layout. fd 1 stays untouched —
        # curses renders through stdout.
        os.makedirs("logs", exist_ok=True)
        stderr_log = open("logs/dashboard_stderr.log", "a", buffering=1)
        saved_stderr_fd = os.dup(2)
        try:
            os.dup2(stderr_log.fileno(), 2)
            sys.stderr = stderr_log
            curses.wrapper(main)
        except KeyboardInterrupt:
            pass
        finally:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
            sys.stderr = sys.__stderr__
            stderr_log.close()
        print("\n👋 Console Dashboard stopped — sayonara ･ﾟ✧")
        print("Trade API still available at: http://localhost:5050/health")


def main():
    """Main entry point"""
    print("🤖 Starting ELVIS Native Console Dashboard...")
    print("📊 Exact replica of the native console dashboard layout")
    print("⚡ Refresh rate: 1 second")
    print("🎮 Press 'q' or ESC to exit")
    print("💰 Paper trading: $1000 USDT + $1000 BNB")
    print()

    dashboard = NativeConsoleDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
