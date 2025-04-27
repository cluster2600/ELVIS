import logging
import time
from binance.client import Client
import websocket
import json
import threading

class PriceFetcher:
    def __init__(self, logger, client=None, symbol='BTCUSDT', timeframe='5m', history_limit=200):
        self.logger = logger or logging.getLogger(__name__)
        self.client = client  # Binance Client instance
        self.symbol = symbol
        self.timeframe = timeframe
        self.history_limit = history_limit
        self.ws = None
        self.candles = []
        self.running = False
        self.lock = threading.Lock()

    def get_historical_data(self):
        if self.client:
            try:
                klines = self.client.get_historical_klines(symbol=self.symbol, interval=self.timeframe, limit=self.history_limit)
                self.candles = klines  # Store as list of lists
                self.logger.info(f"Fetched {len(klines)} historical klines for {self.symbol}.")
            except Exception as e:
                self.logger.error(f"Error fetching historical klines: {e}")
        else:
            self.logger.error("Client not initialized.")

    def on_message(self, ws, message):
        data = json.loads(message)
        if data.get('e') == 'kline':
            kline = data['k']
            with self.lock:
                self.candles.append([kline['t'], float(kline['o']), float(kline['h']), float(kline['l']), float(kline['c']), float(kline['v'])])
            self.logger.info(f"Received kline: {kline}")
        else:
            # Fix: Ignore messages without 'e' key, likely subscription responses
            if 'e' not in data:
                self.logger.debug(f"Ignoring WebSocket message without 'e' key: {data}")
            else:
                self.logger.error(f"WebSocket message missing 'e' key: {data}")

    def on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws):
        self.logger.info("WebSocket closed")
        self.running = False

    def on_open(self, ws):
        self.logger.info("WebSocket opened")
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [f"{self.symbol.lower()}@kline_{self.timeframe}"],
            "id": 1
        }
        ws.send(json.dumps(subscribe_message))

    def start(self):
        self.get_historical_data()  # Fetch initial data
        self.ws = websocket.WebSocketApp(f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.timeframe}",
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.ws.on_open = self.on_open
        self.running = True
        threading.Thread(target=self.ws.run_forever).start()
        self.logger.info("Price fetcher started.")

    def get_current_price(self):
        if self.candles:
            with self.lock:
                return float(self.candles[-1][4])  # Last close price
        return None

    def get_current_candle(self):
        if self.candles:
            with self.lock:
                return self.candles[-1]
        return None

    def get_candle_history(self):
        with self.lock:
            return self.candles  # Return a copy if needed to avoid concurrency issues
