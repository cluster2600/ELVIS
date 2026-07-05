import os

from flask import Flask, jsonify, request

app = Flask(__name__)

# Load environment variables for API keys
import os

BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET")

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise EnvironmentError(
        "Binance API credentials are not set in environment variables."
    )


@app.route("/api/trade_history", methods=["GET"])
def get_trade_history():
    # Placeholder implementation
    # In real implementation, fetch trade history from database or API
    sample_data = {
        "trades": [
            {
                "id": 1,
                "symbol": "BTCUSDT",
                "price": 30000,
                "quantity": 0.1,
                "side": "BUY",
            },
            {
                "id": 2,
                "symbol": "BTCUSDT",
                "price": 31000,
                "quantity": 0.1,
                "side": "SELL",
            },
        ]
    }
    return jsonify(sample_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
