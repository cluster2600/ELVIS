from flask import Flask, jsonify
from flask_cors import CORS
from utils.paper_trade_db import get_all_trades, get_open_positions, get_trade_count, get_total_fees

app = Flask(__name__)
CORS(app)  # Enable CORS for Grafana access

def format_timestamp(ts):
    if hasattr(ts, 'isoformat'):
        return ts.isoformat() + 'Z'
    return str(ts)

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

@app.route('/trades', methods=['GET'])
def trades():
    try:
        trades = get_all_trades(limit=25)
        trade_list = [
            {
                "timestamp": format_timestamp(t[0]),
                "symbol": t[1],
                "side": t[2],
                "price": safe_float(t[3]),
                "quantity": safe_float(t[4]),
                "pnl": safe_float(t[5])
            }
            for t in trades
        ]
        return jsonify(trade_list)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch trades: {str(e)}"}), 500

@app.route('/trades/count', methods=['GET'])
def trades_count():
    try:
        count = get_trade_count()
        return jsonify({"count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/open_positions', methods=['GET'])
def open_positions():
    try:
        positions = get_open_positions()
        pos_list = [
            {
                "symbol": p[0],
                "entry_price": safe_float(p[1]),
                "quantity": safe_float(p[2]),
                "leverage": safe_float(p[3]),
                "entry_time": format_timestamp(p[4])
            }
            for p in positions
        ]
        return jsonify(pos_list)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch open positions: {str(e)}"}), 500

@app.route('/fees', methods=['GET'])
def fees():
    try:
        total_fees = get_total_fees()
        return jsonify({"total_fees": total_fees})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch fees: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)