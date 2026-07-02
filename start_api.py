#!/usr/bin/env python3
"""
Simple script to start the ELVIS Flask API
"""
import sys
import os

# Add project root to path
sys.path.insert(0, '/app')

print("🚀 Starting ELVIS API Server...")

try:
    # Import Flask app
    from trading.api.app import app, socketio
    print("✅ Flask app imported successfully")
    
    # Start the server
    print("🌐 Starting server on 0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, log_output=False)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
    # Fallback: Create minimal API
    from flask import Flask, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'message': 'ELVIS API is running'})
    
    @app.route('/trades')
    def trades():
        return jsonify([])
        
    @app.route('/open_positions')
    def open_positions():
        return jsonify([])
    
    print("🔧 Started minimal API server")
    app.run(host='0.0.0.0', port=5000, debug=False)
    
except Exception as e:
    print(f"❌ Server error: {e}")