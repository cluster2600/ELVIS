"""
Dashboard utilities for the ELVIS project.
This module provides functionality for creating and updating a real-time dashboard.
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import json

from config import FILE_PATHS

# Global variables for dashboard
dashboard_data = {
    'trades': [],
    'portfolio_value': [],
    'metrics': {},
    'model_performance': {},
    'strategy_signals': {},
    'market_data': {}
}

dashboard_lock = threading.Lock()

class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """
    HTTP handler for dashboard server.
    """
    
    def do_GET(self):
        """
        Handle GET requests.
        """
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open(os.path.join(FILE_PATHS['DASHBOARD_DIR'], 'index.html'), 'rb') as file:
                self.wfile.write(file.read())
        
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            with dashboard_lock:
                self.wfile.write(json.dumps(dashboard_data).encode())
        
        elif self.path.startswith('/static/'):
            file_path = os.path.join(FILE_PATHS['DASHBOARD_DIR'], self.path[8:])
            
            if os.path.exists(file_path):
                self.send_response(200)
                
                if file_path.endswith('.css'):
                    self.send_header('Content-type', 'text/css')
                elif file_path.endswith('.js'):
                    self.send_header('Content-type', 'application/javascript')
                elif file_path.endswith('.png'):
                    self.send_header('Content-type', 'image/png')
                elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                    self.send_header('Content-type', 'image/jpeg')
                
                self.end_headers()
                
                with open(file_path, 'rb') as file:
                    self.wfile.write(file.read())
            else:
                self.send_response(404)
                self.end_headers()
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """
        Suppress log messages.
        """
        return

class DashboardServer:
    """
    Server for dashboard.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8080):
        """
        Initialize the dashboard server.
        
        Args:
            host (str): The host to bind to.
            port (int): The port to bind to.
        """
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.running = False
    
    def start(self):
        """
        Start the dashboard server.
        """
        if self.running:
            return
        
        self.running = True
        
        # Create dashboard directory if it doesn't exist
        os.makedirs(FILE_PATHS['DASHBOARD_DIR'], exist_ok=True)
        
        # Create dashboard files
        self._create_dashboard_files()
        
        # Start server in a separate thread
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()
        
        # Open dashboard in browser
        webbrowser.open(f'http://{self.host}:{self.port}')
    
    def stop(self):
        """
        Stop the dashboard server.
        """
        if not self.running:
            return
        
        self.running = False
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
    
    def _run_server(self):
        """
        Run the dashboard server.
        """
        self.server = socketserver.TCPServer((self.host, self.port), DashboardHTTPHandler)
        self.server.serve_forever()
    
    def _create_dashboard_files(self):
        """
        Create dashboard files.
        """
        # Create index.html
        index_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ELVIS Dashboard</title>
            <link rel="stylesheet" href="static/styles.css">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <header>
                <h1>ELVIS Dashboard</h1>
                <p>Enhanced Leveraged Virtual Investment System</p>
            </header>
            
            <div class="dashboard-container">
                <div class="dashboard-row">
                    <div class="dashboard-card">
                        <h2>Portfolio Value</h2>
                        <canvas id="portfolioChart"></canvas>
                    </div>
                    <div class="dashboard-card">
                        <h2>Performance Metrics</h2>
                        <div id="metricsContainer"></div>
                    </div>
                </div>
                
                <div class="dashboard-row">
                    <div class="dashboard-card">
                        <h2>Recent Trades</h2>
                        <div class="table-container">
                            <table id="tradesTable">
                                <thead>
                                    <tr>
                                        <th>Time</th>
                                        <th>Symbol</th>
                                        <th>Side</th>
                                        <th>Price</th>
                                        <th>Quantity</th>
                                        <th>PnL</th>
                                    </tr>
                                </thead>
                                <tbody>
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="dashboard-card">
                        <h2>Strategy Signals</h2>
                        <div id="signalsContainer"></div>
                    </div>
                </div>
                
                <div class="dashboard-row">
                    <div class="dashboard-card">
                        <h2>Model Performance</h2>
                        <canvas id="modelChart"></canvas>
                    </div>
                    <div class="dashboard-card">
                        <h2>Market Data</h2>
                        <canvas id="marketChart"></canvas>
                    </div>
                </div>
            </div>
            
            <footer>
                <p>ELVIS Dashboard - Last updated: <span id="lastUpdated"></span></p>
            </footer>
            
            <script src="static/dashboard.js"></script>
        </body>
        </html>
        """
        
        with open(os.path.join(FILE_PATHS['DASHBOARD_DIR'], 'index.html'), 'w') as f:
            f.write(index_html)
        
        # Create styles.css
        styles_css = """
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f4f4f4;
        }
        
        header {
            background-color: #35424a;
            color: #ffffff;
            padding: 20px;
            text-align: center;
        }
        
        header h1 {
            margin-bottom: 10px;
        }
        
        .dashboard-container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 0 20px;
        }
        
        .dashboard-row {
            display: flex;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .dashboard-card {
            flex: 1;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin: 0 10px;
            min-width: 300px;
        }
        
        .dashboard-card h2 {
            margin-bottom: 15px;
            color: #35424a;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        
        .table-container {
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        table th, table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        table th {
            background-color: #f2f2f2;
        }
        
        .positive {
            color: green;
        }
        
        .negative {
            color: red;
        }
        
        footer {
            background-color: #35424a;
            color: #ffffff;
            text-align: center;
            padding: 10px;
            position: fixed;
            bottom: 0;
            width: 100%;
        }
        
        @media (max-width: 768px) {
            .dashboard-row {
                flex-direction: column;
            }
            
            .dashboard-card {
                margin: 10px 0;
            }
        }
        """
        
        os.makedirs(os.path.join(FILE_PATHS['DASHBOARD_DIR'], 'static'), exist_ok=True)
        
        with open(os.path.join(FILE_PATHS['DASHBOARD_DIR'], 'static', 'styles.css'), 'w') as f:
            f.write(styles_css)
        
        # Create dashboard.js
        dashboard_js = """
        // Dashboard JavaScript
        
        // Refresh interval in milliseconds
        const refreshInterval = 5000;
        
        // Charts
        let portfolioChart;
        let modelChart;
        let marketChart;
        
        // Initialize dashboard
        function initDashboard() {
            // Initialize charts
            initPortfolioChart();
            initModelChart();
            initMarketChart();
            
            // Load initial data
            updateDashboard();
            
            // Set up refresh interval
            setInterval(updateDashboard, refreshInterval);
        }
        
        // Initialize portfolio chart
        function initPortfolioChart() {
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            portfolioChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value (USD)',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
        
        // Initialize model chart
        function initModelChart() {
            const ctx = document.getElementById('modelChart').getContext('2d');
            modelChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Accuracy',
                        data: [],
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgb(75, 192, 192)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
        }
        
        // Initialize market chart
        function initMarketChart() {
            const ctx = document.getElementById('marketChart').getContext('2d');
            marketChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Price',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
        
        // Update dashboard with new data
        function updateDashboard() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    updatePortfolioChart(data.portfolio_value);
                    updateMetrics(data.metrics);
                    updateTradesTable(data.trades);
                    updateSignals(data.strategy_signals);
                    updateModelChart(data.model_performance);
                    updateMarketChart(data.market_data);
                    updateLastUpdated();
                })
                .catch(error => console.error('Error fetching dashboard data:', error));
        }
        
        // Update portfolio chart
        function updatePortfolioChart(portfolioData) {
            if (!portfolioData || portfolioData.length === 0) return;
            
            const labels = portfolioData.map(item => item.time);
            const values = portfolioData.map(item => item.value);
            
            portfolioChart.data.labels = labels;
            portfolioChart.data.datasets[0].data = values;
            portfolioChart.update();
        }
        
        // Update metrics
        function updateMetrics(metrics) {
            if (!metrics) return;
            
            const metricsContainer = document.getElementById('metricsContainer');
            metricsContainer.innerHTML = '';
            
            for (const [key, value] of Object.entries(metrics)) {
                const metricElement = document.createElement('div');
                metricElement.className = 'metric';
                
                const metricName = key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                
                let metricValue = value;
                if (typeof value === 'number') {
                    metricValue = value.toFixed(2);
                    
                    // Add color for percentage metrics
                    if (key.includes('pct') || key.includes('rate')) {
                        const valueClass = value >= 0 ? 'positive' : 'negative';
                        metricValue = `<span class="${valueClass}">${metricValue}%</span>`;
                    }
                }
                
                metricElement.innerHTML = `<strong>${metricName}:</strong> ${metricValue}`;
                metricsContainer.appendChild(metricElement);
            }
        }
        
        // Update trades table
        function updateTradesTable(trades) {
            if (!trades || trades.length === 0) return;
            
            const tableBody = document.querySelector('#tradesTable tbody');
            tableBody.innerHTML = '';
            
            // Sort trades by time (newest first)
            const sortedTrades = [...trades].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            
            // Take only the last 10 trades
            const recentTrades = sortedTrades.slice(0, 10);
            
            for (const trade of recentTrades) {
                const row = document.createElement('tr');
                
                // Format time
                const time = new Date(trade.timestamp).toLocaleTimeString();
                
                // Format PnL with color
                const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
                const pnl = `<span class="${pnlClass}">${trade.pnl.toFixed(2)}</span>`;
                
                row.innerHTML = `
                    <td>${time}</td>
                    <td>${trade.symbol}</td>
                    <td>${trade.side}</td>
                    <td>${trade.price.toFixed(2)}</td>
                    <td>${trade.quantity.toFixed(6)}</td>
                    <td>${pnl}</td>
                `;
                
                tableBody.appendChild(row);
            }
        }
        
        // Update signals
        function updateSignals(signals) {
            if (!signals) return;
            
            const signalsContainer = document.getElementById('signalsContainer');
            signalsContainer.innerHTML = '';
            
            for (const [strategy, signal] of Object.entries(signals)) {
                const signalElement = document.createElement('div');
                signalElement.className = 'signal';
                
                const strategyName = strategy.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                
                let signalText = 'Hold';
                let signalClass = '';
                
                if (signal.buy) {
                    signalText = 'Buy';
                    signalClass = 'positive';
                } else if (signal.sell) {
                    signalText = 'Sell';
                    signalClass = 'negative';
                }
                
                signalElement.innerHTML = `<strong>${strategyName}:</strong> <span class="${signalClass}">${signalText}</span>`;
                signalsContainer.appendChild(signalElement);
            }
        }
        
        // Update model chart
        function updateModelChart(modelData) {
            if (!modelData) return;
            
            const labels = Object.keys(modelData);
            const values = Object.values(modelData).map(model => model.accuracy || 0);
            
            modelChart.data.labels = labels;
            modelChart.data.datasets[0].data = values;
            modelChart.update();
        }
        
        // Update market chart
        function updateMarketChart(marketData) {
            if (!marketData || !marketData.prices || marketData.prices.length === 0) return;
            
            const labels = marketData.prices.map(item => item.time);
            const values = marketData.prices.map(item => item.price);
            
            marketChart.data.labels = labels;
            marketChart.data.datasets[0].data = values;
            marketChart.update();
        }
        
        // Update last updated time
        function updateLastUpdated() {
            const lastUpdated = document.getElementById('lastUpdated');
            lastUpdated.textContent = new Date().toLocaleString();
        }
        
        // Initialize dashboard when DOM is loaded
        document.addEventListener('DOMContentLoaded', initDashboard);
        """
        
        with open(os.path.join(FILE_PATHS['DASHBOARD_DIR'], 'static', 'dashboard.js'), 'w') as f:
            f.write(dashboard_js)

class DashboardManager:
    """
    Manager for dashboard.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the dashboard manager.
        
        Args:
            logger (logging.Logger): The logger to use.
        """
        self.logger = logger
        self.server = None
    
    def start_dashboard(self, host: str = 'localhost', port: int = 8080) -> None:
        """
        Start the dashboard.
        
        Args:
            host (str): The host to bind to.
            port (int): The port to bind to.
        """
        try:
            self.logger.info(f"Starting dashboard on http://{host}:{port}")
            
            # Create server
            self.server = DashboardServer(host, port)
            
            # Start server
            self.server.start()
            
            self.logger.info(f"Dashboard started on http://{host}:{port}")
            
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {e}")
    
    def stop_dashboard(self) -> None:
        """
        Stop the dashboard.
        """
        try:
            if self.server:
                self.logger.info("Stopping dashboard")
                self.server.stop()
                self.logger.info("Dashboard stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping dashboard: {e}")
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade to the dashboard.
        
        Args:
            trade (Dict[str, Any]): The trade to add.
        """
        try:
            with dashboard_lock:
                dashboard_data['trades'].append(trade)
                
                # Keep only the last 100 trades
                if len(dashboard_data['trades']) > 100:
                    dashboard_data['trades'] = dashboard_data['trades'][-100:]
            
        except Exception as e:
            self.logger.error(f"Error adding trade to dashboard: {e}")
    
    def update_portfolio_value(self, value: float) -> None:
        """
        Update the portfolio value.
        
        Args:
            value (float): The portfolio value.
        """
        try:
            with dashboard_lock:
                dashboard_data['portfolio_value'].append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'value': value
                })
                
                # Keep only the last 100 values
                if len(dashboard_data['portfolio_value']) > 100:
                    dashboard_data['portfolio_value'] = dashboard_data['portfolio_value'][-100:]
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update the metrics.
        
        Args:
            metrics (Dict[str, Any]): The metrics.
        """
        try:
            with dashboard_lock:
                dashboard_data['metrics'] = metrics
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def update_model_performance(self, model_performance: Dict[str, Any]) -> None:
        """
        Update the model performance.
        
        Args:
            model_performance (Dict[str, Any]): The model performance.
        """
        try:
            with dashboard_lock:
                dashboard_data['model_performance'] = model_performance
            
        except Exception as e:
            self.logger.error(f"Error updating model performance: {e}")
    
    def update_strategy_signals(self, strategy_signals: Dict[str, Any]) -> None:
        """
        Update the strategy signals.
        
        Args:
            strategy_signals (Dict[str, Any]): The strategy signals.
        """
        try:
            with dashboard_lock:
                dashboard_data['strategy_signals'] = strategy_signals
            
        except Exception as e:
            self.logger.error(f"Error updating strategy signals: {e}")
    
    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """
        Update the market data.
        
        Args:
            market_data (Dict[str, Any]): The market data.
        """
        try:
            with dashboard_lock:
                dashboard_data['market_data'] = market_data
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
