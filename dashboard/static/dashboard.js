
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
                
                const metricName = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                
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
                
                const strategyName = strategy.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                
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
        