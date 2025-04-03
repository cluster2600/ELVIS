"""
Stress Testing Framework for portfolio risk assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class StressTester:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scenarios = {}
        self.positions = {}
        self.historical_data = {}
        self._load_scenarios()

    def _load_scenarios(self):
        """Load predefined stress test scenarios."""
        scenarios_path = self.config.get('scenarios_path', 'config/stress_scenarios.json')
        if os.path.exists(scenarios_path):
            with open(scenarios_path, 'r') as f:
                self.scenarios = json.load(f)
        else:
            # Default scenarios
            self.scenarios = {
                'market_crash': {
                    'description': 'Global market crash scenario',
                    'symbols': {
                        'BTC': {'return': -0.30},
                        'ETH': {'return': -0.35},
                        'SOL': {'return': -0.40}
                    },
                    'correlations': {
                        'BTC-ETH': 0.95,
                        'BTC-SOL': 0.90,
                        'ETH-SOL': 0.85
                    }
                },
                'volatility_spike': {
                    'description': 'High volatility scenario',
                    'symbols': {
                        'BTC': {'volatility_multiplier': 3.0},
                        'ETH': {'volatility_multiplier': 3.5},
                        'SOL': {'volatility_multiplier': 4.0}
                    }
                },
                'liquidity_crisis': {
                    'description': 'Market liquidity crisis',
                    'symbols': {
                        'BTC': {'spread_multiplier': 5.0},
                        'ETH': {'spread_multiplier': 6.0},
                        'SOL': {'spread_multiplier': 7.0}
                    }
                }
            }

    def add_position(self, symbol: str, size: float, price: float):
        """Add a position to the portfolio."""
        self.positions[symbol] = {
            'size': size,
            'price': price,
            'value': size * price,
            'last_updated': datetime.now()
        }

    def add_historical_data(self, symbol: str, returns: List[float], timestamps: List[datetime]):
        """Add historical returns data for a symbol."""
        self.historical_data[symbol] = pd.DataFrame({
            'timestamp': timestamps,
            'returns': returns
        }).set_index('timestamp')

    def run_stress_test(self, scenario_name: str) -> Dict:
        """Run a stress test scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario {scenario_name} not found")
            
        scenario = self.scenarios[scenario_name]
        results = {
            'scenario': scenario_name,
            'description': scenario['description'],
            'timestamp': datetime.now().isoformat(),
            'positions': {},
            'portfolio_impact': 0.0
        }
        
        total_impact = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in scenario['symbols']:
                impact = self._calculate_position_impact(symbol, position, scenario)
                results['positions'][symbol] = impact
                total_impact += impact
                
        results['portfolio_impact'] = total_impact
        return results

    def _calculate_position_impact(self, symbol: str, position: Dict, scenario: Dict) -> float:
        """Calculate the impact of a stress scenario on a position."""
        scenario_params = scenario['symbols'][symbol]
        position_value = position['value']
        
        if 'return' in scenario_params:
            # Direct return impact
            impact = position_value * scenario_params['return']
        elif 'volatility_multiplier' in scenario_params:
            # Volatility-based impact
            if symbol in self.historical_data:
                returns = self.historical_data[symbol]['returns']
                std_dev = returns.std()
                impact = position_value * std_dev * scenario_params['volatility_multiplier']
            else:
                impact = 0.0
        elif 'spread_multiplier' in scenario_params:
            # Liquidity impact
            impact = position_value * 0.01 * scenario_params['spread_multiplier']
        else:
            impact = 0.0
            
        return impact

    def run_all_scenarios(self) -> Dict:
        """Run all available stress test scenarios."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'scenarios': {}
        }
        
        for scenario_name in self.scenarios:
            try:
                scenario_results = self.run_stress_test(scenario_name)
                results['scenarios'][scenario_name] = scenario_results
            except Exception as e:
                self.logger.error(f"Error running scenario {scenario_name}: {e}")
                results['scenarios'][scenario_name] = {
                    'error': str(e)
                }
                
        return results

    def add_custom_scenario(self, name: str, scenario: Dict):
        """Add a custom stress test scenario."""
        self.scenarios[name] = scenario
        self._save_scenarios()

    def _save_scenarios(self):
        """Save scenarios to configuration file."""
        scenarios_path = self.config.get('scenarios_path', 'config/stress_scenarios.json')
        os.makedirs(os.path.dirname(scenarios_path), exist_ok=True)
        
        with open(scenarios_path, 'w') as f:
            json.dump(self.scenarios, f, indent=2)

    def get_scenario_summary(self) -> Dict:
        """Get summary of all available scenarios."""
        return {
            'total_scenarios': len(self.scenarios),
            'scenarios': {
                name: {
                    'description': scenario['description'],
                    'symbols': list(scenario['symbols'].keys())
                }
                for name, scenario in self.scenarios.items()
            }
        }

    def run_historical_stress_test(self, lookback_days: int = 30) -> Dict:
        """Run stress test based on historical worst-case scenarios."""
        if not self.historical_data:
            return {
                'error': 'No historical data available'
            }
            
        results = {
            'timestamp': datetime.now().isoformat(),
            'lookback_days': lookback_days,
            'worst_cases': {}
        }
        
        for symbol, position in self.positions.items():
            if symbol in self.historical_data:
                returns = self.historical_data[symbol]['returns']
                position_value = position['value']
                
                # Calculate worst daily return
                worst_return = returns.min()
                worst_impact = position_value * worst_return
                
                results['worst_cases'][symbol] = {
                    'worst_return': worst_return,
                    'impact': worst_impact,
                    'date': returns.idxmin().isoformat()
                }
                
        return results 