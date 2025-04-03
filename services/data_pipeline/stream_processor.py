"""
Stream Processor for real-time data handling and processing.
"""

import asyncio
import logging
from typing import Dict, List, Optional
import websockets
import json
from datetime import datetime
import pandas as pd
import numpy as np

class StreamProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.websocket = None
        self.data_buffer = []
        self.running = False
        self.last_update = None
        self.quality_metrics = {
            'latency': [],
            'throughput': [],
            'error_rate': 0
        }

    async def connect(self):
        """Establish WebSocket connection to data source."""
        try:
            self.websocket = await websockets.connect(self.config['websocket_url'])
            self.logger.info("WebSocket connection established")
            self.running = True
        except Exception as e:
            self.logger.error(f"Failed to establish WebSocket connection: {e}")
            raise

    async def process_message(self, message: str) -> Optional[Dict]:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            timestamp = datetime.now()
            
            # Validate data structure
            if not self._validate_data(data):
                self.quality_metrics['error_rate'] += 1
                return None
                
            # Calculate latency
            if 'timestamp' in data:
                latency = (timestamp - datetime.fromtimestamp(data['timestamp'])).total_seconds()
                self.quality_metrics['latency'].append(latency)
                
            # Update throughput metrics
            self.quality_metrics['throughput'].append(timestamp)
            
            # Clean old metrics
            self._clean_old_metrics()
            
            return data
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.quality_metrics['error_rate'] += 1
            return None

    def _validate_data(self, data: Dict) -> bool:
        """Validate data structure and content."""
        required_fields = self.config.get('required_fields', [])
        return all(field in data for field in required_fields)

    def _clean_old_metrics(self):
        """Remove metrics older than the retention period."""
        retention_period = self.config.get('metrics_retention_seconds', 300)
        cutoff_time = datetime.now() - pd.Timedelta(seconds=retention_period)
        
        self.quality_metrics['latency'] = [
            lat for lat in self.quality_metrics['latency']
            if lat < retention_period
        ]
        
        self.quality_metrics['throughput'] = [
            ts for ts in self.quality_metrics['throughput']
            if ts > cutoff_time
        ]

    async def start(self):
        """Start processing data stream."""
        await self.connect()
        while self.running:
            try:
                message = await self.websocket.recv()
                processed_data = await self.process_message(message)
                if processed_data:
                    self.data_buffer.append(processed_data)
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed. Attempting to reconnect...")
                await self.connect()
            except Exception as e:
                self.logger.error(f"Error in data stream: {e}")
                await asyncio.sleep(1)

    async def stop(self):
        """Stop the data stream processing."""
        self.running = False
        if self.websocket:
            await self.websocket.close()

    def get_quality_metrics(self) -> Dict:
        """Get current quality metrics."""
        return {
            'latency_avg': np.mean(self.quality_metrics['latency']) if self.quality_metrics['latency'] else 0,
            'throughput': len(self.quality_metrics['throughput']),
            'error_rate': self.quality_metrics['error_rate']
        }

    def get_buffer_stats(self) -> Dict:
        """Get statistics about the data buffer."""
        return {
            'buffer_size': len(self.data_buffer),
            'oldest_timestamp': min(item['timestamp'] for item in self.data_buffer) if self.data_buffer else None,
            'newest_timestamp': max(item['timestamp'] for item in self.data_buffer) if self.data_buffer else None
        } 