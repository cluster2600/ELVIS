import time
import psutil
import requests

class SystemMonitor:
    """
    Monitors system health metrics such as network latency, API rate limits, and error rates.
    """

    def __init__(self, api_url: str):
        """
        Initialize the SystemMonitor.

        Args:
            api_url (str): The URL of the API to monitor.
        """
        self.api_url = api_url
        self.error_rates = {}

    def get_network_latency(self) -> float:
        """
        Get the network latency to the API server.
        """
        try:
            start_time = time.time()
            requests.get(self.api_url, timeout=5)
            end_time = time.time()
            return (end_time - start_time) * 1000  # in ms
        except requests.exceptions.RequestException:
            return -1.0

    def get_api_rate_limits(self, response_headers: dict) -> dict:
        """
        Extract API rate limit information from response headers.
        """
        return {
            'limit': response_headers.get('x-mbx-used-weight-1m'),
            'remaining': response_headers.get('x-mbx-order-count-1d')
        }

    def record_error(self, component: str):
        """
        Record an error for a specific component.
        """
        if component not in self.error_rates:
            self.error_rates[component] = 0
        self.error_rates[component] += 1

    def get_error_rates(self) -> dict:
        """
        Get the current error rates.
        """
        return self.error_rates
