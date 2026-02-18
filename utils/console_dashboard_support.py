import logging
import os

from utils.api_connection_tester import APIConnectionTester


class LogHandler(logging.Handler):
    """Capture logs for dashboard display."""

    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard
        self.setLevel(logging.INFO)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.dashboard.add_log_message(msg)
        except Exception:
            pass


def create_api_tester(logger):
    """
    Initialize API monitoring for the console dashboard.
    """
    if not os.getenv("VAULT_ADDR"):
        os.environ["VAULT_ADDR"] = "http://127.0.0.1:8200"
    if not os.getenv("VAULT_TOKEN"):
        logger.warning("VAULT_TOKEN is not set; Vault auth checks may fail.")

    api_tester = APIConnectionTester(logger)
    api_tester.start_monitoring()

    try:
        api_tester.test_all_apis()
        logger.info("Initial API test completed")
    except Exception as exc:
        logger.warning(f"Initial API test failed: {exc}")

    return api_tester
