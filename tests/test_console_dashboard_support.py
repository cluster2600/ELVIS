import logging

from utils.console_dashboard_support import LogHandler


class DummyDashboard:
    def __init__(self):
        self.messages = []

    def add_log_message(self, message):
        self.messages.append(message)


def test_log_handler_emits_message_to_dashboard():
    dashboard = DummyDashboard()
    handler = LogHandler(dashboard)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    handler.emit(record)

    assert dashboard.messages
    assert "INFO - hello" in dashboard.messages[0]
