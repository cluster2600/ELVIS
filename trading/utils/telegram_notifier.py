# trading/utils/telegram_notifier.py

import logging

import requests


class TelegramNotifier:
    """
    Sends notifications to a Telegram chat using a bot token and chat ID.
    """

    def __init__(self, bot_token: str, chat_id: str, logger: logging.Logger = None):
        """
        Initialize the TelegramNotifier.

        Args:
            bot_token (str): Telegram bot token.
            chat_id (str): Telegram chat ID.
            logger (logging.Logger, optional): Logger instance.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logger or logging.getLogger(__name__)
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, message: str) -> None:
        """
        Send a message to the Telegram chat.

        Args:
            message (str): The message text to send.
        """
        try:
            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
            response = requests.post(self.api_url, json=payload, timeout=5)

            if response.status_code != 200:
                self.logger.error(
                    f"Telegram API error: {response.status_code} - {response.text}"
                )
            else:
                self.logger.debug(f"Telegram message sent: {message}")
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
