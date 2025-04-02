"""
Notification utilities for the BTC_BOT project.
This module provides functions for sending notifications via Telegram.
"""

import requests
import urllib.parse
from utils.logging_utils import print_info, print_error

def telegram_notify(logger, message, token, chat_id):
    """
    Send a notification via Telegram.
    
    Args:
        logger (logging.Logger): The logger to use.
        message (str): The message to send.
        token (str): The Telegram bot token.
        chat_id (str): The Telegram chat ID.
        
    Returns:
        bool: True if the notification was sent successfully, False otherwise.
    """
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={urllib.parse.quote(message)}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        print_info(logger, f"Telegram notification sent: {message}")
        return True
    except requests.exceptions.RequestException as e:
        print_error(logger, f"Failed to send Telegram notification: {e}")
        return False
    except Exception as e:
        print_error(logger, f"Unexpected error sending Telegram notification: {e}")
        return False
