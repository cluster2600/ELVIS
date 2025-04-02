"""
Notification utilities for the BTC_BOT project.
This module provides functions for sending notifications via Telegram.
"""

import requests
import urllib.parse
import os
from utils.logging_utils import print_info, print_error
from config import API_CONFIG

def send_notification(logger, message, notification_type="info"):
    """
    Send a notification.
    
    Args:
        logger (logging.Logger): The logger to use.
        message (str): The message to send.
        notification_type (str): The type of notification (info, warning, error).
        
    Returns:
        bool: True if the notification was sent successfully, False otherwise.
    """
    # Format message based on notification type
    if notification_type == "warning":
        formatted_message = f"⚠️ WARNING: {message}"
    elif notification_type == "error":
        formatted_message = f"🚨 ERROR: {message}"
    else:
        formatted_message = f"ℹ️ {message}"
    
    # Get Telegram credentials from config
    token = API_CONFIG.get('TELEGRAM_TOKEN', os.environ.get('TELEGRAM_TOKEN', ''))
    chat_id = API_CONFIG.get('TELEGRAM_CHAT_ID', os.environ.get('TELEGRAM_CHAT_ID', ''))
    
    # Send notification via Telegram if credentials are available
    if token and chat_id:
        return telegram_notify(logger, formatted_message, token, chat_id)
    else:
        logger.info(f"Notification (not sent - no credentials): {formatted_message}")
        return False

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
