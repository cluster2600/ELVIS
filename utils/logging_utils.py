"""
Logging utilities for the BTC_BOT project.
This module provides functions for setting up loggers with consistent formatting.
"""

import logging
from logging.handlers import RotatingFileHandler
import colorlog
from datetime import datetime
import os

def setup_logger(name, log_to_file=True, log_level=logging.INFO):
    """
    Set up a logger with console and file handlers.
    
    Args:
        name (str): The name of the logger.
        log_to_file (bool, optional): Whether to log to a file. Defaults to True.
        log_level (int, optional): The logging level. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Console handler with color formatting
    console_handler = logging.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        log_colors={
            "DEBUG": "cyan", 
            "INFO": "bold_white", 
            "WARNING": "yellow", 
            "ERROR": "red", 
            "CRITICAL": "bold_red"
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        file_handler = RotatingFileHandler(
            f"{logs_dir}/{name}_{current_datetime}.log", 
            maxBytes=10*1024*1024, 
            backupCount=5
        )
        file_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", 
            datefmt="%d-%m-%Y %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    return logger

def print_info(logger, message):
    """
    Log an info message.
    
    Args:
        logger (logging.Logger): The logger to use.
        message (str): The message to log.
    """
    logger.info(message)
    
def print_error(logger, message, exc_info=False):
    """
    Log an error message.
    
    Args:
        logger (logging.Logger): The logger to use.
        message (str): The message to log.
        exc_info (bool, optional): Whether to include exception info. Defaults to False.
    """
    logger.error(message, exc_info=exc_info)
    
def print_warning(logger, message):
    """
    Log a warning message.
    
    Args:
        logger (logging.Logger): The logger to use.
        message (str): The message to log.
    """
    logger.warning(message)
    
def print_debug(logger, message):
    """
    Log a debug message.
    
    Args:
        logger (logging.Logger): The logger to use.
        message (str): The message to log.
    """
    logger.debug(message)