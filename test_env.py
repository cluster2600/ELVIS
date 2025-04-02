import os
from dotenv import load_dotenv

load_dotenv()
print("BINANCE_API_KEY:", os.getenv('BINANCE_API_KEY'))
print("BINANCE_API_SECRET:", os.getenv('BINANCE_API_SECRET'))
print("TELEGRAM_TOKEN:", os.getenv('TELEGRAM_TOKEN'))
print("TELEGRAM_CHAT_ID:", os.getenv('TELEGRAM_CHAT_ID'))
print("NEWSAPI_KEY:", os.getenv('NEWSAPI_KEY'))
