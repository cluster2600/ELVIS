from binance.um_futures import UMFutures
import os
from dotenv import load_dotenv

load_dotenv()
client = UMFutures(
    key=os.getenv('TESTNET_FUTURES_API'),
    secret=os.getenv('TESTNET_FUTURES_SECRET'),
    base_url='https://testnet.binancefuture.com'
)

exchange_info = client.exchange_info()
symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['status'] == 'TRADING']
print("Available trading symbols:", symbols)