import os
from dotenv import load_dotenv

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "5"))

# Binance-символы (USDT-пары)
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

MAX_SIGNALS_PER_DAY = 3

if not TG_TOKEN:
    raise RuntimeError("TG_TOKEN не задан. Добавь переменную окружения на BotHost.")
