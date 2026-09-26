import os
from dotenv import load_dotenv

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "5"))

# 10 монет из COIN_CONFIGS стратегии TradeMind v9.41
SYMBOLS = [
    "XRP/USDT", "BCH/USDT", "APT/USDT", "SUI/USDT", "INJ/USDT",
    "SOL/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "ARB/USDT",
]

MAX_SIGNALS_PER_DAY = 10

if not TG_TOKEN:
    raise RuntimeError("TG_TOKEN не задан. Добавь переменную окружения на BotHost.")
