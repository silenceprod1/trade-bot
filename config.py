import os
from dotenv import load_dotenv

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

if not TG_TOKEN:
    raise RuntimeError("TG_TOKEN не задан. Добавь переменную окружения на BotHost.")
