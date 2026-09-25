import asyncio
import logging
from datetime import datetime, timezone

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties

from config import TG_TOKEN, TG_CHAT_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

bot = Bot(
    token=TG_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 <b>Trade Signals Bot</b> запущен.\n\n"
        "Команды:\n"
        "/start — приветствие\n"
        "/status — статус бота\n"
        "/ping — проверка связи\n"
        "/test — тестовый сигнал\n"
        "/id — показать твой chat_id"
    )


@dp.message(Command("ping"))
async def cmd_ping(message: Message):
    await message.answer("🏓 pong")


@dp.message(Command("status"))
async def cmd_status(message: Message):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    await message.answer(
        f"✅ Бот работает\n"
        f"🕒 Время сервера: <code>{now}</code>"
    )


@dp.message(Command("id"))
async def cmd_id(message: Message):
    await message.answer(f"Твой chat_id: <code>{message.chat.id}</code>")


@dp.message(Command("test"))
async def cmd_test(message: Message):
    text = (
        "🎯 <b>TEST</b> | EURUSD\n"
        "Направление: <b>BUY</b>\n"
        "Вход: <code>1.08500</code>\n"
        "SL: <code>1.08300</code>\n"
        "TP: <code>1.09000</code>\n"
        "Причина: тестовый сигнал"
    )
    await message.answer(text)
    if TG_CHAT_ID:
        try:
            await bot.send_message(int(TG_CHAT_ID), "🔔 Тест доставки в канал прошёл.")
        except Exception as e:
            log.warning(f"Не удалось отправить в TG_CHAT_ID: {e}")


@dp.message(F.text)
async def echo(message: Message):
    await message.answer("Принял. Используй /start для списка команд.")


async def main():
    log.info("Бот запускается...")
    await bot.delete_webhook(drop_pending_updates=True)
    log.info("Polling started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Бот остановлен")
