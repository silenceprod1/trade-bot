import asyncio
import logging
from datetime import datetime, timezone

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import TG_TOKEN, TG_CHAT_ID, SYMBOLS, SCAN_INTERVAL_MIN, MAX_SIGNALS_PER_DAY
from data import fetch
from setups import setup_a_asia_breakout, setup_c_trend_pullback, Signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

bot = Bot(token=TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

signals_today = 0
last_signal_date = None


def fmt_signal(sig: Signal) -> str:
    arrow = "🟢" if sig.side == "BUY" else "🔴"
    return (
        f"{arrow} <b>{sig.setup}</b> | {sig.symbol}\n"
        f"Направление: <b>{sig.side}</b>\n"
        f"Вход: <code>{sig.entry:.5f}</code>\n"
        f"SL: <code>{sig.sl:.5f}</code>\n"
        f"TP: <code>{sig.tp:.5f}</code>\n"
        f"Причина: {sig.reason}"
    )


async def scan_market(manual: bool = False, notify_chat_id: int | None = None):
    global signals_today, last_signal_date
    today = datetime.now(timezone.utc).date()
    if last_signal_date != today:
        signals_today = 0
        last_signal_date = today

    if not manual and signals_today >= MAX_SIGNALS_PER_DAY:
        return []

    found = []
    for sym in SYMBOLS:
        try:
            df_m5 = await fetch(sym, "5m", 300)
            df_m15 = await fetch(sym, "15m", 300)
            df_h1 = await fetch(sym, "1h", 500)
            df_d1 = await fetch(sym, "1d", 300)
        except Exception as e:
            log.warning(f"{sym} fetch error: {e}")
            continue

        if df_m5.empty or df_m15.empty or df_h1.empty or df_d1.empty:
            continue

        now = datetime.now(timezone.utc)
        sig = None

        # Сетап A работает только в лондонскую сессию
        if 7 <= now.hour < 10:
            sig = setup_a_asia_breakout(sym, df_m5, df_m15)

        # Сетап C — в любое время
        if not sig:
            sig = setup_c_trend_pullback(sym, df_h1, df_d1)

        if sig:
            found.append(sig)
            signals_today += 1
            target = notify_chat_id or (int(TG_CHAT_ID) if TG_CHAT_ID else None)
            if target:
                try:
                    await bot.send_message(target, fmt_signal(sig))
                except Exception as e:
                    log.warning(f"send error: {e}")

    return found


@dp.message(CommandStart())
async def cmd_start(m: Message):
    await m.answer(
        "👋 <b>Trade Signals Bot</b>\n\n"
        "Команды:\n"
        "/scan — просканировать рынок сейчас\n"
        "/status — статус бота\n"
        "/id — узнать chat_id\n"
        "/test — тестовый сигнал"
    )


@dp.message(Command("scan"))
async def cmd_scan(m: Message):
    await m.answer("🔍 Сканирую рынок...")
    sigs = await scan_market(manual=True, notify_chat_id=m.chat.id)
    if not sigs:
        await m.answer("Сигналов нет. Рынок либо спит, либо не по правилам.")


@dp.message(Command("status"))
async def cmd_status(m: Message):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    await m.answer(
        f"✅ Работаю\n"
        f"🕒 {now}\n"
        f"📊 Сигналов сегодня: {signals_today}/{MAX_SIGNALS_PER_DAY}"
    )


@dp.message(Command("id"))
async def cmd_id(m: Message):
    await m.answer(f"Твой chat_id: <code>{m.chat.id}</code>")


@dp.message(Command("test"))
async def cmd_test(m: Message):
    s = Signal("BTC/USDT", "BUY", 60000, 59000, 63000, "TEST", "проверка")
    await m.answer(fmt_signal(s))


@dp.message(F.text)
async def echo(m: Message):
    await m.answer("Используй /scan или /start.")


async def main():
    log.info("Старт бота...")
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(scan_market, "interval", minutes=SCAN_INTERVAL_MIN)
    scheduler.start()
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Остановлен")
