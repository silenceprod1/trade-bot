import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import TG_TOKEN, TG_CHAT_ID, SYMBOLS, SCAN_INTERVAL_MIN, MAX_SIGNALS_PER_DAY
from data import fetch
from setups import setup_a_asia_breakout, setup_c_trend_pullback, Signal
from indicators import snapshot
from backtest import run_backtest, stats_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

bot = Bot(token=TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

signals_today = 0
last_signal_date = None
_backtest_running = False


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
        if 7 <= now.hour < 10:
            sig = setup_a_asia_breakout(sym, df_m5, df_m15)
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
        "/scan — просканировать рынок\n"
        "/debug — данные по символам\n"
        "/backtest — бэктест 90 дней\n"
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


@dp.message(Command("debug"))
async def cmd_debug(m: Message):
    await m.answer("🔎 Собираю данные по символам...")
    lines = ["<b>DEBUG</b>\n"]

    for sym in SYMBOLS:
        try:
            df_m5 = await fetch(sym, "5m", 300)
            df_m15 = await fetch(sym, "15m", 300)
            df_h1 = await fetch(sym, "1h", 500)
            df_d1 = await fetch(sym, "1d", 300)
        except Exception as e:
            lines.append(f"❌ <b>{sym}</b>: ошибка fetch — <code>{e}</code>")
            continue

        if df_m5.empty or df_h1.empty:
            lines.append(f"❌ <b>{sym}</b>: пустые свечи")
            continue

        snap_m5 = snapshot(df_m5)
        snap_h1 = snapshot(df_h1)
        snap_d1 = snapshot(df_d1) if not df_d1.empty else {}

        try:
            sig_a = setup_a_asia_breakout(sym, df_m5, df_m15)
        except Exception as e:
            sig_a = f"ошибка: {e}"
        try:
            sig_c = setup_c_trend_pullback(sym, df_h1, df_d1)
        except Exception as e:
            sig_c = f"ошибка: {e}"

        a_text = f"✅ {sig_a.side}" if hasattr(sig_a, "side") else f"— {sig_a}"
        c_text = f"✅ {sig_c.side}" if hasattr(sig_c, "side") else f"— {sig_c}"

        block = (
            f"\n📊 <b>{sym}</b>\n"
            f"   свечей M5: {len(df_m5)}, H1: {len(df_h1)}, D1: {len(df_d1)}\n"
            f"   M5 close: <code>{snap_m5.get('last_close', 0):.4f}</code> "
            f"RSI: <code>{snap_m5.get('rsi14', 0):.1f}</code>\n"
            f"   H1 EMA50: <code>{snap_h1.get('ema50', 0):.4f}</code> "
            f"EMA200: <code>{snap_h1.get('ema200', 0):.4f}</code>\n"
            f"   H1 ADX: <code>{snap_h1.get('adx14', 0):.1f}</code> "
            f"ATR: <code>{snap_h1.get('atr14', 0):.4f}</code>\n"
            f"   D1 close: <code>{snap_d1.get('last_close', 0):.4f}</code>\n"
            f"   Сетап A: {a_text}\n"
            f"   Сетап C: {c_text}\n"
        )
        lines.append(block)

    text = "\n".join(lines)
    for i in range(0, len(text), 3500):
        await m.answer(text[i:i + 3500])


@dp.message(Command("backtest"))
async def cmd_backtest(m: Message):
    global _backtest_running
    if _backtest_running:
        await m.answer("⏳ Бэктест уже идёт. Дождись результата.")
        return

    _backtest_running = True
    chat_id = m.chat.id
    await m.answer(
        "🧪 Запускаю бэктест на 90 дней.\n"
        "Это займёт 5–15 минут. Я пришлю отчёт, когда закончу.\n"
        "Можешь пользоваться другими командами."
    )

    async def progress(text):
        try:
            await bot.send_message(chat_id, text)
        except Exception:
            pass

    async def worker():
        global _backtest_running
        try:
            trades = await run_backtest(SYMBOLS, days=90, progress_cb=progress)

            if not trades:
                await bot.send_message(
                    chat_id,
                    "❌ Сделок не найдено.\n\n" + stats_report()
                )
                return

            df = pd.DataFrame(trades)
            lines = ["<b>📊 РЕЗУЛЬТАТ БЭКТЕСТА (90 дней)</b>\n"]

            for setup in ["A", "C"]:
                sub = df[df["setup"] == setup]
                if len(sub) == 0:
                    lines.append(f"Сетап {setup}: 0 сделок")
                    continue
                wr = (sub["r"] > 0).sum() / len(sub) * 100
                avg = sub["r"].mean()
                total = sub["r"].sum()
                lines.append(
                    f"<b>Сетап {setup}</b>: {len(sub)} сделок | "
                    f"WR {wr:.1f}% | AvgR {avg:+.2f} | ΣR {total:+.1f}"
                )

            wr = (df["r"] > 0).sum() / len(df) * 100
            avg = df["r"].mean()
            total = df["r"].sum()
            sharpe = np.sqrt(len(df)) * avg / df["r"].std() if df["r"].std() > 0 else 0

            lines.append(
                f"\n<b>ИТОГО</b>\n"
                f"Сделок: {len(df)}\n"
                f"Winrate: <b>{wr:.1f}%</b>\n"
                f"Средний R: <b>{avg:+.3f}</b>\n"
                f"Суммарный R: <b>{total:+.1f}</b>\n"
                f"Sharpe: <b>{sharpe:.2f}</b>\n"
                f"При риске 1% на сделку: <b>{total:+.1f}%</b>\n\n"
                + stats_report()
            )

            text = "\n".join(lines)
            for i in range(0, len(text), 3500):
                await bot.send_message(chat_id, text[i:i + 3500])

        except Exception as e:
            await bot.send_message(chat_id, f"❌ Ошибка бэктеста: <code>{e}</code>")
        finally:
            _backtest_running = False

    asyncio.create_task(worker())


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
    await m.answer("Используй /scan, /debug или /backtest.")


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
