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
from data import fetch, fetch_candles
from levels import get_major_levels
from fvgs import get_fvgs
from trademind import analyze, generate_neurobro_report, STRATEGY_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

bot = Bot(token=TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

signals_today = 0
last_signal_date = None
_recent_ready = {}   # {symbol: last_ready_time} — чтобы не спамить одним и тем же


def _sym_to_code(symbol: str) -> str:
    """'XRP/USDT' -> 'XRPUSDT' (формат стратегии)."""
    return symbol.replace("/", "").upper()


async def build_snapshot(symbol: str):
    """Возвращает (price, c1h, c15, c5, levels, fvgs) либо None."""
    df_m5 = await fetch(symbol, "5m", 300)
    c1h = await fetch_candles(symbol, "1h", 500)
    c15 = await fetch_candles(symbol, "15m", 300)
    c5 = await fetch_candles(symbol, "5m", 300)

    if not c1h or not c15 or not c5 or df_m5.empty:
        return None

    price = float(df_m5["close"].iloc[-1])
    levels = await get_major_levels(symbol)
    fvgs = await get_fvgs(symbol)

    return price, c1h, c15, c5, levels, fvgs


async def scan_market(manual: bool = False, notify_chat_id: int | None = None):
    global signals_today, last_signal_date
    today = datetime.now(timezone.utc).date()
    if last_signal_date != today:
        signals_today = 0
        last_signal_date = today
        _recent_ready.clear()

    if not manual and signals_today >= MAX_SIGNALS_PER_DAY:
        return []

    found = []
    target = notify_chat_id or (int(TG_CHAT_ID) if TG_CHAT_ID else None)

    for sym in SYMBOLS:
        code = _sym_to_code(sym)
        try:
            snap = await build_snapshot(sym)
        except Exception as e:
            log.warning(f"{sym} snapshot error: {e}")
            continue
        if not snap:
            continue

        price, c1h, c15, c5, levels, fvgs = snap

        try:
            result = analyze(
                candles_1h=c1h,
                candles_15m=c15,
                candles_5m=c5,
                current_price=price,
                major_levels=levels,
                fvgs=fvgs,
                symbol=code,
            )
        except Exception as e:
            log.warning(f"{sym} analyze error: {e}")
            continue

        stage = result.get("stage", "WAIT")
        if stage not in ("READY", "WAIT_PULLBACK"):
            continue

        # анти-спам: один сигнал на символ в 30 минут
        now = datetime.now(timezone.utc)
        last = _recent_ready.get(sym)
        if last and (now - last).total_seconds() < 1800:
            continue
        _recent_ready[sym] = now

        found.append(result)
        signals_today += 1

        if target:
            try:
                text = generate_neurobro_report(result, code, risk_pct=1.0)
                await bot.send_message(target, text)
            except Exception as e:
                log.warning(f"send error: {e}")

    return found


@dp.message(CommandStart())
async def cmd_start(m: Message):
    await m.answer(
        f"👋 <b>TradeMind Bot v{STRATEGY_VERSION}</b>\n\n"
        "Команды:\n"
        "/scan — просканировать 10 монет\n"
        "/debug — показать stage/score по всем монетам\n"
        "/status — статус бота\n"
        "/id — узнать chat_id\n"
        "/test — тестовое сообщение"
    )


@dp.message(Command("scan"))
async def cmd_scan(m: Message):
    await m.answer("🔍 Сканирую 10 монет (XRP, BCH, APT, SUI, INJ, SOL, ADA, AVAX, LINK, ARB)...")
    sigs = await scan_market(manual=True, notify_chat_id=m.chat.id)
    if not sigs:
        await m.answer("Сигналов READY нет. Рынок не даёт сетапов.")


@dp.message(Command("debug"))
async def cmd_debug(m: Message):
    await m.answer("🔎 Собираю данные по 10 монетам... Это займёт ~30 секунд.")
    lines = [f"<b>DEBUG TradeMind v{STRATEGY_VERSION}</b>\n"]

    for sym in SYMBOLS:
        code = _sym_to_code(sym)
        try:
            snap = await build_snapshot(sym)
        except Exception as e:
            lines.append(f"❌ {sym}: snapshot error {e}")
            continue
        if not snap:
            lines.append(f"❌ {sym}: нет данных")
            continue

        price, c1h, c15, c5, levels, fvgs = snap

        try:
            r = analyze(
                candles_1h=c1h, candles_15m=c15, candles_5m=c5,
                current_price=price, major_levels=levels,
                fvgs=fvgs, symbol=code,
            )
        except Exception as e:
            lines.append(f"❌ {sym}: analyze error {e}")
            continue

        long_r = r.get("long") or {}
        short_r = r.get("short") or {}
        best = long_r if long_r.get("score", 0) >= short_r.get("score", 0) else short_r

        lines.append(
            f"\n📊 <b>{code}</b> | price <code>{price:.4f}</code>\n"
            f"   context: {r.get('context_direction')} | "
            f"stage: <b>{best.get('stage')}</b> | score: <b>{best.get('score')}</b>\n"
            f"   dir: {best.get('direction')} | "
            f"conf: {best.get('confirmation')} | bos: {best.get('bos')}\n"
            f"   trend: {best.get('trend_activity')} | "
            f"fvg_bonus: {best.get('fvg_bonus')}\n"
            f"   reason: {best.get('reason')}\n"
            f"   levels: {len(levels)} | fvgs: {len(fvgs)}"
        )

    text = "\n".join(lines)
    for i in range(0, len(text), 3500):
        await m.answer(text[i:i + 3500])


@dp.message(Command("status"))
async def cmd_status(m: Message):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    await m.answer(
        f"✅ TradeMind Bot v{STRATEGY_VERSION}\n"
        f"🕒 {now}\n"
        f"📊 Сигналов сегодня: {signals_today}/{MAX_SIGNALS_PER_DAY}"
    )


@dp.message(Command("id"))
async def cmd_id(m: Message):
    await m.answer(f"Твой chat_id: <code>{m.chat.id}</code>")


@dp.message(Command("test"))
async def cmd_test(m: Message):
    fake_result = {
        "stage": "READY",
        "direction": "LONG",
        "score": 88,
        "reason": "Sweep→15M BOS→5M ILM. Trend 0.65. RR 2.0. BOS=True.",
        "entry": 1.2345,
        "sl": 1.2200,
        "tp": 1.2635,
    }
    await m.answer(generate_neurobro_report(fake_result, "TESTUSDT", risk_pct=1.0))


@dp.message(F.text)
async def echo(m: Message):
    await m.answer("Используй /scan, /debug или /start.")


async def main():
    log.info(f"Старт TradeMind Bot v{STRATEGY_VERSION}...")
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
