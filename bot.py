import asyncio
import logging
from datetime import datetime, timezone

import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import TG_TOKEN, TG_CHAT_ID, SYMBOLS, SCAN_INTERVAL_MIN, MAX_SIGNALS_PER_DAY
from data import fetch, fetch_candles, fetch_history, fetch_candles_history, close_exchange
from levels import get_major_levels, build_major_levels
from fvgs import get_fvgs, build_fvgs
from trademind import (
    analyze, generate_neurobro_report, STRATEGY_VERSION,
    detect_5m_ilm, confirmation_15m, find_sweep,
    get_config, measure_trend_activity, _levels_for_dir,
    _level_strength, _f, _t, _c, _h, _l, _o, _body, _range, _body_ratio,
)
from backtest_tm import run_backtest_tm, stats_report_tm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

bot = Bot(token=TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

signals_today = 0
last_signal_date = None
_recent_ready = {}
_backtest_running = False


def _sym_to_code(symbol: str) -> str:
    return symbol.replace("/", "").upper()


async def build_snapshot(symbol: str):
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
        "/debug — stage/score по всем монетам\n"
        "/probe XRPUSDT — глубокая диагностика монеты\n"
        "/backtest — бэктест 30 дней на 2 монетах\n"
        "/status — статус бота\n"
        "/id — узнать chat_id\n"
        "/test — тестовое сообщение"
    )


@dp.message(Command("scan"))
async def cmd_scan(m: Message):
    await m.answer("🔍 Сканирую 10 монет...")
    sigs = await scan_market(manual=True, notify_chat_id=m.chat.id)
    if not sigs:
        await m.answer("Сигналов READY нет.")


@dp.message(Command("debug"))
async def cmd_debug(m: Message):
    await m.answer("🔎 Собираю данные по 10 монетам... ~30 секунд.")
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


@dp.message(Command("probe"))
async def cmd_probe(m: Message):
    parts = m.text.split()
    sym_code = parts[1].upper() if len(parts) > 1 else "XRPUSDT"

    sym = sym_code.replace("USDT", "/USDT") if "/" not in sym_code else sym_code
    if sym not in SYMBOLS:
        await m.answer(f"❌ {sym_code} нет в SYMBOLS. Доступные: " + ", ".join(_sym_to_code(s) for s in SYMBOLS))
        return

    await m.answer(f"🔬 Глубокая диагностика {sym_code}...")

    try:
        c1h = await fetch_candles_history(sym, "1h", 30)
        c15 = await fetch_candles_history(sym, "15m", 30)
        c5 = await fetch_candles_history(sym, "5m", 30)
        df_m5 = await fetch_history(sym, "5m", 30)
    except Exception as e:
        await m.answer(f"❌ fetch error: {e}")
        return

    if not c1h or not c15 or not c5 or df_m5.empty:
        await m.answer("❌ нет данных")
        return

    cut_ts = int(df_m5.iloc[-1]["timestamp"])
    c1h = [x for x in c1h if x["open_time"] <= cut_ts]
    c15 = [x for x in c15 if x["open_time"] <= cut_ts]
    c5 = [x for x in c5 if x["open_time"] <= cut_ts]

    if len(c1h) < 50 or len(c15) < 30 or len(c5) < 30:
        await m.answer(f"❌ мало данных: h1={len(c1h)}, m15={len(c15)}, m5={len(c5)}")
        return

    price = float(df_m5.iloc[-1]["close"])
    levels = build_major_levels(c1h, lookback=300, max_levels=20)
    fvgs = build_fvgs(c15, lookback=150)

    lines = [f"<b>PROBE {sym_code}</b>", f"price: <code>{price:.4f}</code>",
             f"h1: {len(c1h)} | m15: {len(c15)} | m5: {len(c5)}",
             f"levels: {len(levels)} | fvgs: {len(fvgs)}\n"]

    for direction in ("LONG", "SHORT"):
        lines.append(f"<b>=== {direction} ===</b>")

        lv = _levels_for_dir(levels, direction)
        if not lv:
            lines.append("  ❌ нет уровней")
            continue

        strong_lv = [x for x in lv if _level_strength(x) >= 70]
        if not strong_lv:
            lines.append(f"  ❌ сильных уровней нет ({len(lv)} слабых)")
            continue
        lines.append(f"  ✅ уровней: {len(strong_lv)}")

        sweep = find_sweep(c1h, strong_lv, direction, config=get_config(sym_code))
        if not sweep:
            lines.append("  ❌ sweep не найден")
            continue
        lines.append(f"  ✅ sweep: level={sweep.get('level')}, extreme={sweep.get('extreme')}, depth={sweep.get('depth_pct'):.3f}%")

        conf_ok, conf_text, conf_t, bos, conf_str = confirmation_15m(c15, sweep, direction)
        if not conf_ok:
            lines.append(f"  ❌ 15M confirm не сработал (sweep_time={sweep.get('open_time')})")
            continue
        lines.append(f"  ✅ 15M: {conf_text}, bos={bos}, time={conf_t}")

        ilm_ok, ilm = detect_5m_ilm(c5, sweep, direction, conf_t, config=get_config(sym_code))
        if not ilm_ok:
            start = _f(conf_t) or _f(sweep.get("open_time"))
            after = [c for c in c5 if _t(c) is not None and start is not None and _t(c) > start]
            tail = after[-60:] if len(after) > 60 else after
            lines.append(f"  ❌ ILM не найден. M5 после conf_t: {len(after)}, в окне: {len(tail)}")
            if len(tail) >= 5:
                lines.append(f"     tail open: {_t(tail[0])}, close: {_t(tail[-1])}")
            continue
        lines.append(f"  ✅ ILM: {ilm.get('reason')}, rec={ilm.get('recovery_ratio'):.2f}, age={ilm.get('age_candles')}")

    text = "\n".join(lines)
    for i in range(0, len(text), 3500):
        await m.answer(text[i:i + 3500])


@dp.message(Command("backtest"))
async def cmd_backtest(m: Message):
    global _backtest_running
    if _backtest_running:
        await m.answer("⏳ Бэктест уже идёт.")
        return

    _backtest_running = True
    chat_id = m.chat.id
    await m.answer(
        "🧪 Бэктест TradeMind v9.41.\n"
        "30 дней, 2 монеты.\n"
        "3–7 минут."
    )

    async def progress(text):
        try:
            await bot.send_message(chat_id, text)
        except Exception:
            pass

    async def worker():
        global _backtest_running
        try:
            trades = await run_backtest_tm(SYMBOLS[:2], days=30, progress_cb=progress)
            report = stats_report_tm(trades)
            for i in range(0, len(report), 3500):
                await bot.send_message(chat_id, report[i:i + 3500])
        except Exception as e:
            await bot.send_message(chat_id, f"❌ Ошибка бэктеста: <code>{e}</code>")
        finally:
            _backtest_running = False

    asyncio.create_task(worker())


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
    await m.answer("Используй /scan, /debug, /probe, /backtest или /start.")


async def main():
    log.info(f"Старт TradeMind Bot v{STRATEGY_VERSION}...")
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(scan_market, "interval", minutes=SCAN_INTERVAL_MIN)
    scheduler.start()
    await bot.delete_webhook(drop_pending_updates=True)
    try:
        await dp.start_polling(bot)
    finally:
        await close_exchange()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Остановлен")
