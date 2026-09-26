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
    ILM_TRIGGER_WINDOW, MIN_BODY_RATIO_TRIGGER_5M,
    MIN_5M_RECOVERY_RATIO, MAX_5M_RECOVERY_RATIO,
    MAX_5M_ILM_CANDLES, MIN_5M_ILM_SWEEP_DISTANCE_PCT,
)
from ilm_patch import (
    apply_patch as apply_ilm_patch,
    detect_5m_ilm_patched,
    confirmation_15m_patched,
    set_cut_ts,
)
apply_ilm_patch()
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

        # реальное время: cut_ts = время последней M5-свечи
        last_ts = int(c5[-1]["open_time"]) if c5 else None
        if last_ts:
            set_cut_ts(last_ts)

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
async def cmd_start(msg: Message):
    await msg.answer(
        f"👋 <b>TradeMind Bot v{STRATEGY_VERSION} (patched)</b>\n\n"
        "Команды:\n"
        "/scan, /debug, /probehist SYMBOL DAYS,\n"
        "/backtest, /status, /id, /test"
    )


@dp.message(Command("scan"))
async def cmd_scan(msg: Message):
    await msg.answer("🔍 Сканирую 10 монет...")
    sigs = await scan_market(manual=True, notify_chat_id=msg.chat.id)
    if not sigs:
        await msg.answer("Сигналов READY нет.")


@dp.message(Command("debug"))
async def cmd_debug(msg: Message):
    await msg.answer("🔎 Собираю данные... ~30 секунд.")
    lines = [f"<b>DEBUG TradeMind v{STRATEGY_VERSION} (patched)</b>\n"]
    for sym in SYMBOLS:
        code = _sym_to_code(sym)
        try:
            snap = await build_snapshot(sym)
        except Exception as e:
            lines.append(f"❌ {sym}: {e}"); continue
        if not snap:
            lines.append(f"❌ {sym}: нет данных"); continue
        price, c1h, c15, c5, levels, fvgs = snap
        last_ts = int(c5[-1]["open_time"]) if c5 else None
        if last_ts:
            set_cut_ts(last_ts)
        try:
            r = analyze(candles_1h=c1h, candles_15m=c15, candles_5m=c5,
                        current_price=price, major_levels=levels,
                        fvgs=fvgs, symbol=code)
        except Exception as e:
            lines.append(f"❌ {sym}: {e}"); continue
        long_r = r.get("long") or {}
        short_r = r.get("short") or {}
        best = long_r if long_r.get("score", 0) >= short_r.get("score", 0) else short_r
        lines.append(
            f"\n📊 <b>{code}</b> | price <code>{price:.4f}</code>\n"
            f"   context: {r.get('context_direction')} | "
            f"stage: <b>{best.get('stage')}</b> | score: <b>{best.get('score')}</b>\n"
            f"   dir: {best.get('direction')} | conf: {best.get('confirmation')} | bos: {best.get('bos')}\n"
            f"   reason: {best.get('reason')}\n"
            f"   levels: {len(levels)} | fvgs: {len(fvgs)}"
        )
    text = "\n".join(lines)
    for i in range(0, len(text), 3500):
        await msg.answer(text[i:i + 3500])


@dp.message(Command("probehist"))
async def cmd_probehist(msg: Message):
    parts = msg.text.split()
    sym_code = parts[1].upper() if len(parts) > 1 else "XRPUSDT"
    days = int(parts[2]) if len(parts) > 2 else 30
    sym = sym_code.replace("USDT", "/USDT") if "/" not in sym_code else sym_code
    if sym not in SYMBOLS:
        await msg.answer(f"❌ {sym_code} нет в SYMBOLS")
        return

    await msg.answer(f"🔬 Диагностика ILM {sym_code} за {days} дней (patched v3)...")
    try:
        c1h_full = await fetch_candles_history(sym, "1h", days)
        c15_full = await fetch_candles_history(sym, "15m", days)
        c5_full = await fetch_candles_history(sym, "5m", days)
        df_m5 = await fetch_history(sym, "5m", days)
    except Exception as e:
        await msg.answer(f"❌ fetch error: {e}"); return
    if df_m5.empty:
        await msg.answer("❌ df_m5 пуст"); return

    ready_found = 0
    first_case = None

    for i in range(200, len(df_m5) - 1, 15):
        row = df_m5.iloc[i]
        cut_ts = int(row["timestamp"])
        cut_dt = datetime.fromtimestamp(cut_ts / 1000, tz=timezone.utc)

        c1h = [x for x in c1h_full if x["open_time"] <= cut_ts][-500:]
        c15 = [x for x in c15_full if x["open_time"] <= cut_ts][-300:]
        c5 = [x for x in c5_full if x["open_time"] <= cut_ts][-300:]
        if len(c1h) < 50 or len(c15) < 30 or len(c5) < 30:
            continue

        price = float(row["close"])
        levels = build_major_levels(c1h, lookback=300, max_levels=20)
        fvgs = build_fvgs(c15, lookback=150)

        set_cut_ts(cut_ts)

        try:
            r = analyze(candles_1h=c1h, candles_15m=c15, candles_5m=c5,
                        current_price=price, major_levels=levels,
                        fvgs=fvgs, symbol=sym_code)
        except Exception:
            continue

        if r.get("stage")
