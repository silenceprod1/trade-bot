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
    MIN_BODY_RATIO_TRIGGER_5M_PATCH,
    CLOSE_BREAK_FRACTION,
    VSHAPE_TOLERANCE,
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
        f"👋 <b>TradeMind Bot v{STRATEGY_VERSION}</b>\n\n"
        "Команды:\n"
        "/scan, /debug, /probe SYMBOL, /probehist SYMBOL DAYS,\n"
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
    lines = [f"<b>DEBUG TradeMind v{STRATEGY_VERSION}</b>\n"]
    for sym in SYMBOLS:
        code = _sym_to_code(sym)
        try:
            snap = await build_snapshot(sym)
        except Exception as e:
            lines.append(f"❌ {sym}: {e}"); continue
        if not snap:
            lines.append(f"❌ {sym}: нет данных"); continue
        price, c1h, c15, c5, levels, fvgs = snap
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
            f"   dir: {best.get('direction')} | "
            f"conf: {best.get('confirmation')} | bos: {best.get('bos')}\n"
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

    await msg.answer(f"🔬 Глубокая диагностика ILM по {sym_code} за {days} дней...")
    try:
        c1h_full = await fetch_candles_history(sym, "1h", days)
        c15_full = await fetch_candles_history(sym, "15m", days)
        c5_full = await fetch_candles_history(sym, "5m", days)
        df_m5 = await fetch_history(sym, "5m", days)
    except Exception as e:
        await msg.answer(f"❌ fetch error: {e}"); return
    if df_m5.empty:
        await msg.answer("❌ df_m5 пуст"); return

    found_cases = []
    ready_found = 0

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

        try:
            r = analyze(candles_1h=c1h, candles_15m=c15, candles_5m=c5,
                        current_price=price, major_levels=levels,
                        fvgs=fvgs, symbol=sym_code)
        except Exception:
            continue

        if r.get("stage") == "READY":
            ready_found += 1

        for direction in ("LONG", "SHORT"):
            side_r = r.get("long" if direction == "LONG" else "short") or {}
            if side_r.get("stage") not in ("15M_CONFIRMED", "READY"):
                continue
            sweep = side_r.get("sweep")
            conf_t = side_r.get("confirmation_15m_time")
            if not sweep or not conf_t:
                continue
            found_cases.append({
                "time": cut_dt.isoformat(),
                "direction": direction,
                "stage": side_r.get("stage"),
                "conf_text": side_r.get("confirmation"),
                "sweep": sweep,
                "conf_t": conf_t,
                "c5": c5[:],
            })
            if len(found_cases) >= 1:  # только первый случай, чтобы влезло
                break
        if len(found_cases) >= 1:
            break

    lines = [f"<b>DIAG {sym_code} — READY={ready_found}</b>\n"]
    if not found_cases:
        await msg.answer("❌ Не нашёл 15M_CONFIRMED.")
        return

    case = found_cases[0]
    lines.append(f"<b>{case['time']} {case['direction']} {case['stage']}</b>")
    lines.append(f"conf={case['conf_text']}")

    sweep = case["sweep"]
    conf_t = case["conf_t"]
    c5 = case["c5"]
    direction = case["direction"]

    lines.append(f"sweep: level={sweep.get('level')}, extreme={sweep.get('extreme')}")
    lines.append(f"conf_t={conf_t}")

    start = _f(conf_t) or _f(sweep.get("open_time"))
    after = [c for c in c5 if _t(c) is not None and start is not None and _t(c) > start]
    tail = after[-MAX_5M_ILM_CANDLES:]
    lines.append(f"M5 после conf_t: {len(after)}, окно: {len(tail)}\n")

    if len(tail) < 5:
        lines.append("❌ мало свечей")
        await msg.answer("\n".join(lines))
        return

    sl_lvl = _f(sweep.get("level"))
    config = get_config(sym_code)
    min_depth = config.get("MIN_SWEEP_DEPTH_PCT", 0.12)

    # Разбираем по каждой i, где есть local extreme
    counter = 0
    for i in range(2, len(tail) - 2):
        mc = tail[i]
        if direction == "LONG":
            ml = _l(mc); mh = _h(mc)
            if ml is None or mh is None:
                continue
            before = tail[max(0, i - 2):i]
            bl = [_l(x) for x in before if _l(x) is not None]
            if not bl:
                continue
            left_ref = min(bl)
            if left_ref <= ml:
                continue
            m_range = left_ref - ml
            if m_range <= 0:
                continue
            m_pct = m_range / left_ref * 100
            if m_pct < min_depth:
                continue
            # нашли local extreme
            mid_threshold = mh - (mh - ml) * CLOSE_BREAK_FRACTION
            for j in range(i + 1, min(len(tail), i + 1 + ILM_TRIGGER_WINDOW)):
                trig = tail[j]
                tc = _c(trig); to = _o(trig)
                if tc is None or to is None or not (tc > to):
                    continue
                br = _body_ratio(trig)
                h1 = _h(tail[j - 1]) if j >= 1 else None
                h2 = _h(tail[j - 2]) if j >= 2 else None
                vthr = max(h1, h2) * (1 - VSHAPE_TOLERANCE) if h1 is not None and h2 is not None else None
                counter += 1
                lines.append(
                    f"  T{counter}: ml={ml:.4f} mh={mh:.4f} mid={mid_threshold:.4f}\n"
                    f"     tc={tc:.4f} to={to:.4f} body={br:.2f}\n"
                    f"     vthr={vthr} (max2h)\n"
                    f"     body>=0.35? {br >= MIN_BODY_RATIO_TRIGGER_5M_PATCH}\n"
                    f"     vshape? {vthr is not None and tc > vthr}\n"
                    f"     close>=mid? {tc >= mid_threshold}"
                )
                if counter >= 5:
                    break
            if counter >= 5:
                break
        else:
            mh = _h(mc); ml = _l(mc)
            if mh is None or ml is None:
                continue
            before = tail[max(0, i - 2):i]
            bh = [_h(x) for x in before if _h(x) is not None]
            if not bh:
                continue
            left_ref = max(bh)
            if mh <= left_ref:
                continue
            m_range = mh - left_ref
            if m_range <= 0:
                continue
            m_pct = m_range / mh * 100
            if m_pct < min_depth:
                continue
            mid_threshold = ml + (mh - ml) * CLOSE_BREAK_FRACTION
            for j in range(i + 1, min(len(tail), i + 1 + ILM_TRIGGER_WINDOW)):
                trig = tail[j]
                tc = _c(trig); to = _o(trig)
                if tc is None or to is None or not (tc < to):
                    continue
                br = _body_ratio(trig)
                l1 = _l(tail[j - 1]) if j >= 1 else None
                l2 = _l(tail[j - 2]) if j >= 2 else None
                vthr = min(l1, l2) * (1 + VSHAPE_TOLERANCE) if l1 is not None and l2 is not None else None
                counter += 1
                lines.append(
                    f"  T{counter}: mh={mh:.4f} ml={ml:.4f} mid={mid_threshold:.4f}\n"
                    f"     tc={tc:.4f} to={to:.4f} body={br:.2f}\n"
                    f"     vthr={vthr} (min2l)\n"
                    f"     body>=0.35? {br >= MIN_BODY_RATIO_TRIGGER_5M_PATCH}\n"
                    f"     vshape? {vthr is not None and tc < vthr}\n"
                    f"     close<=mid? {tc <= mid_threshold}"
                )
                if counter >= 5:
                    break
            if counter >= 5:
                break
        if counter >= 5:
            break

    if counter == 0:
        lines.append("  ❌ нет ни одного trigger-кандидата вообще")

    text = "\n".join(lines)
    for i in range(0, len(text), 3500):
        await msg.answer(text[i:i + 3500])


@dp.message(Command("backtest"))
async def cmd_backtest(msg: Message):
    global _backtest_running
    if _backtest_running:
        await msg.answer("⏳ Бэктест уже идёт.")
        return
    _backtest_running = True
    chat_id = msg.chat.id
    await msg.answer("🧪 Бэктест TradeMind v9.41 (patched ILM). 30 дней, 2 монеты.")

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
            await bot.send_message(chat_id, f"❌ Ошибка: <code>{e}</code>")
        finally:
            _backtest_running = False

    asyncio.create_task(worker())


@dp.message(Command("status"))
async def cmd_status(msg: Message):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    await msg.answer(f"✅ v{STRATEGY_VERSION} (patched)\n🕒 {now}\n📊 {signals_today}/{MAX_SIGNALS_PER_DAY}")


@dp.message(Command("id"))
async def cmd_id(msg: Message):
    await msg.answer(f"chat_id: <code>{msg.chat.id}</code>")


@dp.message(Command("test"))
async def cmd_test(msg: Message):
    fake = {"stage": "READY", "direction": "LONG", "score": 88,
            "reason": "test", "entry": 1.2345, "sl": 1.2200, "tp": 1.2635}
    await msg.answer(generate_neurobro_report(fake, "TESTUSDT", risk_pct=1.0))


@dp.message(F.text)
async def echo(msg: Message):
    await msg.answer("Используй /scan, /debug, /probehist, /backtest.")


async def main():
    log.info(f"Старт TradeMind Bot v{STRATEGY_VERSION} (patched ILM)...")
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
