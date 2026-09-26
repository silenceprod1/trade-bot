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
        "/scan — просканировать 10 монет\n"
        "/debug — stage/score по всем монетам\n"
        "/probe XRPUSDT — диагностика последней свечи\n"
        "/probehist XRPUSDT 30 — найти 15M_CONFIRMED в истории\n"
        "/backtest — бэктест 30 дней на 2 монетах\n"
        "/status — статус бота\n"
        "/id — chat_id\n"
        "/test — тестовое сообщение"
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
            f"   trend: {best.get('trend_activity')} | fvg_bonus: {best.get('fvg_bonus')}\n"
            f"   reason: {best.get('reason')}\n"
            f"   levels: {len(levels)} | fvgs: {len(fvgs)}"
        )
    text = "\n".join(lines)
    for i in range(0, len(text), 3500):
        await msg.answer(text[i:i + 3500])


@dp.message(Command("probe"))
async def cmd_probe(msg: Message):
    parts = msg.text.split()
    sym_code = parts[1].upper() if len(parts) > 1 else "XRPUSDT"
    sym = sym_code.replace("USDT", "/USDT") if "/" not in sym_code else sym_code
    if sym not in SYMBOLS:
        await msg.answer(f"❌ {sym_code} нет в SYMBOLS")
        return
    await msg.answer(f"🔬 Диагностика {sym_code}...")
    try:
        c1h = await fetch_candles_history(sym, "1h", 30)
        c15 = await fetch_candles_history(sym, "15m", 30)
        c5 = await fetch_candles_history(sym, "5m", 30)
        df_m5 = await fetch_history(sym, "5m", 30)
    except Exception as e:
        await msg.answer(f"❌ fetch error: {e}"); return
    if not c1h or not c15 or not c5 or df_m5.empty:
        await msg.answer("❌ нет данных"); return
    cut_ts = int(df_m5.iloc[-1]["timestamp"])
    c1h = [x for x in c1h if x["open_time"] <= cut_ts]
    c15 = [x for x in c15 if x["open_time"] <= cut_ts]
    c5 = [x for x in c5 if x["open_time"] <= cut_ts]
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
            lines.append("  ❌ нет уровней"); continue
        strong_lv = [x for x in lv if _level_strength(x) >= 70]
        if not strong_lv:
            lines.append(f"  ❌ сильных уровней нет"); continue
        lines.append(f"  ✅ уровней: {len(strong_lv)}")
        sweep = find_sweep(c1h, strong_lv, direction, config=get_config(sym_code))
        if not sweep:
            lines.append("  ❌ sweep не найден"); continue
        lines.append(f"  ✅ sweep: level={sweep.get('level')}, extreme={sweep.get('extreme')}, depth={sweep.get('depth_pct'):.3f}%")
        conf_ok, conf_text, conf_t, bos, conf_str = confirmation_15m(c15, sweep, direction)
        if not conf_ok:
            lines.append(f"  ❌ 15M не сработал"); continue
        lines.append(f"  ✅ 15M: {conf_text}, bos={bos}, time={conf_t}")
        ilm_ok, ilm = detect_5m_ilm(c5, sweep, direction, conf_t, config=get_config(sym_code))
        if not ilm_ok:
            start = _f(conf_t) or _f(sweep.get("open_time"))
            after = [c for c in c5 if _t(c) is not None and start is not None and _t(c) > start]
            lines.append(f"  ❌ ILM не найден. M5 после conf_t: {len(after)}")
            continue
        lines.append(f"  ✅ ILM: {ilm.get('reason')}, rec={ilm.get('recovery_ratio'):.2f}")
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

    await msg.answer(f"🔬 Ищу 15M_CONFIRMED в истории {sym_code} за {days} дней...")
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
                "price": price,
                "sweep": sweep,
                "conf_t": conf_t,
                "conf_text": side_r.get("confirmation"),
                "c5": c5[:],
                "stage": side_r.get("stage"),
            })
            if len(found_cases) >= 3:
                break
        if len(found_cases) >= 3:
            break

    if not found_cases:
        await msg.answer("❌ Не нашёл ни одного 15M_CONFIRMED за этот период.")
        return

    lines = [f"<b>PROBEHIST {sym_code} — найдено {len(found_cases)} случаев</b>\n"]

    for case in found_cases:
        lines.append(f"\n<b>=== {case['time']} {case['direction']} ===</b>")
        lines.append(f"stage={case['stage']} conf={case['conf_text']}")

        sweep = case["sweep"]
        conf_t = case["conf_t"]
        c5 = case["c5"]
        direction = case["direction"]

        lines.append(f"sweep: level={sweep.get('level')}, extreme={sweep.get('extreme')}")
        lines.append(f"conf_t={conf_t}")

        start = _f(conf_t) or _f(sweep.get("open_time"))
        after = [c for c in c5 if _t(c) is not None and start is not None and _t(c) > start]
        tail = after[-MAX_5M_ILM_CANDLES:]
        lines.append(f"M5 после conf_t: {len(after)}, в окне (60): {len(tail)}")

        if len(tail) < 5:
            lines.append("  ❌ слишком мало свечей")
            continue

        sl_lvl = _f(sweep.get("level"))
        config = get_config(sym_code)
        min_depth = config.get("MIN_SWEEP_DEPTH_PCT", 0.12)

        local_extremes = 0
        has_trigger = 0
        vshape_ok = 0
        close_ok = 0
        recovery_ok = 0
        distance_ok = 0

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
                local_extremes += 1
                m_range = left_ref - ml
                if m_range <= 0:
                    continue
                m_pct = m_range / left_ref * 100
                if m_pct < min_depth:
                    continue
                for j in range(i + 1, min(len(tail), i + 1 + ILM_TRIGGER_WINDOW)):
                    trig = tail[j]
                    tc = _c(trig)
                    if tc is None or not (tc > _o(trig)):
                        continue
                    if _body_ratio(trig) < MIN_BODY_RATIO_TRIGGER_5M:
                        continue
                    has_trigger += 1
                    if j >= 2:
                        h1 = _h(tail[j - 1]); h2 = _h(tail[j - 2])
                        if h1 is not None and h2 is not None and tc > max(h1, h2):
                            vshape_ok += 1
                    if tc > mh:
                        close_ok += 1
                    rec = (tc - ml) / m_range
                    if MIN_5M_RECOVERY_RATIO <= rec <= MAX_5M_RECOVERY_RATIO:
                        recovery_ok += 1
                    if sl_lvl is not None:
                        d = abs(ml - sl_lvl) / sl_lvl * 100
                        if d <= MIN_5M_ILM_SWEEP_DISTANCE_PCT:
                            distance_ok += 1
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
                local_extremes += 1
                m_range = mh - left_ref
                if m_range <= 0:
                    continue
                m_pct = m_range / mh * 100
                if m_pct < min_depth:
                    continue
                for j in range(i + 1, min(len(tail), i + 1 + ILM_TRIGGER_WINDOW)):
                    trig = tail[j]
                    tc = _c(trig)
                    if tc is None or not (tc < _o(trig)):
                        continue
                    if _body_ratio(trig) < MIN_BODY_RATIO_TRIGGER_5M:
                        continue
                    has_trigger += 1
                    if j >= 2:
                        l1 = _l(tail[j - 1]); l2 = _l(tail[j - 2])
                        if l1 is not None and l2 is not None and tc < min(l1, l2):
                            vshape_ok += 1
                    if tc < ml:
                        close_ok += 1
                    rec = (mh - tc) / m_range
                    if MIN_5M_RECOVERY_RATIO <= rec <= MAX_5M_RECOVERY_RATIO:
                        recovery_ok += 1
                    if sl_lvl is not None:
                        d = abs(mh - sl_lvl) / sl_lvl * 100
                        if d <= MIN_5M_ILM_SWEEP_DISTANCE_PCT:
                            distance_ok += 1
                    break

        lines.append(f"  локальных экстремумов: {local_extremes}")
        lines.append(f"  trigger-свечей (body≥{MIN_BODY_RATIO_TRIGGER_5M}): {has_trigger}")
        lines.append(f"  V-shape (close за max2/min2): {vshape_ok}")
        lines.append(f"  close прошёл экстремум: {close_ok}")
        lines.append(f"  recovery в [0.30;1.30]: {recovery_ok}")
        lines.append(f"  расстояние до sweep ≤ {MIN_5M_ILM_SWEEP_DISTANCE_PCT}%: {distance_ok}")

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
    await msg.answer("🧪 Бэктест TradeMind v9.41. 30 дней, 2 монеты. 3–7 минут.")

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
    await msg.answer(f"✅ v{STRATEGY_VERSION}\n🕒 {now}\n📊 {signals_today}/{MAX_SIGNALS_PER_DAY}")


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
    await msg.answer("Используй /scan, /debug, /probe, /probehist, /backtest.")


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
