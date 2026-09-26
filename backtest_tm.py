"""
Бэктест TradeMind v9.41 на 90 дней.
Прогоняет analyze() на исторических свечах Binance по 10 монетам.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from data import fetch, fetch_candles
from levels import build_major_levels
from fvgs import build_fvgs
from trademind import analyze

# издержки (доля от цены)
FEE_TAKER = 0.001
SPREAD = 0.0005
TOTAL_COST_RATIO = FEE_TAKER * 2 + SPREAD

# шаг симуляции по M5 (5 свечей = 25 минут)
STEP = 5

# лимит сделок в день на символ
MAX_TRADES_PER_DAY_PER_SYMBOL = 3

STATS = {
    "total_steps": 0,
    "analyze_calls": 0,
    "ready_found": 0,
    "trades": 0,
    "by_stage": {},
}


def _slice(candles, cut_ts):
    """Оставляет только свечи с open_time <= cut_ts."""
    if not candles:
        return []
    return [c for c in candles if c.get("open_time") is not None and c["open_time"] <= cut_ts]


def _simulate(side, entry, sl, tp, df_m5, from_idx, max_bars=600):
    """Проверяет, что сработало первым: SL или TP. Возвращает R с издержками."""
    risk = (entry - sl) if side == "LONG" else (sl - entry)
    reward = (tp - entry) if side == "LONG" else (entry - tp)
    if risk <= 0 or reward <= 0:
        return None, from_idx

    cost_in_r = (TOTAL_COST_RATIO * entry) / risk

    for i in range(from_idx + 1, min(from_idx + 1 + max_bars, len(df_m5))):
        b = df_m5.iloc[i]
        if side == "LONG":
            if b["low"] <= sl:
                return round(-1.0 - cost_in_r, 3), i
            if b["high"] >= tp:
                return round(reward / risk - cost_in_r, 3), i
        else:
            if b["high"] >= sl:
                return round(-1.0 - cost_in_r, 3), i
            if b["low"] <= tp:
                return round(reward / risk - cost_in_r, 3), i

    return round(0.0 - cost_in_r, 3), from_idx + max_bars


async def backtest_symbol(symbol, days=90, progress_cb=None):
    """Возвращает список сделок."""
    trades = []

    df_m5 = await fetch(symbol, "5m", days)
    if df_m5.empty:
        return trades

    c1h = await fetch_candles(symbol, "1h", int(days * 24 * 2))
    c15 = await fetch_candles(symbol, "15m", int(days * 24 * 4))
    c5_full = await fetch_candles(symbol, "5m", int(days * 24 * 12))

    if not c1h or not c15 or not c5_full:
        return trades

    # кэш уровней и FVG строим раз в сутки
    levels_cache = {}

    trades_per_day = {}

    for i in range(200, len(df_m5) - 1, STEP):
        STATS["total_steps"] += 1
        row = df_m5.iloc[i]
        cut_ts = int(row["timestamp"])
        cut_dt = datetime.fromtimestamp(cut_ts / 1000, tz=timezone.utc)
        day = cut_dt.date()

        # срез свечей
        c1h_sliced = _slice(c1h, cut_ts)[-500:]
        c15_sliced = _slice(c15, cut_ts)[-300:]
        c5_sliced = _slice(c5_full, cut_ts)[-300:]

        if len(c1h_sliced) < 50 or len(c15_sliced) < 30 or len(c5_sliced) < 30:
            continue

        price = float(row["close"])

        # уровни и FVG — кэш по дню + час
        cache_key = (day, cut_dt.hour)
        if cache_key not in levels_cache:
            levels = build_major_levels(c1h_sliced, lookback=300, max_levels=20)
            fvgs = build_fvgs(c15_sliced, lookback=150)
            levels_cache[cache_key] = (levels, fvgs)
        levels, fvgs = levels_cache[cache_key]

        try:
            r = analyze(
                candles_1h=c1h_sliced,
                candles_15m=c15_sliced,
                candles_5m=c5_sliced,
                current_price=price,
                major_levels=levels,
                fvgs=fvgs,
                symbol=symbol,
            )
            STATS["analyze_calls"] += 1
        except Exception:
            continue

        stage = r.get("stage", "WAIT")
        STATS["by_stage"][stage] = STATS["by_stage"].get(stage, 0) + 1

        if stage != "READY":
            continue

        STATS["ready_found"] += 1

        if trades_per_day.get(day, 0) >= MAX_TRADES_PER_DAY_PER_SYMBOL:
            continue

        entry = r.get("entry")
        sl = r.get("sl")
        tp = r.get("tp")
        direction = r.get("direction")
        if not all([entry, sl, tp, direction]):
            continue

        rr, _ = _simulate(direction, entry, sl, tp, df_m5, i)
        if rr is None:
            continue

        STATS["trades"] += 1
        trades_per_day[day] = trades_per_day.get(day, 0) + 1

        trades.append({
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "r": rr,
            "time": cut_dt.isoformat(),
            "score": r.get("score"),
            "reason": r.get("reason", "")[:120],
        })

        if progress_cb and len(trades) % 5 == 0:
            await progress_cb(
                f"  {symbol}: {len(trades)} сделок, "
                f"последняя R={rr:+.2f}"
            )

    return trades


async def run_backtest_tm(symbols, days=90, progress_cb=None):
    all_trades = []
    total = len(symbols)

    for idx, sym in enumerate(symbols, 1):
        if progress_cb:
            await progress_cb(f"📥 [{idx}/{total}] Скачиваю {sym}...")
        try:
            trades = await backtest_symbol(sym, days=days, progress_cb=progress_cb)
            all_trades.extend(trades)
            if progress_cb:
                await progress_cb(f"✅ [{idx}/{total}] {sym}: {len(trades)} сделок")
        except Exception as e:
            if progress_cb:
                await progress_cb(f"❌ [{idx}/{total}] {sym}: ошибка — {e}")

    return all_trades


def stats_report_tm(trades):
    lines = ["🔬 <b>БЭКТЕСТ TradeMind v9.41 — 90 дней</b>\n"]

    lines.append("<b>ВОРОНКА:</b>")
    lines.append(f"  всего шагов: {STATS['total_steps']}")
    lines.append(f"  вызовов analyze: {STATS['analyze_calls']}")
    lines.append(f"  найдено READY: {STATS['ready_found']}")
    lines.append(f"  симулировано сделок: {STATS['trades']}")

    lines.append("\n<b>СТАДИИ (сколько раз застали):</b>")
    for k, v in sorted(STATS["by_stage"].items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v}")

    if not trades:
        lines.append("\n❌ Сделок не найдено.")
        return "\n".join(lines)

    r_arr = np.array([t["r"] for t in trades])
    wr = (r_arr > 0).sum() / len(r_arr) * 100
    avg = r_arr.mean()
    total_r = r_arr.sum()
    sharpe = np.sqrt(len(r_arr)) * avg / r_arr.std() if r_arr.std() > 0 else 0

    longs = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    long_wr = (np.array([t["r"] for t in longs]) > 0).sum() / len(longs) * 100 if longs else 0
    short_wr = (np.array([t["r"] for t in shorts]) > 0).sum() / len(shorts) * 100 if shorts else 0

    lines.append(f"\n<b>ИТОГО:</b>")
    lines.append(f"  Сделок: <b>{len(trades)}</b>")
    lines.append(f"  Winrate: <b>{wr:.1f}%</b>")
    lines.append(f"  AvgR: <b>{avg:+.3f}</b>")
    lines.append(f"  ΣR: <b>{total_r:+.1f}</b>")
    lines.append(f"  Sharpe: <b>{sharpe:.2f}</b>")
    lines.append(f"  При риске 1% на сделку: <b>{total_r:+.1f}%</b>")

    lines.append(f"\n<b>LONG:</b> {len(longs)} сд. | WR {long_wr:.1f}%")
    lines.append(f"<b>SHORT:</b> {len(shorts)} сд. | WR {short_wr:.1f}%")

    # топ-5 монет
    by_sym = {}
    for t in trades:
        by_sym.setdefault(t["symbol"], []).append(t["r"])
    lines.append("\n<b>ПО МОНЕТАМ:</b>")
    for sym, rs in sorted(by_sym.items(), key=lambda x: -sum(x[1])):
        rs_arr = np.array(rs)
        s_wr = (rs_arr > 0).sum() / len(rs_arr) * 100
        lines.append(f"  {sym}: {len(rs)} сд. | WR {s_wr:.1f}% | ΣR {rs_arr.sum():+.1f}")

    return "\n".join(lines)
