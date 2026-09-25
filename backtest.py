import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta, timezone

exchange = ccxt.binance({"enableRateLimit": True})

# ОТКЛЮЧАЕМ ИЗДЕРЖКИ ДЛЯ ДИАГНОСТИКИ
FEE_TAKER = 0.0
SPREAD = 0.0
TOTAL_COST_RATIO = 0.0

STATS = {
    "total_bars": 0,
    "setup_b_checked": 0,
    "fakeouts_found": 0,
    "buy_signals": 0,
    "sell_signals": 0,
    "buy_wins": 0,
    "sell_wins": 0,
    "buy_r_sum": 0.0,
    "sell_r_sum": 0.0,
    "signals": 0,
    "simulated": 0,
}


async def fetch_all(symbol, timeframe, days):
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )
    now = exchange.milliseconds()
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    all_data = []

    while since < now:
        try:
            batch = await exchange.fetch_ohlcv(symbol, timeframe, since, 1000)
        except Exception:
            await asyncio.sleep(2)
            continue
        if not batch:
            break
        all_data.extend(batch)
        since = batch[-1][0] + tf_ms
        await asyncio.sleep(0.05)

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()


def setup_b_raw(df_h4):
    """
    БЕЗ ФИЛЬТРОВ. Просто ищем любой ложный пробой.
    Вход по open СЛЕДУЮЩЕЙ H4-свечи, которая ещё не закрыта на момент cut.
    """
    STATS["setup_b_checked"] += 1
    if len(df_h4) < 30:
        return None

    # уровень — по 20 свечам ДО последней закрытой
    window = df_h4.iloc[-22:-2]
    if len(window) < 10:
        return None
    resistance = window["high"].max()
    support = window["low"].min()

    bar = df_h4.iloc[-2]           # последняя закрытая H4

    fake_up = bar["high"] > resistance and bar["close"] < resistance
    fake_down = bar["low"] < support and bar["close"] > support

    if not (fake_up or fake_down):
        return None

    STATS["fakeouts_found"] += 1

    a = atr(df_h4, 14).iloc[-1]
    if pd.isna(a) or a <= 0:
        a = (bar["high"] - bar["low"]) or 1.0

    # вход по open следующей H4-свечи (индекс -1)
    entry = df_h4.iloc[-1]["open"]

    if fake_up:
        sl = bar["high"] + 0.3 * a
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - 2.0 * risk
        STATS["sell_signals"] += 1
        return {"side": "SELL", "entry": entry, "sl": sl, "tp": tp, "setup": "B"}

    if fake_down:
        sl = bar["low"] - 0.3 * a
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + 2.0 * risk
        STATS["buy_signals"] += 1
        return {"side": "BUY", "entry": entry, "sl": sl, "tp": tp, "setup": "B"}

    return None


def simulate(side, entry, sl, tp, df_m5, from_idx, max_bars=600):
    """Без издержек. Чистый R."""
    risk = (entry - sl) if side == "BUY" else (sl - entry)
    reward = (tp - entry) if side == "BUY" else (entry - tp)
    if risk <= 0 or reward <= 0:
        return None, from_idx

    for i in range(from_idx + 1, min(from_idx + 1 + max_bars, len(df_m5))):
        b = df_m5.iloc[i]
        if side == "BUY":
            if b["low"] <= sl:
                return -1.0, i
            if b["high"] >= tp:
                return round(reward / risk, 2), i
        else:
            if b["high"] >= sl:
                return -1.0, i
            if b["low"] <= tp:
                return round(reward / risk, 2), i
    return 0.0, from_idx + max_bars


async def run_backtest(symbols, days=90, progress_cb=None):
    trades = []
    total = len(symbols)
    for idx, sym in enumerate(symbols, 1):
        if progress_cb:
            await progress_cb(f"📥 [{idx}/{total}] Скачиваю {sym}...")
        df_m5 = await fetch_all(sym, "5m", days)
        df_h4 = await fetch_all(sym, "4h", max(days, 200))

        if df_m5.empty or df_h4.empty:
            continue

        if progress_cb:
            await progress_cb(
                f"🔬 [{idx}/{total}] Прогоняю {sym} "
                f"(M5: {len(df_m5)}, H4: {len(df_h4)})..."
            )

        for i in range(50, len(df_m5) - 1, 6):
            STATS["total_bars"] += 1
            cut = df_m5["datetime"].iloc[i]

            win_h4 = df_h4[df_h4["datetime"] <= cut].tail(30)
            if len(win_h4) < 30:
                continue

            sig = setup_b_raw(win_h4)
            if not sig:
                continue

            STATS["signals"] += 1
            r, _ = simulate(sig["side"], sig["entry"], sig["sl"], sig["tp"], df_m5, i)
            if r is None:
                continue
            STATS["simulated"] += 1

            if sig["side"] == "BUY":
                STATS["buy_r_sum"] += r
                if r > 0:
                    STATS["buy_wins"] += 1
            else:
                STATS["sell_r_sum"] += r
                if r > 0:
                    STATS["sell_wins"] += 1

            trades.append({
                "symbol": sym, "setup": sig["setup"], "side": sig["side"],
                "r": r, "time": cut,
            })

    return trades


def stats_report():
    buy_total = STATS["buy_signals"]
    sell_total = STATS["sell_signals"]
    buy_wr = (STATS["buy_wins"] / buy_total * 100) if buy_total else 0
    sell_wr = (STATS["sell_wins"] / sell_total * 100) if sell_total else 0
    return (
        "🔍 <b>ДИАГНОСТИКА — БЕЗ ФИЛЬТРОВ, БЕЗ ИЗДЕРЖЕК</b>\n"
        f"Всего проверок: {STATS['total_bars']}\n"
        f"Найдено ложных пробоев: {STATS['fakeouts_found']}\n"
        f"Сделок: {STATS['simulated']}\n\n"
        f"<b>BUY</b>: {buy_total} сигналов | WR {buy_wr:.1f}% | ΣR {STATS['buy_r_sum']:+.1f}\n"
        f"<b>SELL</b>: {sell_total} сигналов | WR {sell_wr:.1f}% | ΣR {STATS['sell_r_sum']:+.1f}"
    )
