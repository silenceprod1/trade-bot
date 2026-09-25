import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta, timezone

exchange = ccxt.binance({"enableRateLimit": True})

STATS = {
    "total_bars": 0,
    "setup_a_checked": 0,
    "setup_a_asia_missing": 0,
    "setup_a_range_wide": 0,
    "setup_a_no_break": 0,
    "setup_a_no_retest": 0,
    "setup_a_low_volume": 0,
    "setup_a_rsi_block": 0,
    "setup_c_checked": 0,
    "setup_c_adx_low": 0,
    "setup_c_no_trend": 0,
    "setup_c_far_from_ema": 0,
    "setup_c_weak_wick": 0,
    "setup_c_session": 0,
    "setup_c_no_confirm": 0,
    "setup_c_atr": 0,
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


def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def rsi(df, p=14):
    d = df["close"].diff()
    gain = d.clip(lower=0).rolling(p).mean()
    loss = (-d.clip(upper=0)).rolling(p).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()


def adx(df, p=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_ = tr.rolling(p).mean()
    pdi = 100 * pd.Series(plus, index=df.index).rolling(p).mean() / atr_
    mdi = 100 * pd.Series(minus, index=df.index).rolling(p).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.rolling(p).mean()


def setup_a(df_m5, df_m15):
    STATS["setup_a_checked"] += 1
    if len(df_m5) < 50 or len(df_m15) < 30:
        return None
    today = df_m5["datetime"].dt.date.iloc[-1]
    asia = df_m5[(df_m5["datetime"].dt.date == today) & (df_m5["datetime"].dt.hour < 7)]
    if len(asia) < 6:
        STATS["setup_a_asia_missing"] += 1
        return None
    hi, lo = asia["high"].max(), asia["low"].min()
    if (hi - lo) > 0.05 * hi:
        STATS["setup_a_range_wide"] += 1
        return None
    last, prev = df_m5.iloc[-1], df_m5.iloc[-2]
    vol_avg = df_m5["volume"].tail(20).mean() or 1
    r = rsi(df_m15).iloc[-1]

    broke_up = prev["close"] > hi
    broke_down = prev["close"] < lo
    if not (broke_up or broke_down):
        STATS["setup_a_no_break"] += 1
        return None

    if broke_up:
        if not (last["low"] <= hi and last["close"] > hi):
            STATS["setup_a_no_retest"] += 1
            return None
        if last["volume"] <= 1.3 * vol_avg:
            STATS["setup_a_low_volume"] += 1
            return None
        if r >= 70:
            STATS["setup_a_rsi_block"] += 1
            return None
        return {"side": "BUY", "entry": last["close"], "sl": lo,
                "tp": last["close"] + 2.5 * (last["close"] - lo), "setup": "A"}

    if broke_down:
        if not (last["high"] >= lo and last["close"] < lo):
            STATS["setup_a_no_retest"] += 1
            return None
        if last["volume"] <= 1.3 * vol_avg:
            STATS["setup_a_low_volume"] += 1
            return None
        if r <= 30:
            STATS["setup_a_rsi_block"] += 1
            return None
        return {"side": "SELL", "entry": last["close"], "sl": hi,
                "tp": last["close"] - 2.5 * (hi - last["close"]), "setup": "A"}
    return None


def setup_c(df_h1, df_d1):
    """Сетап C: трендовый откат с подтверждением. TP = 3R."""
    STATS["setup_c_checked"] += 1
    if len(df_h1) < 220 or len(df_d1) < 200:
        return None

    ema200_d1 = ema(df_d1["close"], 200).iloc[-1]
    price_d1 = df_d1["close"].iloc[-1]
    up = price_d1 > ema200_d1
    down = price_d1 < ema200_d1
    if not (up or down):
        STATS["setup_c_no_trend"] += 1
        return None

    ax = adx(df_h1, 14).iloc[-1]
    if pd.isna(ax) or ax < 25:
        STATS["setup_c_adx_low"] += 1
        return None

    a_series = atr(df_h1, 14)
    a = a_series.iloc[-1]
    a_avg = a_series.tail(100).mean()
    if pd.isna(a) or pd.isna(a_avg) or a_avg == 0:
        STATS["setup_c_atr"] += 1
        return None
    if not (0.5 * a_avg <= a <= 2.0 * a_avg):
        STATS["setup_c_atr"] += 1
        return None

    ema50 = ema(df_h1["close"], 50).iloc[-2]
    prev = df_h1.iloc[-2]
    last = df_h1.iloc[-1]

    h = last["datetime"].hour
    if not (7 <= h < 20):
        STATS["setup_c_session"] += 1
        return None

    touch_buy = up and prev["low"] <= ema50 * 1.002
    touch_sell = down and prev["high"] >= ema50 * 0.998

    if not (touch_buy or touch_sell):
        STATS["setup_c_far_from_ema"] += 1
        return None

    if touch_buy:
        confirm = last["close"] > last["open"] and last["close"] > prev["high"]
        if not confirm:
            STATS["setup_c_no_confirm"] += 1
            return None
        sl = min(prev["low"], last["low"]) - 0.2 * a
        entry = last["close"]
        risk = entry - sl
        tp = entry + 3.0 * risk      # было 1.5R, теперь 3R
        return {"side": "BUY", "entry": entry, "sl": sl, "tp": tp, "setup": "C"}

    if touch_sell:
        confirm = last["close"] < last["open"] and last["close"] < prev["low"]
        if not confirm:
            STATS["setup_c_no_confirm"] += 1
            return None
        sl = max(prev["high"], last["high"]) + 0.2 * a
        entry = last["close"]
        risk = sl - entry
        tp = entry - 3.0 * risk      # было 1.5R, теперь 3R
        return {"side": "SELL", "entry": entry, "sl": sl, "tp": tp, "setup": "C"}

    STATS["setup_c_weak_wick"] += 1
    return None


def simulate(side, entry, sl, tp, df_m5, from_idx, max_bars=300):
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
        df_m15 = await fetch_all(sym, "15m", days)
        df_h1 = await fetch_all(sym, "1h", days)
        df_d1 = await fetch_all(sym, "1d", max(days, 300))

        if df_m5.empty or df_h1.empty:
            continue

        if progress_cb:
            await progress_cb(
                f"🔬 [{idx}/{total}] Прогоняю {sym} "
                f"(M5: {len(df_m5)}, H1: {len(df_h1)}, D1: {len(df_d1)})..."
            )

        for i in range(50, len(df_m5) - 1, 3):
            STATS["total_bars"] += 1
            win_m5 = df_m5.iloc[:i + 1]
            cut = win_m5["datetime"].iloc[-1]
            win_m15 = df_m15[df_m15["datetime"] <= cut].tail(50)
            win_h1 = df_h1[df_h1["datetime"] <= cut].tail(220)
            win_d1 = df_d1[df_d1["datetime"] <= cut].tail(200)
            if len(win_m15) < 30 or len(win_h1) < 220 or len(win_d1) < 200:
                continue

            sig = None
            h = win_m5["datetime"].iloc[-1].hour
            if 7 <= h < 10:
                sig = setup_a(win_m5, win_m15)
            if not sig:
                sig = setup_c(win_h1, win_d1)

            if sig:
                STATS["signals"] += 1
                r, _ = simulate(sig["side"], sig["entry"], sig["sl"], sig["tp"], df_m5, i)
                if r is None:
                    continue
                STATS["simulated"] += 1
                trades.append({
                    "symbol": sym, "setup": sig["setup"], "side": sig["side"],
                    "r": r, "time": win_m5["datetime"].iloc[-1],
                })

    return trades


def stats_report():
    return (
        "🔍 <b>ВОРОНКА ОТСЕВА</b>\n"
        f"Всего проверок: {STATS['total_bars']}\n\n"
        f"<b>Сетап A</b> (проверок: {STATS['setup_a_checked']}):\n"
        f"  нет азиатского диапазона: {STATS['setup_a_asia_missing']}\n"
        f"  диапазон больше 5%: {STATS['setup_a_range_wide']}\n"
        f"  нет пробоя: {STATS['setup_a_no_break']}\n"
        f"  нет ретеста: {STATS['setup_a_no_retest']}\n"
        f"  слабый объём: {STATS['setup_a_low_volume']}\n"
        f"  RSI блок: {STATS['setup_a_rsi_block']}\n\n"
        f"<b>Сетап C</b> (проверок: {STATS['setup_c_checked']}):\n"
        f"  ADX меньше 25: {STATS['setup_c_adx_low']}\n"
        f"  нет тренда: {STATS['setup_c_no_trend']}\n"
        f"  ATR вне коридора: {STATS['setup_c_atr']}\n"
        f"  вне сессии: {STATS['setup_c_session']}\n"
        f"  далеко от EMA50: {STATS['setup_c_far_from_ema']}\n"
        f"  нет подтверждения: {STATS['setup_c_no_confirm']}\n\n"
        f"<b>Найдено сигналов:</b> {STATS['signals']}\n"
        f"<b>Сделок:</b> {STATS['simulated']}"
    )
