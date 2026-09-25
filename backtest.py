import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta, timezone

exchange = ccxt.binance({"enableRateLimit": True})

FEE_TAKER = 0.001
SPREAD = 0.0005
TOTAL_COST_RATIO = FEE_TAKER * 2 + SPREAD

MAX_TRADES_PER_DAY_PER_SYMBOL = 2

STATS = {
    "total_bars": 0,
    "setup_b_checked": 0,
    "setup_b_atr_out": 0,
    "setup_b_adx_low": 0,
    "setup_b_no_level": 0,
    "setup_b_no_fakeout": 0,
    "setup_b_session": 0,
    "daily_limit_skipped": 0,
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


def setup_b(df_h4):
    """Ложный пробой H4-уровня. Возвращает сигнал или None."""
    STATS["setup_b_checked"] += 1

    if len(df_h4) < 120:
        return None

    a_series = atr(df_h4, 14)
    a = a_series.iloc[-1]
    a_avg = a_series.tail(100).mean()
    if pd.isna(a) or pd.isna(a_avg) or a_avg == 0:
        STATS["setup_b_atr_out"] += 1
        return None
    if not (0.5 * a_avg <= a <= 2.0 * a_avg):
        STATS["setup_b_atr_out"] += 1
        return None

    ax = adx(df_h4, 14).iloc[-1]
    if pd.isna(ax) or ax < 15:
        STATS["setup_b_adx_low"] += 1
        return None

    # уровень = high/low за 20 свечей до последней закрытой
    window = df_h4.iloc[-22:-2]
    if len(window) < 10:
        STATS["setup_b_no_level"] += 1
        return None
    resistance = window["high"].max()
    support = window["low"].min()

    # последняя закрытая свеча H4
    bar = df_h4.iloc[-2]

    # ложный пробой сопротивления: high > resistance, close < resistance
    fake_up = bar["high"] > resistance and bar["close"] < resistance
    # ложный пробой поддержки: low < support, close > support
    fake_down = bar["low"] < support and bar["close"] > support

    if not (fake_up or fake_down):
        STATS["setup_b_no_fakeout"] += 1
        return None

    if fake_up:
        # вход SELL на открытии следующей свечи (индекс -1)
        entry_bar = df_h4.iloc[-1]
        entry = entry_bar["open"]
        sl = bar["high"] + 0.3 * a
        risk = sl - entry
        tp = entry - 2.0 * risk
        return {"side": "SELL", "entry": entry, "sl": sl, "tp": tp, "setup": "B"}

    if fake_down:
        entry_bar = df_h4.iloc[-1]
        entry = entry_bar["open"]
        sl = bar["low"] - 0.3 * a
        risk = entry - sl
        tp = entry + 2.0 * risk
        return {"side": "BUY", "entry": entry, "sl": sl, "tp": tp, "setup": "B"}

    return None


def simulate(side, entry, sl, tp, df_m5, from_idx, max_bars=600):
    risk = (entry - sl) if side == "BUY" else (sl - entry)
    reward = (tp - entry) if side == "BUY" else (entry - tp)
    if risk <= 0 or reward <= 0:
        return None, from_idx

    cost_in_r = (TOTAL_COST_RATIO * entry) / risk

    for i in range(from_idx + 1, min(from_idx + 1 + max_bars, len(df_m5))):
        b = df_m5.iloc[i]
        if side == "BUY":
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

        trades_per_day = {}

        for i in range(50, len(df_m5) - 1, 6):
            STATS["total_bars"] += 1
            cut = df_m5["datetime"].iloc[i]

            # берём H4-свечи, закрытые ДО момента cut
            win_h4 = df_h4[df_h4["datetime"] <= cut].tail(120)
            if len(win_h4) < 120:
                continue

            sig = setup_b(win_h4)
            if not sig:
                continue

            day = cut.date()
            if trades_per_day.get(day, 0) >= MAX_TRADES_PER_DAY_PER_SYMBOL:
                STATS["daily_limit_skipped"] += 1
                continue

            STATS["signals"] += 1
            r, _ = simulate(sig["side"], sig["entry"], sig["sl"], sig["tp"], df_m5, i)
            if r is None:
                continue
            STATS["simulated"] += 1
            trades_per_day[day] = trades_per_day.get(day, 0) + 1
            trades.append({
                "symbol": sym, "setup": sig["setup"], "side": sig["side"],
                "r": r, "time": cut,
            })

    return trades


def stats_report():
    return (
        "🔍 <b>ВОРОНКА ОТСЕВА — СЕТАП B</b>\n"
        f"Всего проверок: {STATS['total_bars']}\n"
        f"Отсечено лимитом дня: {STATS['daily_limit_skipped']}\n\n"
        f"<b>Сетап B</b> (проверок: {STATS['setup_b_checked']}):\n"
        f"  ATR вне коридора: {STATS['setup_b_atr_out']}\n"
        f"  ADX меньше 15: {STATS['setup_b_adx_low']}\n"
        f"  нет уровня: {STATS['setup_b_no_level']}\n"
        f"  нет ложного пробоя: {STATS['setup_b_no_fakeout']}\n\n"
        f"<b>Найдено сигналов:</b> {STATS['signals']}\n"
        f"<b>Сделок:</b> {STATS['simulated']}"
    )
