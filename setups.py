from dataclasses import dataclass
from typing import Optional
import pandas as pd

from indicators import rsi, ema, atr, adx


@dataclass
class Signal:
    symbol: str
    side: str
    entry: float
    sl: float
    tp: float
    setup: str
    reason: str


def setup_a_asia_breakout(symbol: str, df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> Optional[Signal]:
    if len(df_m5) < 50 or len(df_m15) < 30:
        return None

    today = df_m5["datetime"].dt.date.iloc[-1]
    asia = df_m5[
        (df_m5["datetime"].dt.date == today) &
        (df_m5["datetime"].dt.hour < 7)
    ]
    if len(asia) < 6:
        return None

    hi, lo = asia["high"].max(), asia["low"].min()
    rng = hi - lo
    if rng > 0.03 * hi:
        return None

    last = df_m5.iloc[-1]
    prev = df_m5.iloc[-2]
    vol_avg = df_m5["volume"].tail(20).mean() or 1

    r = rsi(df_m15).iloc[-1]

    if prev["close"] > hi and last["low"] <= hi and last["close"] > hi:
        if last["volume"] > 1.3 * vol_avg and r < 70:
            sl = lo
            tp = last["close"] + 2.5 * (last["close"] - sl)
            return Signal(symbol, "BUY", last["close"], sl, tp, "A",
                          f"Asia breakout+retest, RSI={r:.1f}")

    if prev["close"] < lo and last["high"] >= lo and last["close"] < lo:
        if last["volume"] > 1.3 * vol_avg and r > 30:
            sl = hi
            tp = last["close"] - 2.5 * (sl - last["close"])
            return Signal(symbol, "SELL", last["close"], sl, tp, "A",
                          f"Asia breakdown+retest, RSI={r:.1f}")

    return None


def setup_c_trend_pullback(symbol: str, df_h1: pd.DataFrame, df_d1: pd.DataFrame) -> Optional[Signal]:
    if len(df_h1) < 200 or len(df_d1) < 200:
        return None

    ema200_d1 = ema(df_d1["close"], 200).iloc[-1]
    price = df_h1["close"].iloc[-1]
    trend_up = price > ema200_d1
    trend_down = price < ema200_d1

    ema50 = ema(df_h1["close"], 50).iloc[-1]
    a = atr(df_h1, 14).iloc[-1]
    adx_val = adx(df_h1, 14).iloc[-1]

    if pd.isna(a) or pd.isna(adx_val) or adx_val < 25:
        return None

    last = df_h1.iloc[-1]
    body = abs(last["close"] - last["open"]) or 1e-9
    lower_wick = min(last["open"], last["close"]) - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])

    if trend_up and last["low"] <= ema50 * 1.001 and lower_wick > 2 * body:
        sl = last["low"] - 0.2 * a
        tp = last["close"] + 2 * (last["close"] - sl)
        return Signal(symbol, "BUY", last["close"], sl, tp, "C",
                      f"Trend pullback EMA50 H1, ADX={adx_val:.1f}")

    if trend_down and last["high"] >= ema50 * 0.999 and upper_wick > 2 * body:
        sl = last["high"] + 0.2 * a
        tp = last["close"] - 2 * (sl - last["close"])
        return Signal(symbol, "SELL", last["close"], sl, tp, "C",
                      f"Trend pullback EMA50 H1, ADX={adx_val:.1f}")

    return None
