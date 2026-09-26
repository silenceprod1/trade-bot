"""
Fair Value Gaps на M15.
FVG bullish: low[i+1] > high[i-1]  (разрыв вверх)
FVG bearish: high[i+1] < low[i-1]  (разрыв вниз)
"""


def build_fvgs(candles_15m, lookback=100):
    if not candles_15m or len(candles_15m) < 5:
        return []

    window = candles_15m[-lookback:] if len(candles_15m) > lookback else candles_15m
    fvgs = []

    for i in range(1, len(window) - 1):
        c_prev = window[i - 1]
        c_next = window[i + 1]

        # bullish FVG
        if c_next["low"] > c_prev["high"]:
            fvgs.append({
                "type": "bullish",
                "top": c_next["low"],
                "bottom": c_prev["high"],
                "open_time": window[i]["open_time"],
            })

        # bearish FVG
        if c_next["high"] < c_prev["low"]:
            fvgs.append({
                "type": "bearish",
                "top": c_prev["low"],
                "bottom": c_next["high"],
                "open_time": window[i]["open_time"],
            })

    # оставляем только те, что ещё не перекрыты ценой полностью
    return fvgs[-30:]


async def get_fvgs(symbol: str):
    from data import fetch_candles
    candles = await fetch_candles(symbol, "15m", 200)
    return build_fvgs(candles, lookback=150)
