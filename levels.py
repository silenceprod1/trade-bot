"""
Построение major_levels для TradeMind v9.41.
Уровни — свинг-хаи (BSL) и свинг-лои (SSL) на H1, с силой = количеству касаний.
"""

from data import fetch_candles


def _swing_high(candles, i, left=2, right=2):
    if i < left or i >= len(candles) - right:
        return False
    cur = candles[i]["high"]
    for k in range(1, left + 1):
        if candles[i - k]["high"] >= cur:
            return False
    for k in range(1, right + 1):
        if candles[i + k]["high"] >= cur:
            return False
    return True


def _swing_low(candles, i, left=2, right=2):
    if i < left or i >= len(candles) - right:
        return False
    cur = candles[i]["low"]
    for k in range(1, left + 1):
        if candles[i - k]["low"] <= cur:
            return False
    for k in range(1, right + 1):
        if candles[i + k]["low"] <= cur:
            return False
    return True


def _count_touches(candles, price, tol_pct=0.15):
    """Сколько раз цена касалась уровня в пределах tol%."""
    tol = price * tol_pct / 100
    touches = 0
    for c in candles:
        if abs(c["high"] - price) <= tol or abs(c["low"] - price) <= tol:
            touches += 1
    return touches


def build_major_levels(candles_h1, lookback=200, max_levels=15):
    """
    Возвращает список уровней:
    {"price": float, "side": "LONG"|"SHORT", "type": "SSL"|"BSL", "strength": float, "touches": int}
    LONG (SSL) — поддержки (свинг-лои), SHORT (BSL) — сопротивления (свинг-хаи).
    """
    if not candles_h1 or len(candles_h1) < 30:
        return []

    window = candles_h1[-lookback:] if len(candles_h1) > lookback else candles_h1
    levels = []

    for i in range(len(window)):
        if _swing_low(window, i):
            price = window[i]["low"]
            touches = _count_touches(window, price)
            strength = min(100.0, touches * 15.0 + 40.0)
            levels.append({
                "price": price,
                "side": "LONG",
                "type": "SSL",
                "strength": strength,
                "touches": touches,
                "swept": False,
            })

        if _swing_high(window, i):
            price = window[i]["high"]
            touches = _count_touches(window, price)
            strength = min(100.0, touches * 15.0 + 40.0)
            levels.append({
                "price": price,
                "side": "SHORT",
                "type": "BSL",
                "strength": strength,
                "touches": touches,
                "swept": False,
            })

    # убираем дубликаты (близкие уровни) и оставляем самые сильные
    levels.sort(key=lambda x: x["strength"], reverse=True)
    filtered = []
    for lv in levels:
        close = False
        for exist in filtered:
            if abs(lv["price"] - exist["price"]) / exist["price"] < 0.002:
                close = True
                break
        if not close:
            filtered.append(lv)
        if len(filtered) >= max_levels:
            break

    return filtered


async def get_major_levels(symbol: str):
    candles = await fetch_candles(symbol, "1h", 400)
    return build_major_levels(candles, lookback=300, max_levels=20)
