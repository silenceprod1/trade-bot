import asyncio
import ccxt.async_support as ccxt
import pandas as pd
from typing import Optional

_exchange: Optional[ccxt.binance] = None


def get_exchange() -> ccxt.binance:
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    return _exchange


async def close_exchange():
    global _exchange
    if _exchange is not None:
        try:
            await _exchange.close()
        except Exception:
            pass
        _exchange = None


TF_MAP = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def _to_strategy_format(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    candles = []
    for _, row in df.iterrows():
        candles.append({
            "open_time": int(row["timestamp"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    return candles


async def fetch(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Быстрая функция для /scan и /debug — берёт последние `limit` свечей."""
    ex = get_exchange()
    try:
        ohlcv = await ex.fetch_ohlcv(symbol, TF_MAP[timeframe], limit=limit)
    except Exception as e:
        print(f"Binance error {symbol} {timeframe}: {e}")
        return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


async def fetch_history(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Качает всю историю за `days` дней пошагово (по 1000 свечей за запрос)."""
    ex = get_exchange()
    tf_ms = ex.parse_timeframe(TF_MAP[timeframe]) * 1000
    now_ms = ex.milliseconds()
    since = now_ms - days * 24 * 60 * 60 * 1000
    all_data = []
    safety = 0

    while since < now_ms and safety < 200:
        safety += 1
        try:
            batch = await ex.fetch_ohlcv(symbol, TF_MAP[timeframe], since=since, limit=1000)
        except Exception as e:
            print(f"fetch_history error {symbol} {timeframe}: {e}")
            await asyncio.sleep(1)
            continue

        if not batch:
            break

        all_data.extend(batch)
        since = batch[-1][0] + tf_ms
        await asyncio.sleep(0.15)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


async def fetch_candles(symbol: str, timeframe: str, limit: int = 500) -> list:
    df = await fetch(symbol, timeframe, limit)
    return _to_strategy_format(df)


async def fetch_candles_history(symbol: str, timeframe: str, days: int) -> list:
    df = await fetch_history(symbol, timeframe, days)
    return _to_strategy_format(df)
