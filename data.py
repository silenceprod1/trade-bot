import ccxt.async_support as ccxt
import pandas as pd

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})

TF_MAP = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def _to_strategy_format(df: pd.DataFrame) -> list:
    """Превращает DataFrame в список dict-свечей формата стратегии."""
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
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, TF_MAP[timeframe], limit=limit)
    except Exception as e:
        print(f"Binance error {symbol} {timeframe}: {e}")
        return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


async def fetch_candles(symbol: str, timeframe: str, limit: int = 500) -> list:
    """Возвращает свечи в формате стратегии TradeMind."""
    df = await fetch(symbol, timeframe, limit)
    return _to_strategy_format(df)
