import ccxt.async_support as ccxt
import pandas as pd

# Публичный доступ, ключи не нужны
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


async def fetch(symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    """
    symbol: 'BTC/USDT'
    timeframe: '5m', '15m', '1h', '4h', '1d'
    """
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, TF_MAP[timeframe], limit=limit)
    except Exception as e:
        print(f"Binance error {symbol} {timeframe}: {e}")
        return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    return df
