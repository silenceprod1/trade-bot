import pandas as pd
import numpy as np


def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def rsi(df: pd.DataFrame, p: int = 14) -> pd.Series:
    d = df["close"].diff()
    gain = d.clip(lower=0).rolling(p).mean()
    loss = (-d.clip(upper=0)).rolling(p).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(p).mean()


def adx(df: pd.DataFrame, p: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_ = tr.rolling(p).mean()
    pdi = 100 * pd.Series(plus, index=df.index).rolling(p).mean() / atr_
    mdi = 100 * pd.Series(minus, index=df.index).rolling(p).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.rolling(p).mean()
