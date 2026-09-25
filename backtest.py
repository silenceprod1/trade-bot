"""
Бэктест сетапов A и C на 90 дней данных Binance.
Запуск: python backtest.py
Результат: отчёт по каждому сетапу + общий винрейт и Sharpe.
"""

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# === НАСТРОЙКИ БЭКТЕСТА ===
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
DAYS = 90
INITIAL_BALANCE = 1000      # стартовый депозит (условный)
RISK_PER_TRADE = 0.01       # 1% риска на сделку
FEE = 0.001                 # 0.1% комиссия Binance (taker)

exchange = ccxt.binance({"enableRateLimit": True})


# === ЗАГРУЗКА ИСТОРИИ ===
def fetch_all(symbol, timeframe, days):
    """Скачивает свечи с Binance по кускам (макс 1000 за раз)."""
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )
    now = exchange.milliseconds()
    all_data = []
    tf_ms = exchange.parse_timeframe(timeframe) * 1000

    while since < now:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since, 1000)
        except Exception as e:
            print(f"  ошибка fetch {symbol}: {e}, пауза...")
            time.sleep(2)
            continue
        if not batch:
            break
        all_data.extend(batch)
        since = batch[-1][0] + tf_ms
        time.sleep(0.1)

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# === ИНДИКАТОРЫ (дублируем, чтобы бэктест был автономным) ===
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


# === СЕТАПЫ (те же условия, что в боте) ===
def setup_a_asia_breakout(df_m5, df_m15):
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
            return {"side": "BUY", "entry": last["close"], "sl": lo,
                    "tp": last["close"] + 2.5 * (last["close"] - lo), "setup": "A"}

    if prev["close"] < lo and last["high"] >= lo and last["close"] < lo:
        if last["volume"] > 1.3 * vol_avg and r > 30:
            return {"side": "SELL", "entry": last["close"], "sl": hi,
                    "tp": last["close"] - 2.5 * (hi - last["close"]), "setup": "A"}

    return None


def setup_c_trend_pullback(df_h1, df_d1):
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
        return {"side": "BUY", "entry": last["close"], "sl": sl,
                "tp": last["close"] + 2 * (last["close"] - sl), "setup": "C"}

    if trend_down and last["high"] >= ema50 * 0.999 and upper_wick > 2 * body:
        sl = last["high"] + 0.2 * a
        return {"side": "SELL", "entry": last["close"], "sl": sl,
                "tp": last["close"] - 2 * (sl - last["close"]), "setup": "C"}

    return None


# === СИМУЛЯЦИЯ ОДНОЙ СДЕЛКИ ===
def simulate_trade(side, entry, sl, tp, df_m5, from_idx, max_bars=200):
    """
    Идём вперёд по M5-свечам и смотрим, что сработает раньше: TP или SL.
    Возвращаем R-результат и индекс выхода.
    """
    if side == "BUY":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp

    if risk <= 0 or reward <= 0:
        return None, from_idx

    for i in range(from_idx + 1, min(from_idx + 1 + max_bars, len(df_m5))):
        bar = df_m5.iloc[i]
        if side == "BUY":
            if bar["low"] <= sl:
                return -1.0, i
            if bar["high"] >= tp:
                return round(reward / risk, 2), i
        else:
            if bar["high"] >= sl:
                return -1.0, i
            if bar["low"] <= tp:
                return round(reward / risk, 2), i

    return 0.0, from_idx + max_bars  # не закрылась — считаем как 0R


# === ОСНОВНОЙ БЭКТЕСТ ===
def run_backtest():
    print(f"\n=== БЭКТЕСТ {DAYS} ДНЕЙ ===")
    print(f"Символы: {', '.join(SYMBOLS)}")
    print(f"Комиссия: {FEE*100:.2f}% на сделку\n")

    all_trades = []

    for sym in SYMBOLS:
        print(f"Скачиваю {sym}...")
        df_m5 = fetch_all(sym, "5m", DAYS)
        df_m15 = fetch_all(sym, "15m", DAYS)
        df_h1 = fetch_all(sym, "1h", DAYS)
        df_d1 = fetch_all(sym, "1d", DAYS)
        print(f"  M5: {len(df_m5)} свечей, H1: {len(df_h1)}, D1: {len(df_d1)}")

        # сдвигаем окна, чтобы не подглядывать в будущее
        step = 3  # проверяем каждые 3 свечи M5 (15 минут)
        bars_h1 = 200
        bars_d1 = 200

        for i in range(50, len(df_m5) - 1, step):
            # окно M5 до i
            win_m5 = df_m5.iloc[:i + 1].copy()
            win_m15 = df_m15[df_m15["datetime"] <= win_m5["datetime"].iloc[-1]].tail(50)
            if len(win_m15) < 30:
                continue

            # H1 и D1 до текущего момента
            cut = win_m5["datetime"].iloc[-1]
            win_h1 = df_h1[df_h1["datetime"] <= cut].tail(bars_h1)
            win_d1 = df_d1[df_d1["datetime"] <= cut].tail(bars_d1)
            if len(win_h1) < 200 or len(win_d1) < 200:
                continue

            sig = None
            now_hour = win_m5["datetime"].iloc[-1].hour
            if 7 <= now_hour < 10:
                sig = setup_a_asia_breakout(win_m5, win_m15)
            if not sig:
                sig = setup_c_trend_pullback(win_h1, win_d1)

            if sig:
                # имитируем сделку на M5-свечах вперёд
                r, exit_idx = simulate_trade(
                    sig["side"], sig["entry"], sig["sl"], sig["tp"], df_m5, i
                )
                if r is None:
                    continue
                # вычитаем комиссию (0.1% x2 = 0.2% от цены, в R-выражении ≈ риск*0.002)
                r_net = r - (FEE * 2) / (abs(sig["entry"] - sig["sl"]) / sig["entry"]) if abs(sig["entry"] - sig["sl"]) > 0 else r
                all_trades.append({
                    "symbol": sym,
                    "setup": sig["setup"],
                    "side": sig["side"],
                    "entry": sig["entry"],
                    "sl": sig["sl"],
                    "tp": sig["tp"],
                    "r": round(r_net, 3),
                    "time": win_m5["datetime"].iloc[-1],
                })
                print(f"  {win_m5['datetime'].iloc[-1]} | {sym} | {sig['setup']} | {sig['side']} | R={r_net:.2f}")

    # === СТАТИСТИКА ===
    if not all_trades:
        print("\nСделок не найдено. Стратегия слишком строгая для этого периода.")
        return

    df = pd.DataFrame(all_trades)
    df.to_csv("backtest_trades.csv", index=False)
    print(f"\nВсего сделок: {len(df)}")
    print(f"Сохранено в backtest_trades.csv\n")

    for setup in ["A", "C"]:
        sub = df[df["setup"] == setup]
        if len(sub) == 0:
            print(f"Сетап {setup}: 0 сделок")
            continue
        wins = (sub["r"] > 0).sum()
        losses = (sub["r"] < 0).sum()
        wr = wins / len(sub) * 100
        avg_r = sub["r"].mean()
        total_r = sub["r"].sum()
        print(f"Сетап {setup}: {len(sub)} сделок | Winrate {wr:.1f}% | AvgR {avg_r:+.3f} | TotalR {total_r:+.1f}")

    # Общие метрики
    wins = (df["r"] > 0).sum()
    losses = (df["r"] < 0).sum()
    wr = wins / len(df) * 100
    avg_r = df["r"].mean()
    total_r = df["r"].sum()

    # Sharpe (по сделкам, annualized приблизительно)
    if df["r"].std() > 0:
        sharpe = np.sqrt(len(df)) * avg_r / df["r"].std()
    else:
        sharpe = 0.0

    print(f"\n=== ИТОГО ===")
    print(f"Сделок: {len(df)}")
    print(f"Winrate: {wr:.1f}%")
    print(f"Средний R: {avg_r:+.3f}")
    print(f"Суммарный R: {total_r:+.1f}")
    print(f"Sharpe (на сделку): {sharpe:.2f}")
    print(f"Ожидаемый результат при риске {RISK_PER_TRADE*100:.0f}%: {total_r * RISK_PER_TRADE * 100:+.1f}%")


if __name__ == "__main__":
    run_backtest()
