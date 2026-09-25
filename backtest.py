import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta, timezone

exchange = ccxt.binance({"enableRateLimit": True})


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


def rsi(s, p=2):
    d = s.diff()
    gain = d.clip(lower=0).rolling(p).mean()
    loss = (-d.clip(upper=0)).rolling(p).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()


def simulate_fixed_r(side, entry, sl, tp, df_m5, from_idx, max_bars=600):
    """Считаем чистый R без издержек."""
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
    """
    Возвращает dict: {hypothesis_name: [trade dicts]}
    """
    results = {
        "h1_breakout": [],
        "h2_rsi2": [],
        "h3_momentum": [],
        "h4_volatility": [],
        "h5_session": [],
    }

    total = len(symbols)
    for idx, sym in enumerate(symbols, 1):
        if progress_cb:
            await progress_cb(f"📥 [{idx}/{total}] Скачиваю {sym}...")
        df_m5 = await fetch_all(sym, "5m", days)
        df_h1 = await fetch_all(sym, "1h", max(days, 200))
        df_d1 = await fetch_all(sym, "1d", max(days, 200))

        if df_m5.empty or df_h1.empty:
            continue

        if progress_cb:
            await progress_cb(f"🔬 [{idx}/{total}] Прогоняю гипотезы на {sym}...")

        # === ГИПОТЕЗА 1: Пробой вчерашнего дня ===
        df_d1_calc = df_d1.copy()
        df_d1_calc["prev_high"] = df_d1_calc["high"].shift(1)
        df_d1_calc["prev_low"] = df_d1_calc["low"].shift(1)

        for j in range(2, len(df_d1_calc)):
            day = df_d1_calc.iloc[j]
            prev = df_d1_calc.iloc[j - 1]
            cut = day["datetime"]
            # ищем M5-бар в этот день
            mask = (df_m5["datetime"] >= cut) & (df_m5["datetime"] < cut + timedelta(days=1))
            day_m5 = df_m5[mask]
            if len(day_m5) < 100:
                continue

            # пробой вверх
            if day["open"] > prev["prev_high"] if False else False:
                pass

            # проверяем пробой во время дня
            up_break = day_m5[day_m5["high"] > prev["prev_high"]]
            down_break = day_m5[day_m5["low"] < prev["prev_low"]]

            if len(up_break) > 0:
                idx_break = up_break.index[0]
                entry_bar = df_m5.iloc[idx_break]
                entry = entry_bar["close"]
                sl = entry * 0.985
                tp = entry * 1.03
                r, _ = simulate_fixed_r("BUY", entry, sl, tp, df_m5, idx_break)
                if r is not None:
                    results["h1_breakout"].append({"symbol": sym, "r": r, "side": "BUY"})

            if len(down_break) > 0:
                idx_break = down_break.index[0]
                entry_bar = df_m5.iloc[idx_break]
                entry = entry_bar["close"]
                sl = entry * 1.015
                tp = entry * 0.97
                r, _ = simulate_fixed_r("SELL", entry, sl, tp, df_m5, idx_break)
                if r is not None:
                    results["h1_breakout"].append({"symbol": sym, "r": r, "side": "SELL"})

        # === ГИПОТЕЗА 2: RSI(2) возврат к среднему ===
        df_h1_calc = df_h1.copy()
        df_h1_calc["rsi2"] = rsi(df_h1_calc["close"], 2)

        for j in range(20, len(df_h1_calc) - 1):
            row = df_h1_calc.iloc[j]
            if pd.isna(row["rsi2"]):
                continue
            cut = row["datetime"]
            if row["rsi2"] < 10:
                # ищем M5 после cut
                mask = df_m5["datetime"] > cut
                sub = df_m5[mask]
                if len(sub) < 50:
                    continue
                idx_in_m5 = sub.index[0]
                entry = sub.iloc[0]["close"]
                sl = entry * 0.99
                tp = entry * 1.02
                r, _ = simulate_fixed_r("BUY", entry, sl, tp, df_m5, idx_in_m5, max_bars=300)
                if r is not None:
                    results["h2_rsi2"].append({"symbol": sym, "r": r, "side": "BUY"})
            elif row["rsi2"] > 90:
                mask = df_m5["datetime"] > cut
                sub = df_m5[mask]
                if len(sub) < 50:
                    continue
                idx_in_m5 = sub.index[0]
                entry = sub.iloc[0]["close"]
                sl = entry * 1.01
                tp = entry * 0.98
                r, _ = simulate_fixed_r("SELL", entry, sl, tp, df_m5, idx_in_m5, max_bars=300)
                if r is not None:
                    results["h2_rsi2"].append({"symbol": sym, "r": r, "side": "SELL"})

        # === ГИПОТЕЗА 3: Импульс 3 свечи H1 ===
        for j in range(5, len(df_h1_calc) - 1):
            b1 = df_h1_calc.iloc[j - 2]
            b2 = df_h1_calc.iloc[j - 1]
            b3 = df_h1_calc.iloc[j]
            up = b1["close"] < b1["open"] and b2["close"] < b2["open"] and b3["close"] < b3["open"]
            down = b1["close"] > b1["open"] and b2["close"] > b2["open"] and b3["close"] > b3["open"]
            if not (up or down):
                continue
            cut = b3["datetime"]
            sub = df_m5[df_m5["datetime"] > cut]
            if len(sub) < 50:
                continue
            idx_in_m5 = sub.index[0]
            entry = sub.iloc[0]["close"]
            if up:
                sl = entry * 0.99
                tp = entry * 1.02
                r, _ = simulate_fixed_r("SELL", entry, sl, tp, df_m5, idx_in_m5, max_bars=300)
                if r is not None:
                    results["h3_momentum"].append({"symbol": sym, "r": r, "side": "SELL"})
            else:
                sl = entry * 1.01
                tp = entry * 0.98
                r, _ = simulate_fixed_r("BUY", entry, sl, tp, df_m5, idx_in_m5, max_bars=300)
                if r is not None:
                    results["h3_momentum"].append({"symbol": sym, "r": r, "side": "BUY"})

        # === ГИПОТЕЗА 4: Волатильность ===
        a_series = atr(df_h1_calc, 14)
        a_avg_series = a_series.rolling(100).mean()
        for j in range(110, len(df_h1_calc) - 1):
            a = a_series.iloc[j]
            a_avg = a_avg_series.iloc[j]
            if pd.isna(a) or pd.isna(a_avg) or a_avg == 0:
                continue
            if a > 2.0 * a_avg:
                b = df_h1_calc.iloc[j]
                cut = b["datetime"]
                sub = df_m5[df_m5["datetime"] > cut]
                if len(sub) < 50:
                    continue
                idx_in_m5 = sub.index[0]
                entry = sub.iloc[0]["close"]
                # против направления последней свечи
                side = "SELL" if b["close"] > b["open"] else "BUY"
                if side == "SELL":
                    sl = entry * 1.01
                    tp = entry * 0.98
                else:
                    sl = entry * 0.99
                    tp = entry * 1.02
                r, _ = simulate_fixed_r(side, entry, sl, tp, df_m5, idx_in_m5, max_bars=300)
                if r is not None:
                    results["h4_volatility"].append({"symbol": sym, "r": r, "side": side})

        # === ГИПОТЕЗА 5: Сессия ===
        for j in range(1, len(df_h1_calc) - 1):
            b = df_h1_calc.iloc[j]
            h = b["datetime"].hour
            if h == 7:
                cut = b["datetime"]
                sub = df_m5[df_m5["datetime"] > cut]
                if len(sub) < 50:
                    continue
                idx_in_m5 = sub.index[0]
                entry = sub.iloc[0]["close"]
                sl = entry * 0.99
                tp = entry * 1.02
                r, _ = simulate_fixed_r("BUY", entry, sl, tp, df_m5, idx_in_m5, max_bars=300)
                if r is not None:
                    results["h5_session"].append({"symbol": sym, "r": r, "side": "BUY"})
            elif h == 20:
                cut = b["datetime"]
                sub = df_m5[df_m5["datetime"] > cut]
                if len(sub) < 50:
                    continue
                idx_in_m5 = sub.index[0]
                entry = sub.iloc[0]["close"]
                sl = entry * 1.01
                tp = entry * 0.98
                r, _ = simulate_fixed_r("SELL", entry, sl, tp, df_m5, idx_in_m5, max_bars=300)
                if r is not None:
                    results["h5_session"].append({"symbol": sym, "r": r, "side": "SELL"})

    return results


def stats_report(results):
    names = {
        "h1_breakout": "1. Пробой дня",
        "h2_rsi2": "2. RSI(2) возврат",
        "h3_momentum": "3. Импульс 3H1",
        "h4_volatility": "4. Волатильность",
        "h5_session": "5. Сессия",
    }
    lines = ["🔬 <b>ДИАГНОСТИКА 5 ГИПОТЕЗ</b>\n"]
    for key, label in names.items():
        trades = results.get(key, [])
        if not trades:
            lines.append(f"<b>{label}</b>: 0 сделок")
            continue
        r_arr = np.array([t["r"] for t in trades])
        wr = (r_arr > 0).sum() / len(r_arr) * 100
        avg = r_arr.mean()
        total = r_arr.sum()
        lines.append(
            f"<b>{label}</b>: {len(trades)} сд. | "
            f"WR {wr:.1f}% | AvgR {avg:+.3f} | ΣR {total:+.1f}"
        )
    return "\n".join(lines)
