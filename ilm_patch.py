"""
Патч для TradeMind v9.41 (v4).
- Ослаблен ILM: body 0.20, close через 10% диапазона, окно триггера 10 свечей, recovery с 0.15.
- confirmation_15m: walk-forward (игнорирует незакрытые 15M).
"""

import trademind as tm
from trademind import (
    _f, _t, _c, _h, _l, _o, _body, _range, _body_ratio,
    _is_local_high, _is_local_low,
    MAX_5M_ILM_CANDLES,
    MIN_5M_ILM_SWEEP_DISTANCE_PCT, MIN_SWEEP_DEPTH_PCT,
    MAX_15M_CONFIRM_CANDLES,
)

MIN_BODY_RATIO_TRIGGER_5M_PATCH = 0.20
CLOSE_BREAK_FRACTION = 0.10
VSHAPE_TOLERANCE = 0.005
ILM_TRIGGER_WINDOW_PATCH = 10
MIN_5M_RECOVERY_RATIO_PATCH = 0.15
MAX_5M_RECOVERY_RATIO_PATCH = 1.80

_CURRENT_CUT_TS = None
TF_15M_MS = 15 * 60 * 1000


def set_cut_ts(cut_ts):
    global _CURRENT_CUT_TS
    _CURRENT_CUT_TS = cut_ts


def _ilm_long_patched(candles, i, sweep_lvl, sweep_ext, min_depth):
    mc = candles[i]
    if not _is_local_low(candles, i):
        return None
    ml = _l(mc); mh = _h(mc)
    if ml is None or mh is None:
        return None
    before = candles[max(0, i - 2):i]
    if not before:
        return None
    bl = [_l(x) for x in before if _l(x) is not None]
    if not bl:
        return None
    left_ref = min(bl)
    if left_ref <= ml:
        return None
    target = left_ref
    if target <= ml:
        return None
    m_range = target - ml
    if m_range <= 0:
        return None
    m_pct = m_range / target * 100
    if m_pct < min_depth:
        return None

    mid_threshold = mh - (mh - ml) * CLOSE_BREAK_FRACTION
    trig_idx = None
    end = min(len(candles), i + 1 + ILM_TRIGGER_WINDOW_PATCH)
    for j in range(i + 1, end):
        trig = candles[j]
        tc = _c(trig)
        if tc is None or not (tc > _o(trig)):
            continue
        if _body_ratio(trig) < MIN_BODY_RATIO_TRIGGER_5M_PATCH:
            continue
        if j >= 2:
            h1 = _h(candles[j - 1]); h2 = _h(candles[j - 2])
            if h1 is not None and h2 is not None:
                threshold = max(h1, h2) * (1 - VSHAPE_TOLERANCE)
                if tc <= threshold:
                    continue
        if tc >= mid_threshold:
            trig_idx = j
            break
    if trig_idx is None:
        return None
    trig = candles[trig_idx]
    tc = _c(trig)
    if tc is None:
        return None
    rec = (tc - ml) / m_range
    if rec < MIN_5M_RECOVERY_RATIO_PATCH or rec > MAX_5M_RECOVERY_RATIO_PATCH:
        return None
    if sweep_lvl is not None:
        d = abs(ml - sweep_lvl) / sweep_lvl * 100
        if d > MIN_5M_ILM_SWEEP_DISTANCE_PCT:
            return None
    if sweep_ext is not None:
        lim = sweep_ext * (1 + MIN_5M_ILM_SWEEP_DISTANCE_PCT / 100)
        if ml > lim:
            return None
    return {
        "direction": "LONG", "extreme": ml,
        "trigger_time": _t(trig), "trigger_price": tc,
        "reason": "5M V-ILM (patch v4)",
        "recovery_ratio": rec, "manipulation_pct": m_pct,
        "age_candles": len(candles) - 1 - trig_idx,
    }


def _ilm_short_patched(candles, i, sweep_lvl, sweep_ext, min_depth):
    mc = candles[i]
    if not _is_local_high(candles, i):
        return None
    mh = _h(mc); ml = _l(mc)
    if mh is None or ml is None:
        return None
    before = candles[max(0, i - 2):i]
    if not before:
        return None
    bh = [_h(x) for x in before if _h(x) is not None]
    if not bh:
        return None
    left_ref = max(bh)
    if mh <= left_ref:
        return None
    target = left_ref
    if mh <= target:
        return None
    m_range = mh - target
    if m_range <= 0:
        return None
    m_pct = m_range / mh * 100
    if m_pct < min_depth:
        return None

    mid_threshold = ml + (mh - ml) * CLOSE_BREAK_FRACTION
    trig_idx = None
    end = min(len(candles), i + 1 + ILM_TRIGGER_WINDOW_PATCH)
    for j in range(i + 1, end):
        trig = candles[j]
        tc = _c(trig)
        if tc is None or not (tc < _o(trig)):
            continue
        if _body_ratio(trig) < MIN_BODY_RATIO_TRIGGER_5M_PATCH:
            continue
        if j >= 2:
            l1 = _l(candles[j - 1]); l2 = _l(candles[j - 2])
            if l1 is not None and l2 is not None:
                threshold = min(l1, l2) * (1 + VSHAPE_TOLERANCE)
                if tc >= threshold:
                    continue
        if tc <= mid_threshold:
            trig_idx = j
            break
    if trig_idx is None:
        return None
    trig = candles[trig_idx]
    tc = _c(trig)
    if tc is None:
        return None
    rec = (mh - tc) / m_range
    if rec < MIN_5M_RECOVERY_RATIO_PATCH or rec > MAX_5M_RECOVERY_RATIO_PATCH:
        return None
    if sweep_lvl is not None:
        d = abs(mh - sweep_lvl) / sweep_lvl * 100
        if d > MIN_5M_ILM_SWEEP_DISTANCE_PCT:
            return None
    if sweep_ext is not None:
        lim = sweep_ext * (1 - MIN_5M_ILM_SWEEP_DISTANCE_PCT / 100)
        if mh < lim:
            return None
    return {
        "direction": "SHORT", "extreme": mh,
        "trigger_time": _t(trig), "trigger_price": tc,
        "reason": "5M L-ILM (patch v4)",
        "recovery_ratio": rec, "manipulation_pct": m_pct,
        "age_candles": len(candles) - 1 - trig_idx,
    }


def detect_5m_ilm_patched(candles_5m, sweep, direction, conf_time=None, config=None):
    if config is None:
        config = {}
    min_depth = config.get("MIN_SWEEP_DEPTH_PCT", MIN_SWEEP_DEPTH_PCT)
    if not sweep:
        return False, None
    if direction not in ("LONG", "SHORT"):
        return False, None
    start = _f(conf_time) or _f(sweep.get("open_time"))
    candles = []
    for c in candles_5m or []:
        t = _t(c)
        if start is None:
            candles.append(c)
        elif t is not None and t > start:
            candles.append(c)
    candles = candles[-MAX_5M_ILM_CANDLES:]
    if len(candles) < 5:
        return False, None
    sl = _f(sweep.get("level"))
    se = _f(sweep.get("extreme"))
    cands = []
    for i in range(2, len(candles) - 2):
        if direction == "LONG":
            ilm = _ilm_long_patched(candles, i, sl, se, min_depth)
        else:
            ilm = _ilm_short_patched(candles, i, sl, se, min_depth)
        if ilm:
            cands.append(ilm)
    if not cands:
        return False, None
    best = None
    best_score = -1e9
    for cand in cands:
        sc = cand["recovery_ratio"] * 30 + cand["manipulation_pct"] * 5
        if sc > best_score:
            best_score = sc
            best = cand
    return True, best


def confirmation_15m_patched(candles_15m, sweep, direction):
    if not sweep or not candles_15m:
        return False, None, None, False, 0.0
    if direction not in ("LONG", "SHORT"):
        return False, None, None, False, 0.0

    sweep_t = _f(sweep.get("open_time"))

    closed = []
    for c in candles_15m:
        t = _t(c)
        if t is None:
            continue
        if _CURRENT_CUT_TS is not None and (t + TF_15M_MS) > _CURRENT_CUT_TS:
            continue
        closed.append(c)

    candidates = []
    for c in closed:
        t = _t(c)
        if sweep_t is None:
            candidates.append(c)
        elif t is not None and t > sweep_t:
            candidates.append(c)
    candidates = candidates[-MAX_15M_CONFIRM_CANDLES:]
    if len(candidates) < 3:
        return False, None, None, False, 0.0

    for i in range(1, len(candidates)):
        c = candidates[i]
        br = _body_ratio(c)
        if br < 0.50:
            continue
        close = _c(c)
        if close is None:
            continue
        if direction == "LONG":
            if not _body(c) or close <= _o(c):
                continue
            highs = [_h(candidates[j]) for j in range(i - 1)
                     if _is_local_high(candidates, j)
                     and _h(candidates[j]) is not None]
            if not highs:
                continue
            ref = max(highs[-3:])
            if close > ref:
                strength = min(1.0, br * 1.2)
                return True, "15M BOS", _t(c), True, strength
        else:
            if close >= _o(c):
                continue
            lows = [_l(candidates[j]) for j in range(i - 1)
                    if _is_local_low(candidates, j)
                    and _l(candidates[j]) is not None]
            if not lows:
                continue
            ref = min(lows[-3:])
            if close < ref:
                strength = min(1.0, br * 1.2)
                return True, "15M BOS", _t(c), True, strength

    inv_ok, inv_reason = tm._check_fvg_inversion_15m(
        candidates, direction, len(candidates) - 1)
    if inv_ok:
        last_c = candidates[-1]
        return True, inv_reason or "15M FVG inv", _t(last_c), False, 0.85

    return False, None, None, False, 0.0


def apply_patch():
    tm.detect_5m_ilm = detect_5m_ilm_patched
    tm.confirmation_15m = confirmation_15m_patched
    tm.MIN_BODY_RATIO_TRIGGER_5M = MIN_BODY_RATIO_TRIGGER_5M_PATCH
    return True
