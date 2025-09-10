from __future__ import annotations
import os
import numpy as np
import pandas as pd

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)

# Tunables via environment
BBW_PCTL        = _env_float("GATE_BBW_PCTL", 15.0)     # bottom % of BB widths
TIGHTRANGE_PCT  = _env_float("GATE_TIGHTRANGE_PCT", 0.06)  # N-day range vs price
ATR_RATIO_MAX   = _env_float("GATE_ATR_RATIO", 0.65)    # ATR14/ATR100 <= X
HH_N            = int(os.environ.get("GATE_HH_N", "20")) # breakout lookback

def _true_range(df: pd.DataFrame) -> pd.Series:
    pc = df['close'].shift(1)
    return pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - pc).abs(),
        (df['low']  - pc).abs()
    ], axis=1).max(axis=1)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _true_range(df).rolling(n, min_periods=n).mean()

def _bb_width(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    m = df['close'].rolling(n, min_periods=n).mean()
    s = df['close'].rolling(n, min_periods=n).std(ddof=0)
    upper = m + k*s
    lower = m - k*s
    return (upper - lower) / m  # normalized width

def _bull_engulfing(df: pd.DataFrame) -> pd.Series:
    p = df.shift(1)
    return (
        (p['close'] < p['open']) &
        (df['close'] > df['open']) &
        (df['close'] >= p[['open','close']].max(axis=1)) &
        (df['open']  <= p[['open','close']].min(axis=1))
    )

def _hammer(df: pd.DataFrame, body_ratio: float = 0.35, lower_shadow_ratio: float = 2.0) -> pd.Series:
    body  = (df['close'] - df['open']).abs()
    rng   = (df['high'] - df['low']).replace(0, np.nan)
    lowsh = (df[['open','close']].min(axis=1) - df['low']).abs()
    upsh  = (df['high'] - df[['open','close']].max(axis=1)).abs()
    small_body  = (body / rng) <= body_ratio
    long_lower  = (lowsh / body.replace(0, np.nan)) >= lower_shadow_ratio
    small_upper = upsh <= body
    return small_body & long_lower & small_upper

def pattern_gate(df: pd.DataFrame, now_idx=None) -> dict:
    """ ok = breakout(>HH_N) AND consolidation/squeeze AND bullish candle """
    cols = {c.lower(): c for c in df.columns}
    o,h,l,c = [cols.get(k, k) for k in ('open','high','low','close')]
    x = df.rename(columns={o:'open', h:'high', l:'low', c:'close'}).copy()

    if now_idx is None:
        now_idx = x.index[-1]
    x = x.loc[:now_idx].tail(160)

    atr14  = _atr(x, 14)
    atr100 = _atr(x, 100)
    bbw    = _bb_width(x, 20, 2.0)
    bbw_pct = (bbw.rank(pct=True) * 100.0)

    rngN = x['high'].rolling(HH_N).max() - x['low'].rolling(HH_N).min()
    tight_range  = (rngN / x['close'] <= TIGHTRANGE_PCT)
    low_bbw      = (bbw_pct <= BBW_PCTL)
    atr_compress = (atr14 / atr100.replace(0, np.nan) <= ATR_RATIO_MAX)

    consolidating = (tight_range & low_bbw) | (low_bbw & atr_compress)
    breakout = x['close'] > x['close'].rolling(HH_N).max().shift(1)
    bullish  = _bull_engulfing(x) | _hammer(x)

    ok = (False if GATE_FORCE_FALSE else bool((consolidating & breakout & bullish).iloc[-1]))
    print('PATTERN_GATE_CALLED', ok); return {"ok": ok,
            "signals": {"consolidating": bool(consolidating.iloc[-1]),
                        "breakout": bool(breakout.iloc[-1]),
                        "bullish": bool(bullish.iloc[-1])}}



