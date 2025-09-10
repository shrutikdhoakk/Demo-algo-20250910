from __future__ import annotations
import math
from typing import Dict, Any
import numpy as np
import pandas as pd

__all__ = ["pattern_gate"]

def _canon(col) -> str:
    # Normalize a column label (supports strings or tuple MultiIndex parts)
    if isinstance(col, tuple):
        # Prefer the last non-empty string part (handles ('Open','TICKER') or ('TICKER','Open'))
        for part in reversed(col):
            if isinstance(part, str) and part.strip():
                return part.strip().lower()
        return str(col).lower()
    return str(col).strip().lower()

def _col(df: pd.DataFrame, candidates):
    # Build lookup maps for exact/loose matching
    cmap: dict[str, object] = {}
    for c in df.columns:
        key = _canon(c)
        cmap[key] = c
        cmap[key.replace(" ", "")] = c
        cmap[key.replace(" ", "").replace("_", "")] = c

    want = []
    for name in candidates:
        k = name.lower()
        want.extend([k, k.replace(" ", ""), k.replace(" ", "").replace("_", "")])

    # exact first
    for w in want:
        if w in cmap:
            return cmap[w]

    # loose contains
    for k, v in cmap.items():
        if any(w in k for w in want):
            return v

    raise KeyError(f"Missing required column. Tried: {candidates} in {list(map(_canon, df.columns))}")

def _last_valid(x: pd.Series) -> float | None:
    try:
        v = float(x.dropna().iloc[-1])
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def pattern_gate(df: pd.DataFrame, bb_n: int = 20, bb_k: float = 2.0, atr_n: int = 14, pct_window: int = 252) -> Dict[str, Any]:
    """
    Minimal technical gate:
      - Bollinger Band width & its percentile
      - ATR & its percentile
      - HH20 / HH50 breakout flags
      - candidate_ok = (squeeze) AND (breakout)
    No external TA libs; works with plain pandas/numpy. Robust to MultiIndex columns.
    """
    if not isinstance(df, pd.DataFrame) or len(df) < max(bb_n, atr_n) + 2:
        return {"ok": False, "reason": "insufficient_rows", "rows": int(len(df)) if hasattr(df, "__len__") else 0}

    # Column resolution (robust to MultiIndex)
    o = _col(df, ["open", "o"])
    h = _col(df, ["high", "h"])
    l = _col(df, ["low", "l"])
    c = _col(df, ["close", "adj close", "c"])
    # volume optional
    try:
        vcol = _col(df, ["volume", "vol", "v"])
        has_volume = True
    except Exception:
        vcol = None
        has_volume = False

    close = pd.to_numeric(df[c], errors="coerce")
    high  = pd.to_numeric(df[h], errors="coerce")
    low   = pd.to_numeric(df[l], errors="coerce")

    # --- Bollinger Band width ---
    mid = close.rolling(bb_n, min_periods=bb_n).mean()
    std = close.rolling(bb_n, min_periods=bb_n).std(ddof=0)
    upper = mid + bb_k * std
    lower = mid - bb_k * std
    bb_width = (upper - lower) / mid.replace(0, np.nan)

    # percentile rank of last value in rolling window
    def _pct_rank(s: pd.Series) -> float:
        if len(s) == 0 or pd.isna(s.iloc[-1]):
            return np.nan
        return float(s.rank(pct=True).iloc[-1])

    bb_width_pctile = bb_width.rolling(pct_window, min_periods=min(bb_n*3, pct_window)).apply(_pct_rank, raw=False)

    # --- ATR (Wilder-style simple mean for simplicity) ---
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_n, min_periods=atr_n).mean()
    atr_pctile = atr.rolling(pct_window, min_periods=min(atr_n*3, pct_window)).apply(_pct_rank, raw=False)

    # --- Breakouts ---
    hh20 = high.shift(1).rolling(20, min_periods=20).max()
    hh50 = high.shift(1).rolling(50, min_periods=50).max()
    breakout_hh20 = (close > hh20)
    breakout_hh50 = (close > hh50)

    # --- Latest values
    last_bb_w = _last_valid(bb_width)
    last_bb_p = _last_valid(bb_width_pctile)
    last_atr  = _last_valid(atr)
    last_atr_p= _last_valid(atr_pctile)

    is_squeeze = (last_bb_p is not None and last_bb_p <= 0.10) or (last_atr_p is not None and last_atr_p <= 0.20)

    bo20 = bool(breakout_hh20.dropna().iloc[-1]) if len(breakout_hh20.dropna()) else False
    bo50 = bool(breakout_hh50.dropna().iloc[-1]) if len(breakout_hh50.dropna()) else False

    candidate_ok = bool(is_squeeze and (bo20 or bo50))

    out = {
        "ok": True,
        "is_squeeze": bool(is_squeeze),
        "bb_width": last_bb_w,
        "bb_width_pctile": last_bb_p,
        "atr": last_atr,
        "atr_pctile": last_atr_p,
        "breakout_hh20": bo20,
        "breakout_hh50": bo50,
        "candidate_ok": candidate_ok,
        "rows": int(len(df)),
        "has_volume": has_volume,
    }
    return out
