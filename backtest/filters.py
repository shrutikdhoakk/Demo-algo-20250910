# backtest/filters.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n // 2)).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=close.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index)
    up   = gain.ewm(alpha=1/period, adjust=False).mean()
    dn   = loss.ewm(alpha=1/period, adjust=False).mean()
    rs   = up / dn.replace(0, np.nan)
    out  = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)

def confirm_breakout(
    df: pd.DataFrame,
    high_n: int = 20,
    rsi_len: int = 14,
    rsi_thr: float = 55.0,
    vol_len: int = 20,
    vol_mult: float = 1.5,
    require_volume: bool = True,
) -> pd.Series:
    """
    True when ALL are satisfied:
      - Close > prior rolling-high(high_n)
      - RSI(close, rsi_len) >= rsi_thr
      - Volume >= vol_mult * SMA(volume, vol_len)  (skippable if no volume)
    Columns expected: 'close', 'high', optional 'volume'
    """
    close = df["close"]
    high  = df["high"]

    # 1) Breakout vs prior HH
    hh = high.rolling(high_n, min_periods=high_n).max()
    breakout = close > hh.shift(1)

    # 2) Momentum (RSI)
    r = rsi(close, rsi_len)
    rsi_ok = r >= rsi_thr

    # 3) Volume confirmation (gracefully skip if not available/usable)
    if "volume" in df and df["volume"].notna().sum() >= max(vol_len, 10) and df["volume"].sum() > 0:
        vma = _sma(df["volume"].fillna(0.0), vol_len)
        vol_ok = df["volume"] >= (vol_mult * vma)
        # guard early warm-up region
        vol_ok = vol_ok.fillna(False)
    else:
        vol_ok = pd.Series(
            True if not require_volume else False,
            index=df.index,
            dtype=bool
        )

    return (breakout & rsi_ok & vol_ok)
