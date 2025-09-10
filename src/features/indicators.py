"""Technical indicators used by the strategy.

This module implements a small set of core technical indicators without
relying on external packages such as TA‑Lib.  The functions operate on
pandas Series or DataFrames and return pandas Series aligned with the
input index.  Some indicators (RSI, ATR, ADX) require a minimum number
of periods before producing non‑NaN values.  Where appropriate a
rolling window is used with a sensible minimum for warm‑up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=n, min_periods=max(1, n // 2)).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index.

    Computes the RSI by separating positive and negative closes and
    applying a rolling average to each side.  A value above 70 implies
    overbought conditions and below 30 implies oversold conditions.
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(n, min_periods=n).mean()
    loss = down.rolling(n, min_periods=n).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range.

    The ATR measures market volatility by smoothing the true range.  The
    true range is the maximum of the high‑low range, the absolute
    difference between the current high and the previous close and the
    absolute difference between the current low and the previous close.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=n).mean()


def hhv(series: pd.Series, n: int) -> pd.Series:
    """Highest high value over a rolling window."""
    return series.rolling(window=n, min_periods=1).max()


def zscore(series: pd.Series, n: int) -> pd.Series:
    """Rolling z‑score of a series."""
    mean = series.rolling(window=n, min_periods=n).mean()
    std = series.rolling(window=n, min_periods=n).std(ddof=0)
    return (series - mean) / std


def bb_width(series: pd.Series, n: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger band width as a fraction of the moving average.

    Computes (upperBand - lowerBand) / movingAverage.  A narrow band
    indicates low volatility and a potential squeeze.
    """
    ma = series.rolling(window=n, min_periods=n).mean()
    std = series.rolling(window=n, min_periods=n).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma
    return width


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average Directional Index.

    This function implements a simplified version of Welles Wilder's
    ADX.  It calculates plus and minus directional movement (DM), the
    true range, directional indicators (DI) and then smooths the
    resulting DX.  NaN values will be returned for the first few
    periods as sufficient history is required to compute the averages.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # True range
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(n, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n, min_periods=n).sum() / atr_series
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n, min_periods=n).sum() / atr_series
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(n, min_periods=n).mean()
    return adx