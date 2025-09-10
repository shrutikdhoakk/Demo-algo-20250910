"""Chart pattern detection for tight consolidations and reversals.

The Breakout + Momentum Swing strategy looks for periods of tight
consolidation preceding a breakout.  Such consolidations are identified
via a small Bollinger band width, low ATR (volatility compression) and
price squeezing around its moving average.  This module exposes
functions to flag such conditions and compute a continuous score
capturing the degree of tightness.

It also provides a very simple bearish reversal detector for early
exit rules.  This detector flags when a recent high is no longer
exceeding a previous high and the close has fallen below its level
three bars back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ..features.indicators import bb_width, atr, sma

__all__ = ['bullish_pattern_confirm', 'pattern_score', 'bearish_reversal_detect']


def _rolling_quantile(s: pd.Series, q: float, win: int = 100) -> pd.Series:
    """Rolling quantile helper with a sensible minimum period."""
    return s.rolling(win, min_periods=min(50, win // 2)).quantile(q)


def bullish_pattern_confirm(
    df: pd.DataFrame,
    bb_n: int = 20,
    bb_pct: int = 5,
    atr_win: int = 20,
    squeeze_win: int = 20,
) -> pd.Series:
    """Flag bars during tight consolidation suitable for a breakout.

    A bar is considered part of a bullish consolidation if:

    1. The Bollinger band width is in the lowest `bb_pct` percentile over a
       100‑bar window.
    2. The ATR is below its median over a rolling `atr_win` window.
    3. The price is tightly squeezed around its moving average; this
       implementation deems a squeeze if the distance between the close
       and its moving average is less than 20 % of the daily range.

    All three conditions must be simultaneously true.  The result is
    returned as a boolean Series aligned with the input index.
    """
    close = df['Close'].astype(float)
    width = bb_width(close, n=bb_n)
    q = _rolling_quantile(width, bb_pct / 100.0)
    tight = width <= q
    atr_series = atr(df, 14)
    atr_median = atr_series.rolling(atr_win, min_periods=max(5, atr_win // 2)).median()
    compression = atr_series <= atr_median
    ma = sma(close, squeeze_win)
    rng = (df['High'].astype(float) - df['Low'].astype(float)).replace(0, np.nan)
    band = (close - ma).abs() / rng
    squeeze = band <= 0.20
    return (tight & compression & squeeze).fillna(False)


def pattern_score(
    df: pd.DataFrame,
    bb_n: int = 20,
    bb_pct: int = 5,
    atr_win: int = 20,
    squeeze_win: int = 20,
) -> pd.Series:
    """Compute a continuous score measuring consolidation tightness.

    The score combines three sub‑components:

    * **Tightness score (50 % weight):** Ranks the current Bollinger band width
      relative to its rolling min/max range over the past 100 bars.  A
      smaller width results in a higher score.
    * **ATR compression score (30 % weight):** Binary indicator (0 or 1)
      denoting whether the ATR is below its rolling median.
    * **Squeeze score (20 % weight):** Measures how close the price is to
      its moving average relative to the daily range; closer is better.

    The final score is bounded between 0 and 1.
    """
    close = df['Close'].astype(float)
    width = bb_width(close, n=bb_n)
    roll_min = width.rolling(100, min_periods=50).min()
    roll_max = width.rolling(100, min_periods=50).max()
    pct_rank = (width - roll_min) / (roll_max - roll_min)
    pct_rank = pct_rank.clip(0, 1)
    tight_score = 1 - pct_rank
    atr_series = atr(df, 14)
    atr_median = atr_series.rolling(atr_win, min_periods=max(5, atr_win // 2)).median()
    atr_score = (atr_series <= atr_median).astype(float)
    ma = sma(close, squeeze_win)
    rng = (df['High'].astype(float) - df['Low'].astype(float)).replace(0, np.nan)
    band = (close - ma).abs() / rng
    band_score = (1 - (band / 0.20)).clip(0, 1)
    score = 0.5 * tight_score + 0.3 * atr_score + 0.2 * band_score
    return score.fillna(0).clip(0, 1)


def bearish_reversal_detect(df: pd.DataFrame) -> pd.Series:
    """Detect a simple bearish reversal for early exits.

    Flags True when the most recent 20‑bar high is not greater than the
    high from five bars ago and the close has dropped below its value
    three bars back.  This crude detector helps identify double top–like
    structures.
    """
    close = df['Close'].astype(float)
    hh = close.rolling(20, min_periods=5).max()
    prev = hh.shift(5)
    dt = ((hh <= prev * 1.02) & (close < close.shift(3))).astype(bool)
    return dt.fillna(False)