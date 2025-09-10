"""Bullish candlestick pattern detection.

This module provides a simple detector that flags each bar as bullish
when it meets one of several classic candlestick patterns: bullish
engulfing, hammer, marubozu or a simplified morning star.  The
implementation is intentionally lightweight and conservative; it does
not attempt to cover every nuance of candle analysis but is
sufficient for systematic backtesting.
"""

from __future__ import annotations

import pandas as pd


def bullish_on_bar(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series indicating bullish candlestick patterns.

    A bar is flagged True if it satisfies any of the following:

    * **Bullish engulfing:** The current candle closes higher than it opens
      while the previous candle closes lower than it opens, and the
      current candle's body engulfs the previous candle's body.
    * **Hammer:** The current candle has a long lower shadow (at least
      twice the body) and closes higher than it opens.
    * **Marubozu:** The candle opens near the low, closes near the high
      (within 10 % of the total range) and is bullish.
    * **Morning star:** A simple three‑bar pattern: a long bearish
      candle, followed by a small indecision candle (body less than
      half of the previous body) and then a bullish candle that closes
      above the open of the first bar.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least the columns ``Open``, ``High``, ``Low`` and
        ``Close``.

    Returns
    -------
    pandas.Series
        Boolean series aligned with the input index; True where a
        bullish pattern is detected, False otherwise.
    """
    results = []
    for i in range(len(df)):
        if i == 0:
            results.append(False)
            continue
        open_prev = df.iloc[i - 1]['Open']
        close_prev = df.iloc[i - 1]['Close']
        open_cur = df.iloc[i]['Open']
        close_cur = df.iloc[i]['Close']
        high_cur = df.iloc[i]['High']
        low_cur = df.iloc[i]['Low']
        # Bullish engulfing
        engulfing = (
            (close_cur > open_cur)
            and (close_prev < open_prev)
            and (close_cur >= open_prev)
            and (open_cur <= close_prev)
        )
        # Hammer: long lower shadow
        body = abs(close_cur - open_cur)
        lower_shadow = min(open_cur, close_cur) - low_cur
        hammer = (lower_shadow >= 2 * body) and (close_cur > open_cur)
        # Marubozu: opens near low, closes near high
        candle_range = high_cur - low_cur if (high_cur - low_cur) != 0 else 1
        marubozu = (
            (abs(open_cur - low_cur) <= 0.1 * candle_range)
            and (abs(close_cur - high_cur) <= 0.1 * candle_range)
            and (close_cur > open_cur)
        )
        # Morning star: three‑bar pattern
        morning_star = False
        if i >= 2:
            open_2 = df.iloc[i - 2]['Open']
            close_2 = df.iloc[i - 2]['Close']
            open_1 = df.iloc[i - 1]['Open']
            close_1 = df.iloc[i - 1]['Close']
            prev_body = abs(close_2 - open_2)
            indecision_body = abs(close_1 - open_1)
            morning_star = (
                (close_2 < open_2)
                and (indecision_body <= 0.5 * prev_body)
                and (close_cur > open_cur)
                and (close_cur >= open_2)
            )
        results.append(bool(engulfing or hammer or marubozu or morning_star))
    return pd.Series(results, index=df.index)