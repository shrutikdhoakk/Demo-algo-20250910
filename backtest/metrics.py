"""Performance metrics for the backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(dates: pd.Series, values: pd.Series) -> float:
    """Compound annual growth rate of an equity curve.

    Parameters
    ----------
    dates : pandas.Series
        Series of datetime64 values corresponding to the equity curve.
    values : pandas.Series
        Series of portfolio values aligned with ``dates``.

    Returns
    -------
    float
        Annualised return expressed as a decimal (0.10 = 10 %).  If
        fewer than two data points are provided or the time span is
        zero, returns 0.0.
    """
    if len(values) < 2:
        return 0.0
    t0, tN = dates.iloc[0], dates.iloc[-1]
    days = (tN - t0).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    start, end = values.iloc[0], values.iloc[-1]
    if start <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def max_drawdown(values: pd.Series) -> float:
    """Maximum drawdown of an equity curve.

    The drawdown is defined as the percentage decline from the running
    maximum.  The maximum drawdown is the minimum of all drawdown
    values (most negative value).
    """
    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    return drawdown.min()