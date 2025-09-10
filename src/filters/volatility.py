"""Volatility and derivative filters.

These helpers encapsulate simple checks on beta and delta values.  If a
value is missing or NaN the filter passes.  Otherwise the numeric
thresholds defined by the strategy configuration are applied.
"""

from __future__ import annotations

import numpy as np

def beta_filter(beta: float, beta_min: float = 0.9, beta_max: float = 1.5) -> bool:
    """Return True if beta is within the allowed range or missing."""
    if beta is None or (isinstance(beta, float) and np.isnan(beta)):
        return True
    return beta_min <= beta <= beta_max


def delta_filter(delta: float, delta_min: float = 0.7) -> bool:
    """Return True if delta exceeds the minimum threshold or is missing."""
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return True
    return delta >= delta_min