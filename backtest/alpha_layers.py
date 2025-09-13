from __future__ import annotations
import os
from typing import Dict
import numpy as np
import pandas as pd

def _get_env_floats(key: str, default: list[float]) -> list[float]:
    raw = os.getenv(key, "")
    if not raw:
        return default
    return [float(x) for x in raw.split(",") if x.strip()]

def momentum_score(close: pd.Series) -> pd.Series:
    """
    Weighted sum of % changes over multiple lookbacks.
    - Forces numeric dtype to avoid object-Series ops errors.
    - Uses column_stack (not vstack) and explicit float arrays for Pyright/mypy.
    - Avoids deprecated implicit padding in pct_change (fill_method=None).
    """
    s = pd.to_numeric(close, errors="coerce").astype(float)

    lbs = _get_env_floats("MOM_LOOKBACKS_D", [126, 63, 21])
    wts = np.array(_get_env_floats("MOM_WEIGHTS", [0.5, 0.3, 0.2]), dtype=float)
    if wts.sum() == 0:
        wts = np.array([1.0] * len(lbs), dtype=float)
    wts = wts / wts.sum()

    arrs: list[np.ndarray] = []
    for lb in lbs:
        lb_i = int(lb)
        pct = s.pct_change(lb_i, fill_method=None) \
              .replace([np.inf, -np.inf], np.nan) \
              .fillna(0.0)
        arrs.append(pct.to_numpy(dtype=float))

    if len(arrs) == 0:
        sc = np.zeros(len(s), dtype=float)
    else:
        M = np.column_stack(tuple(arrs))  # shape (T, K)
        sc = (M @ wts).astype(float)

    return pd.Series(sc, index=s.index, name="mom_score")

def build_momentum_mask(price_panel: Dict[str, pd.DataFrame], pctile: float | None = None) -> Dict[str, pd.Series]:
    """
    Returns symbol->Series[bool] where True means the symbol is in the top momentum quantile that day.
    Expects each df to have a 'Close' column.
    """
    thr = float(os.getenv("MOM_PCTILE", "0.70" if pctile is None else pctile))

    # Use the first dataframe's index as the date grid (same as engine)
    idx = next(iter(price_panel.values())).index
    symbols = list(price_panel.keys())

    score_df = pd.DataFrame(index=idx, columns=symbols, dtype=float)
    for sname, df in price_panel.items():
        c = df["Close"]
        score = momentum_score(c).reindex(idx).ffill().fillna(0.0)
        score_df[sname] = score.astype(float)

    row_q = score_df.quantile(thr, axis=1)
    mask_df = score_df.ge(row_q, axis=0)

    return {sname: mask_df[sname] for sname in symbols}
