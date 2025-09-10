from __future__ import annotations
from pathlib import Path
import numpy as np
"""Data ingestion routines for OHLCV files.

By default the backtest engine expects to find individual CSV files for
each symbol under `data/ohlcv/`.  Each file should contain at least
columns named `Date`, `Open`, `High`, `Low`, `Close` and `Volume`.
Additional columns (such as beta or delta) will be carried through to
the strategy if present.
"""
import os
import pandas as pd


def load_csv(symbol: str) -> pd.DataFrame:
    """Load symbol CSV -> tz-naive DatetimeIndex + numeric OHLCV + _WARMUP."""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "ohlcv"
    path = DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data file for symbol {symbol}: {path}")

    df = pd.read_csv(path)

    # Normalize headers
    col_map = {
        "date":"Date", "open":"Open", "high":"High", "low":"Low",
        "close":"Close", "adj close":"Adj Close", "adj_close":"Adj Close",
        "volume":"Volume",
    }
    ren = {}
    for c in list(df.columns):
        key = str(c).strip().lower()
        if key in col_map and c != col_map[key]:
            ren[c] = col_map[key]
    if ren:
        df = df.rename(columns=ren)

    if "Date" not in df.columns:
        raise ValueError(f"Data file for {symbol} must contain a Date column")

    # Parse Date, strip timezone to tz-naive
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    if getattr(df["Date"].dt, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    # Coerce numerics
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort, dedupe dates, set Date as tz-naive index
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").set_index("Date")
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Warmup mask (skip early bars in engine)
    warmup = 200
    df["_WARMUP"] = False
    if len(df) > warmup:
        df.iloc[:warmup, df.columns.get_loc("_WARMUP")] = True

    return df
