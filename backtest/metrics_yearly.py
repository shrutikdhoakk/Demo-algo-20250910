# backtest/metrics_yearly.py
from __future__ import annotations
import argparse, sys, os
from typing import Optional
import numpy as np
import pandas as pd

def _ensure_dt_series(raw: pd.Series) -> pd.Series:
    """Coerce to float Series with DatetimeIndex, dedupe, and collapse to 1 row/day (last)."""
    if not isinstance(raw, pd.Series):
        raise TypeError("Expected a pandas Series.")
    idx = pd.to_datetime(raw.index, errors="coerce")
    vals = pd.to_numeric(raw.to_numpy(), errors="coerce").astype(float)
    ser = pd.Series(vals, index=idx).dropna()
    # drop duplicate timestamps, then one sample per calendar day via Grouper
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    ser = ser.groupby(pd.Grouper(freq="D")).last().dropna()
    # guarantee DatetimeIndex for type checkers:
    ser.index = pd.DatetimeIndex(ser.index)
    return ser

def compute_yearly_returns(s: pd.Series) -> pd.Series:
    ser = _ensure_dt_series(s)
    # 'Y' deprecated ⇒ use year-end anchored freq
    year_end = ser.resample("YE-DEC").ffill()
    return year_end.pct_change().dropna()

def _years_from_index(idx: pd.Index) -> np.ndarray:
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(0)
    didx: pd.DatetimeIndex = pd.DatetimeIndex(idx)
    return didx.year.astype(int).to_numpy()

def pretty_print(yr: pd.Series) -> None:
    if yr.empty:
        print("No yearly returns found."); return
    years = _years_from_index(yr.index)
    ret_pct = np.round(yr.to_numpy(dtype=float) * 100.0, 2)
    df = pd.DataFrame({"Year": years, "Return %": ret_pct})
    wy = max(4, df["Year"].astype(str).map(len).max())
    wr = max(8, df["Return %"].astype(str).map(len).max())
    print("\nYearly Returns")
    print("-" * (wy + wr + 3))
    print(f'{"Year":<{wy}} | {"Return %":>{wr}}')
    print("-" * (wy + wr + 3))
    for _, r in df.iterrows():
        print(f'{int(r["Year"]):<{wy}} | {r["Return %"]:>{wr}.2f}')
    print("-" * (wy + wr + 3))

def auto_pick_value_col(df: pd.DataFrame) -> str:
    for c in ["equity", "nav", "portfolio", "curve", "value", "balance"]:
        if c in df.columns: return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric: raise ValueError("Could not find a numeric column for equity values.")
    return numeric[0]

def main():
    ap = argparse.ArgumentParser(description="Compute per-year returns from an equity curve CSV.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--value-col", default=None)
    ap.add_argument("--save-csv", default="out/yearly_returns.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # pick columns
    dtcol: Optional[str] = args.date_col if (args.date_col and args.date_col in df.columns) else None
    if dtcol is None:
        for c in ["date","Date","dt","timestamp","time"]:
            if c in df.columns: dtcol = c; break
        if dtcol is None: dtcol = df.columns[0]
    vcol = args.value_col if (args.value_col and args.value_col in df.columns) else auto_pick_value_col(df)

    idx = pd.to_datetime(df[dtcol], errors="coerce")
    vals = pd.to_numeric(df[vcol], errors="coerce").astype(float)
    ser = pd.Series(vals.to_numpy(), index=idx).dropna()

    yr = compute_yearly_returns(ser)
    pretty_print(yr)

    # ensure folder exists
    parent = os.path.dirname(args.save_csv)
    if parent: os.makedirs(parent, exist_ok=True)
    out = pd.DataFrame({"year": _years_from_index(yr.index), "return_pct": (yr.to_numpy(dtype=float) * 100.0)})
    out.to_csv(args.save_csv, index=False)
    print(f"\nSaved: {args.save_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
