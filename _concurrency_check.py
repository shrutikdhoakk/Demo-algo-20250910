import pandas as pd
from pathlib import Path

def max_concurrency(trades_path: str):
    df = pd.read_csv(trades_path)
    # Heuristics for entry/exit columns
    cols = {c.lower(): c for c in df.columns}
    entry = next((cols[c] for c in cols if "entry" in c and "date" in c or "time" in c), None)
    exitc = next((cols[c] for c in cols if ("exit" in c or "close" in c) and ("date" in c or "time" in c)), None)
    if entry is None or exitc is None:
        raise SystemExit(f"Couldn't find entry/exit date columns in {trades_path}. Columns: {list(df.columns)}")
    df[entry] = pd.to_datetime(df[entry])
    df[exitc] = pd.to_datetime(df[exitc])
    # sweep days between global min/max
    days = pd.date_range(df[entry].min(), df[exitc].max(), freq="D")
    # count open positions per day
    open_counts = []
    for d in days:
        open_counts.append(((df[entry] <= d) & (df[exitc] >= d)).sum())
    return max(open_counts), (sum(open_counts)/len(open_counts))

files = [
    "backtest_trades_buf0p35_trail1p10_atrp0p04_p3_2022_2024.csv",
    "backtest_trades_buf0p35_trail1p10_atrp0p04_p5_2022_2024.csv",
]
for f in files:
    m, avg = max_concurrency(f)
    print(f"{f}: max_concurrent={m}, avg_concurrent={avg:.2f}")
