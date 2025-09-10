# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")  # headless backend so it works in terminals/servers

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_curve(path_str: str):
    f = Path(path_str)
    if not f.exists():
        raise SystemExit(f"Missing file: {f}")
    df = pd.read_csv(f)

    # pick an equity-like column
    candidates = [c for c in df.columns if c.lower() in ("equity","nav","equity_curve","cum_pnl")]
    if not candidates:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise SystemExit(f"No numeric columns found in {f.name}")
        col = numeric_cols[-1]
    else:
        col = candidates[0]

    plt.figure(figsize=(9, 4.5))
    plt.plot(df[col].values)
    plt.title(f"Equity Curve - {f.name}")
    plt.xlabel("Bars")
    plt.ylabel(col)
    plt.tight_layout()
    out_path = f.with_suffix(".png")
    plt.savefig(out_path, dpi=160)
    print("Saved", out_path.name)

save_curve("equity_curve_buf0p35_trail1p10_atrp0p04_p3_2022_2024.csv")
save_curve("equity_curve_buf0p35_trail1p10_atrp0p04_p5_2022_2024.csv")
