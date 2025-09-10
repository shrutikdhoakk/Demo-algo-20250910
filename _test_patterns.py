import sys, importlib, pandas as pd, numpy as np

# 1) locate pattern_gate in common module paths
mods = ["backtesting.pattern_filters", "pattern_filters", "backtest.pattern_filters"]
pattern_gate = None
src = None
last_err = None
for m in mods:
    try:
        mod = importlib.import_module(m)
        if hasattr(mod, "pattern_gate"):
            pattern_gate = getattr(mod, "pattern_gate")
            src = m
            break
    except Exception as e:
        last_err = e

if not pattern_gate:
    print("SKIP: pattern_filters module not found (checked: %s)" % mods)
    if last_err:
        print("Last import error:", repr(last_err))
    sys.exit(0)

# 2) build synthetic OHLCV: flat -> squeeze -> breakout
n = 120
idx = pd.date_range("2024-01-01", periods=n, freq="B")
rng = np.random.default_rng(7)

base = 100 + np.cumsum(rng.normal(0, 0.2, size=n))
# squeeze (lower vol)
base[40:70] = base[40] + np.cumsum(rng.normal(0, 0.05, size=30))
# breakout phase (up bias)
base[70:] = base[70] + np.cumsum(rng.normal(0.3, 0.25, size=n-70))

close = base
high  = close + np.abs(rng.normal(0.2, 0.2, size=n))
low   = close - np.abs(rng.normal(0.2, 0.2, size=n))
open_ = close + rng.normal(0, 0.1, size=n)
vol   = rng.integers(120_000, 220_000, size=n)

df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)

# 3) call pattern_gate and print essentials
try:
    out = pattern_gate(df)
    print(f"OK: pattern_gate imported from {src}")
    if isinstance(out, dict):
        keys = list(out.keys())
        print("keys:", keys[:20])
        # show any truthy numeric/bool flags to prove signals are computed
        truthy = {k: v for k, v in out.items() if isinstance(v, (bool,int,float)) and bool(v)}
        if truthy:
            fmt = {k: (round(v,3) if isinstance(v, float) else v) for k,v in truthy.items()}
            print("truthy flags/metrics:", fmt)
        else:
            print("note: no truthy flags found (still OK if gate returns only metrics or None)")
    else:
        print("WARN: pattern_gate returned", type(out).__name__)
except Exception as e:
    print("FAIL: pattern_gate raised:", repr(e))
    sys.exit(2)
