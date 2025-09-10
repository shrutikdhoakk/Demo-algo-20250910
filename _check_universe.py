import sys, pathlib
import pandas as pd

candidates = [
    pathlib.Path("data/symbols_nifty500_clean.csv"),
    pathlib.Path("data/symbols.csv"),
    pathlib.Path("data/universe.csv"),
]

src = next((p for p in candidates if p.exists()), None)
if not src:
    print("FAIL: no universe file found at:", ", ".join(str(p) for p in candidates))
    sys.exit(1)

try:
    df = pd.read_csv(src)
except Exception:
    # fallback for headerless/odd CSVs
    df = pd.read_csv(src, header=None, usecols=[0])
    df.columns = ["symbol"]

# normalize to a single 'symbol' column
lower = [c.lower() for c in getattr(df, "columns", [])]
if "symbol" in lower:
    col = df.columns[lower.index("symbol")]
    df = df.rename(columns={col: "symbol"})[["symbol"]]
else:
    df = df.iloc[:, [0]].copy()
    df.columns = ["symbol"]

# clean
df["symbol"] = df["symbol"].astype(str).str.strip()
df = df[df["symbol"].ne("")].drop_duplicates()

print(f"OK: universe file -> {src}")
print("Universe size:", len(df))
print("Head:")
print(df.head(12).to_string(index=False))

if len(df) == 0:
    print("FAIL: universe is empty"); sys.exit(2)

# create a small smoke list for quick tests (if not already present)
smoke = pathlib.Path("data/smoke30.csv")
if not smoke.exists():
    df.head(30).to_csv(smoke, index=False, header=True)
    print(f"Created {smoke} with first 30 symbols for smoke tests.")
else:
    print(f"Found existing {smoke} (not overwritten).")
