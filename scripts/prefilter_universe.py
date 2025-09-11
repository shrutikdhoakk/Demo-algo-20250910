import argparse, sys, time
import pandas as pd, numpy as np
from datetime import datetime, timedelta, UTC
try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance", file=sys.stderr); sys.exit(1)

def load_symbols(p):
    df = pd.read_csv(p); col = df.columns[0]
    return df[col].dropna().astype(str).str.strip().tolist(), col

def hist(sym, period="400d"):
    try:
        return yf.Ticker(sym).history(period=period, auto_adjust=False)
    except Exception:
        return pd.DataFrame()

def fetch(sym):
    d = hist(sym)
    if isinstance(d, pd.DataFrame) and not d.empty and "Close" in d.columns:
        return d
    if not sym.endswith(".NS"):
        d = hist(sym + ".NS")
        if isinstance(d, pd.DataFrame) and not d.empty and "Close" in d.columns:
            return d
    return pd.DataFrame()

def atr(df, n=14):
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def bbw(df, n=20, k=2.0):
    m = df["Close"].rolling(n).mean()
    s = df["Close"].rolling(n).std(ddof=0)
    upper, lower = m + k*s, m - k*s
    return (upper - lower) / m

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--days", type=int, default=360)
ap.add_argument("--min_price", type=float, default=10.0)
ap.add_argument("--keep_top", type=int, default=220)
args = ap.parse_args()

syms, col = load_symbols(args.input)
rows = []
for i,s in enumerate(syms,1):
    try:
        df = fetch(s)
        if df.empty: 
            continue
        df = df.dropna(subset=["Open","High","Low","Close"])
        if len(df) < 60:
            continue

        last_close = float(df["Close"].iloc[-1])
        if last_close < args.min_price:
            continue

        df["ATR14"]  = atr(df,14)
        df["ATR_PCT"] = (df["ATR14"] / df["Close"]).clip(lower=0, upper=1)

        # RS lookback: use up to 84d if available, else fall back to len-1
        look = min(84, max(1, len(df)-1))
        rs = (df["Close"].iloc[-1] / df["Close"].iloc[-look]) - 1.0

        # BBW: robust scaling using last 60 bars; if insufficient data, fall back
        bbw_series = bbw(df,20,2.0)
        tail = bbw_series.tail(60)
        if tail.dropna().empty:
            bbw_scaled = 1.0
        else:
            denom = tail.quantile(0.95)
            bbw_scaled = float((tail.iloc[-1] / (denom if denom and denom>0 else 1.0)))
            bbw_scaled = float(np.clip(bbw_scaled, 0, 1))

        rows.append({
            col: s if s.endswith(".NS") else s + ".NS",
            "rs": rs,
            "bbw_scaled": bbw_scaled
        })
    except Exception:
        pass
    finally:
        if i % 25 == 0:
            time.sleep(0.2)

if not rows:
    print("No symbols passed filters.", file=sys.stderr); sys.exit(2)

X = pd.DataFrame(rows)
X["rs_rank"]       = X["rs"].rank(pct=True)
X["bbw_rank_inv"]  = 1 - X["bbw_scaled"].rank(pct=True)
X["score"]         = 0.70*X["rs_rank"] + 0.30*X["bbw_rank_inv"]
X = X.sort_values("score", ascending=False).head(args.keep_top)

out = X[[col]].copy()
out.to_csv(args.output, index=False)
print(f"Saved {out.shape[0]} symbols to {args.output}")
print(out.head(10).to_string(index=False))
