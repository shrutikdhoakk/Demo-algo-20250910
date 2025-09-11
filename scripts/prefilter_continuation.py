import argparse, sys, time
import pandas as pd, numpy as np
from datetime import datetime, timedelta
try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance", file=sys.stderr); sys.exit(1)

def load_syms(p):
    df = pd.read_csv(p); col=df.columns[0]
    return df[col].dropna().astype(str).str.strip().tolist(), col

def fetch(sym):
    # robust: use history(period) to avoid date math issues
    try:
        d = yf.Ticker(sym).history(period="420d", auto_adjust=False)
        if isinstance(d, pd.DataFrame) and not d.empty and "Close" in d.columns:
            return d
        if not sym.endswith(".NS"):
            d = yf.Ticker(sym+".NS").history(period="420d", auto_adjust=False)
            if isinstance(d, pd.DataFrame) and not d.empty and "Close" in d.columns:
                return d
    except Exception:
        pass
    return pd.DataFrame()

def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def sma(x, n): return x.rolling(n).mean()

def atr(df, n=14):
    h,l,c = df["High"],df["Low"],df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100/(1+rs))

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--keep_top", type=int, default=200)
ap.add_argument("--min_price", type=float, default=20.0)
args = ap.parse_args()

syms, col = load_syms(args.input)
rows=[]
for i,s in enumerate(syms,1):
    try:
        df = fetch(s)
        if df.empty or df.shape[0] < 120: 
            continue
        df = df.dropna(subset=["Open","High","Low","Close"])

        last_close = float(df["Close"].iloc[-1])
        if last_close < args.min_price:
            continue

        # trend filters
        sma50  = sma(df["Close"], 50)
        sma200 = sma(df["Close"], 200)
        trend_ok = (sma50.iloc[-1] > sma200.iloc[-1]) if not (pd.isna(sma50.iloc[-1]) or pd.isna(sma200.iloc[-1])) else False

        # momentum
        if df.shape[0] < 100:
            continue
        rs84 = (df["Close"].iloc[-1] / df["Close"].iloc[-85]) - 1.0

        # volatility & pullback
        atr14 = atr(df,14)
        atrp  = (atr14 / df["Close"]).clip(0,1)
        atr_last = float(atrp.iloc[-1]) if not pd.isna(atrp.iloc[-1]) else np.nan

        rsi14 = rsi(df["Close"],14)
        rsi_last = float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else np.nan

        # mid-band ATR% preferred (not too quiet, not too wild)
        atr_ok = 0.02 <= (atr_last if not np.isnan(atr_last) else 0) <= 0.08

        # gentle pullback: RSI between 48–62 is ideal; outside band penalized
        if np.isnan(rsi_last):
            rsi_score = 0.4
        elif 48 <= rsi_last <= 62:
            rsi_score = 1.0
        elif 45 <= rsi_last <= 70:
            rsi_score = 0.7
        else:
            rsi_score = 0.3

        if not (trend_ok and atr_ok and rs84 > 0):
            continue

        rows.append({
            col: s if s.endswith(".NS") else s+".NS",
            "rs84": rs84,
            "atrp_last": atr_last,
            "rsi_last": rsi_last,
            "rsi_score": rsi_score
        })
    except Exception:
        pass
    finally:
        if i % 25 == 0:
            time.sleep(0.15)

if not rows:
    print("No symbols passed continuation filter.", file=sys.stderr); sys.exit(2)

X = pd.DataFrame(rows)
X["rs_rank"] = X["rs84"].rank(pct=True)

# score: mostly momentum, with a pullback bonus
X["score"] = 0.80*X["rs_rank"] + 0.20*X["rsi_score"]
X = X.sort_values("score", ascending=False).head(args.keep_top)

out = X[[col]].copy()
out.to_csv(args.output, index=False)
print(f"Saved {out.shape[0]} symbols to {args.output}")
print(out.head(10).to_string(index=False))
