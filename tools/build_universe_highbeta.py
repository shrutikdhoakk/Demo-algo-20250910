import argparse, pandas as pd, numpy as np, yfinance as yf, sys, time

def atr14(df):
    h,l,c = df["High"], df["Low"], df["Close"]
    tr = np.maximum(h-l, np.maximum((h-c.shift()).abs(), (l-c.shift()).abs()))
    return pd.Series(tr).rolling(14).mean()

def load_list(path):
    df = pd.read_csv(path)
    # try common headers, else first column
    for col in ["symbol","SYMBOL","ticker","TICKER"]:
        if col in df.columns: s = df[col].astype(str).tolist(); break
    else: s = df.iloc[:,0].astype(str).tolist()
    # append .NS if missing a suffix
    out=[]
    for x in s:
        out.append(x if "." in x else f"{x}.NS")
    return list(dict.fromkeys(out))  # unique, keep order

def score_symbol(sym):
    try:
        df = yf.download(sym, period="3y", interval="1d", progress=False, auto_adjust=False)
        if df.empty: return None
        df = df[["High","Low","Close"]].dropna()
        a = atr14(df)
        atr_pct = (a / df["Close"]).dropna()
        if atr_pct.empty: return None
        return float(np.nanmedian(atr_pct.values))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--outfile", default="data/universe_highbeta.csv")
    args = ap.parse_args()

    syms = load_list(args.symbols)
    rows=[]
    for i,s in enumerate(syms,1):
        v = score_symbol(s)
        if v is not None: rows.append((s, v))
        if i % 25 == 0: time.sleep(0.5)  # be polite
    if not rows:
        print("No scores computed.", file=sys.stderr); sys.exit(2)

    df = pd.DataFrame(rows, columns=["symbol","atr_pct_med"]).sort_values("atr_pct_med", ascending=False)
    df_top = df.head(args.top)
    # Save just the symbols (engine expects a simple CSV list)
    df_top[["symbol"]].to_csv(args.outfile, index=False)
    print(f"Saved {len(df_top)} symbols to {args.outfile}")
    print(df_top.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
