"""Simple backtest engine for the Breakout Momentum strategy."""
from __future__ import annotations

# --- pattern gate hook (auto-injected) ---
import os
try:
    from .pattern_gate import pattern_gate
except Exception:
    pattern_gate = None

USE_PATTERN_GATE = os.environ.get("USE_PATTERN_GATE", "0") == "1"

def _first_df_for_gate(ns):
    """Return the first pandas DataFrame in the given namespace that
    looks like OHLC data (has open/high/low/close)."""
    try:
        import pandas as pd
    except Exception:
        return None
    for k in ('df_sym', 'df', 'ohlc', 'bars', 'data', 'frame', 'window'):
        v = ns.get(k)
        try:
            if isinstance(v, pd.DataFrame):
                cols = {c.lower() for c in v.columns}
                if {'open', 'high', 'low', 'close'}.issubset(cols):
                    return v
        except Exception:
            pass
    return None

def _apply_pattern_gate(df, buy_signal):
    if not USE_PATTERN_GATE or not buy_signal or pattern_gate is None:
        return buy_signal
    try:
        return bool(buy_signal and pattern_gate(df).get("ok", False))
    except Exception:
        return False
# --- end pattern gate hook ---

import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from dataclasses import fields, MISSING
from types import SimpleNamespace as _SNS

# Import available strategies.  We support both v3 (default) and v7 variant.
from src.strategy.breakout_momentum_v3 import StrategyConfig, BreakoutMomentumV3
try:
    from src.strategy.breakout_momentum_v7 import StrategyConfigV7, BreakoutMomentumV7  # type: ignore
except Exception:
    StrategyConfigV7 = None  # type: ignore
    BreakoutMomentumV7 = None  # type: ignore

from src.data.ingest import load_csv
from .metrics import cagr, max_drawdown


# --- RSI + Volume helpers (Step 1: returns-boost gate) ---
def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n // 2)).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=close.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index)
    up   = gain.ewm(alpha=1/period, adjust=False).mean()
    dn   = loss.ewm(alpha=1/period, adjust=False).mean()
    rs   = up / dn.replace(0, np.nan)
    out  = 100 - (100 / (1 + rs))
    # FutureWarning-safe: use bfill() instead of fillna(method="bfill")
    return out.bfill().fillna(50)

def confirm_breakout_for_index(
    df: pd.DataFrame,
    idx: int,
    high_n: int = 20,
    rsi_len: int = 14,
    rsi_thr: float = 55.0,
    vol_len: int = 20,
    vol_mult: float = 1.5,
    require_volume: bool = True,
) -> bool:
    """Gate: uses precomputed columns (RSI14, HH_N_PREV, VOL_SMA_N) if present."""
    if idx <= 0 or idx >= len(df):
        return False

    # Close (case-insensitive)
    close = df["Close"] if "Close" in df.columns else df["close"]

    # 1) Breakout vs prior rolling high (cached with fallback)
    hh_prev = df.get("HH_N_PREV")
    if hh_prev is None or pd.isna(hh_prev.iloc[idx]):
        high = df["High"] if "High" in df.columns else df["high"]
        hh_prev = high.rolling(high_n, min_periods=high_n).max().shift(1)
    breakout = bool(pd.notna(hh_prev.iloc[idx]) and (close.iloc[idx] > hh_prev.iloc[idx]))

    # 2) RSI confirmation (cached with fallback)
    rsi_col = df.get("RSI14")
    if rsi_col is None or pd.isna(rsi_col.iloc[idx]):
        rsi_col = _rsi(close, rsi_len).bfill().fillna(50)
    rsi_ok = bool(rsi_col.iloc[idx] >= rsi_thr)

    # 3) Volume confirmation (cached if possible)
    vol_ok = False
    vol_sma = df.get("VOL_SMA_N")
    if vol_sma is not None:
        if "Volume" in df.columns: vol = df["Volume"]
        elif "volume" in df.columns: vol = df["volume"]
        else: vol = None
        if vol is not None and pd.notna(vol_sma.iloc[idx]):
            vol_ok = bool(vol.iloc[idx] >= vol_mult * float(vol_sma.iloc[idx]))
        else:
            vol_ok = (not require_volume)
    else:
        # no volume in data → allow if not required
        vol_ok = (not require_volume)

    return bool(breakout and rsi_ok and vol_ok)
# --- end helpers ---


def _parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d")).normalize()


def _read_universe_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if len(df.columns) == 1:
        return df.iloc[:, 0].astype(str).dropna().tolist()
    if "symbol" in cols:
        return df[df.columns[cols.index("symbol")]].astype(str).dropna().tolist()
    if "symbols" in cols:
        return df[df.columns[cols.index("symbols")]].astype(str).dropna().tolist()
    return df.iloc[:, 0].astype(str).dropna().tolist()


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.to_datetime(df.index, errors="coerce")
    mask = ~pd.isna(parsed)
    df = df.loc[mask].copy()
    df.index = pd.DatetimeIndex(parsed[mask])
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def _pos_at(df: pd.DataFrame, date: pd.Timestamp) -> Optional[int]:
    idx = int(df.index.get_indexer([date])[0])
    return None if idx == -1 else idx


class BacktestEngine:
    def __init__(
        self,
        config_path: str = "backtest/config.yaml",
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        universe_csv: Optional[str] = None,
        max_positions_override: Optional[int] = None,
    ):
        # --- load YAML ---
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f) or {}

            # Flatten nested 'strategycfg' into top-level keys (keep top-level on collision)
            scfg = self.cfg.get('strategycfg')
            if isinstance(scfg, dict):
                for _k, _v in scfg.items():
                    self.cfg.setdefault(_k, _v)
            print(
                "USING_PARAMS", "scope=engine_flat",
                f"breakout_atr_buf={self.cfg.get('breakout_atr_buf')}",
                f"trail_atr_mult={self.cfg.get('trail_atr_mult')}",
                f"atr_pct_max={self.cfg.get('atr_pct_max')}",
                sep=" | ",
            )
        # log what we loaded
        try:
            import json, hashlib
            _hash = hashlib.sha1(json.dumps(self.cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
        except Exception:
            _hash = "nohash"
        print(
            "USING_PARAMS", "scope=engine", f"hash={_hash}",
            f"breakout_atr_buf={self.cfg.get('breakout_atr_buf')}",
            f"trail_atr_mult={self.cfg.get('trail_atr_mult')}",
            f"atr_pct_max={self.cfg.get('atr_pct_max')}",
            sep=" | ",
        )

        # --- risk/settings ---
        self.total_capital: float = float(self.cfg.get("TOTAL_CAPITAL", 10000.0))
        self.max_invested: float  = float(self.cfg.get("MAX_INVESTED", 5000.0))
        self.per_trade_risk: float = float(self.cfg.get("PER_TRADE_RISK", 1000.0))
        self.slippage: float      = float(self.cfg.get("SLIPPAGE", 0.001))
        self.fee: float           = float(self.cfg.get("FEE", 0.0003))

        # positions
        default_max_pos = int(self.cfg.get("MAX_POSITIONS", 5))
        self.max_positions: int = int(max_positions_override or default_max_pos)

        # universe
        if universe_csv:
            self.symbols: List[str] = _read_universe_csv(universe_csv)
        else:
            self.symbols = list(self.cfg.get("SYMBOLS", []))

        # --- Determine which strategy to use (v3 by default) ---
        strategy_name = (self.cfg.get('STRATEGY') or self.cfg.get('STRATEGY_VERSION') or 'v3').lower()

        # Build a read-only cfg view based on the selected strategy's default dataclass.
        if strategy_name == 'v7' and StrategyConfigV7 is not None:
            dataclass_type = StrategyConfigV7
        else:
            dataclass_type = StrategyConfig

        base: Dict[str, Any] = {}
        try:
            for f in fields(dataclass_type):
                # read field defaults without creating the object
                if f.default is not MISSING:
                    base[f.name] = f.default
                else:
                    dfac = getattr(f, 'default_factory', MISSING)
                    if dfac is not MISSING:
                        try:
                            base[f.name] = dfac()
                        except Exception:
                            pass
        except Exception as e:
            print("WARN | could not read StrategyConfig defaults:", e)

        # Override defaults with YAML values when present (only known keys)
        for k, v in self.cfg.items():
            if k in base:
                base[k] = v

        sc_view = _SNS(**base)
        if strategy_name == 'v7' and BreakoutMomentumV7 is not None:
            print(
                "USING_PARAMS", "scope=engine->strategycfg(v7)",
                " | ".join(f"{k}={getattr(sc_view,k, None)}" for k in base.keys()),
                sep=" | ",
            )
            self.strategy = BreakoutMomentumV7(sc_view)
        else:
            # Merge legacy overrides for v3
            for k in ("breakout_atr_buf", "trail_atr_mult", "atr_pct_max"):
                if k in self.cfg and self.cfg[k] is not None:
                    try:
                        setattr(sc_view, k, float(self.cfg[k]))
                    except Exception:
                        setattr(sc_view, k, self.cfg[k])
            print(
                "USING_PARAMS", "scope=engine->strategycfg(v3)",
                f"breakout_atr_buf={getattr(sc_view,'breakout_atr_buf', None)}",
                f"trail_atr_mult={getattr(sc_view,'trail_atr_mult', None)}",
                f"atr_pct_max={getattr(sc_view,'atr_pct_max', None)}",
                sep=" | ",
            )
            self.strategy = BreakoutMomentumV3(sc_view)

        # --- data prep (symbols) ---
        self.data: Dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            df = load_csv(sym)
            if start_date is not None: df = df[df.index >= start_date]
            if end_date   is not None: df = df[df.index <= end_date]
            if df.empty: continue
            df = self.strategy.compute_features(df)
            df = _normalize_index(df)

            # --- precompute for fast confirmation gate ---
            # Use case-insensitive access
            close = df["Close"] if "Close" in df.columns else df["close"]
            high  = df["High"]  if "High"  in df.columns else df["high"]

            # RSI cache
            df["RSI14"] = _rsi(close, int(os.environ.get("RSI_LEN", "14"))).bfill().fillna(50)

            # Rolling high (prior bar) cache
            high_n = int(os.environ.get("HIGH_N", "20"))
            df["HH_N_PREV"] = high.rolling(high_n, min_periods=high_n).max().shift(1)

            # Volume SMA cache (only if a volume column is present)
            if "Volume" in df.columns:
                vol = df["Volume"]
            elif "volume" in df.columns:
                vol = df["volume"]
            else:
                vol = None

            if vol is not None:
                vol_len = int(os.environ.get("VOL_LEN", "20"))
                df["VOL_SMA_N"] = vol.rolling(vol_len, min_periods=max(1, vol_len // 2)).mean()

            self.data[sym] = df

        # build union of dates
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index)
        dates_sorted = sorted(all_dates)
        if start_date is not None:
            dates_sorted = [d for d in dates_sorted if d >= start_date]
        if end_date is not None:
            dates_sorted = [d for d in dates_sorted if d <= end_date]
        self.dates = dates_sorted

        # --- load index for cross-sectional RS filter ---
        self.index_symbol = str(self.cfg.get("INDEX_SYMBOL") or os.environ.get("INDEX_SYMBOL") or "^NIFTY500")
        try:
            idx_df = load_csv(self.index_symbol)
            if start_date is not None: idx_df = idx_df[idx_df.index >= start_date]
            if end_date   is not None: idx_df = idx_df[idx_df.index <= end_date]
            idx_df = _normalize_index(idx_df)
            # prefer "Close" casing if present
            self.idx_close = (idx_df["Close"] if "Close" in idx_df.columns else idx_df.iloc[:, 0]).reindex(self.dates).astype(float)
            if self.idx_close.isna().all():
                self.idx_close = None
        except Exception:
            self.idx_close = None  # graceful: RS filter becomes a no-op if missing

        # --- cross-sectional RS panel: RS = Close / IndexClose, then pct-rank per day ---
        self.rs_top_pct = float(os.environ.get("RS_TOP_PCT", self.cfg.get("RS_TOP_PCT", 0.70)))
        self._RS_FILTER_ON = self.idx_close is not None

        if self._RS_FILTER_ON and len(self.data) > 0:
            # Build panel of Close prices aligned to engine dates
            close_panel = {}
            for sym, df in self.data.items():
                if "Close" in df.columns:
                    s = df["Close"].reindex(self.dates)
                else:
                    s = df.iloc[:, 0].reindex(self.dates)
                close_panel[sym] = s.astype(float)
            close_panel = pd.DataFrame(close_panel, index=self.dates)

            # Avoid division by zero / NaN
            idx = self.idx_close.replace(0, np.nan)
            rs_panel = close_panel.divide(idx, axis=0)

            # Percentile rank per day (cross-sectional)
            rs_pct = rs_panel.rank(axis=1, pct=True, method="min")

            # Push back to individual symbol frames
            for sym, df in self.data.items():
                df["RS_PCT"] = rs_pct.get(sym, pd.Series(index=self.dates, dtype=float)).reindex(df.index)

        # state
        self.capital: float = self.total_capital
        self.portfolio: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self._cand_total = 0
        self._cand_dates = 0
        self._orders_attempted = 0
        self._orders_filled = 0

        # Step-1 tunables (also overridable via ENV without changing code)
        self._RSI_LEN  = int(os.environ.get("RSI_LEN", "14"))
        self._RSI_THR  = float(os.environ.get("RSI_THR", "55"))
        self._VOL_LEN  = int(os.environ.get("VOL_LEN", "20"))
        self._VOL_MULT = float(os.environ.get("VOL_MULT", "1.5"))
        self._HIGH_N   = int(os.environ.get("HIGH_N", "20"))
        self._REQ_VOL  = os.environ.get("REQUIRE_VOLUME", "1") == "1"

    def run(self) -> None:
        for date in self.dates:
            # --- exits ---
            for sym in list(self.portfolio.keys()):
                pos = self.portfolio[sym]
                df_sym = self.data.get(sym)
                if df_sym is None: continue
                i = _pos_at(df_sym, date)
                if i is None: continue
                row = df_sym.iloc[i]
                price = float(row["Close"])

                exit_reason = None
                exit_price: Optional[float] = None

                # trailing logic
                if pos.get("trail_active", False):
                    trail_stop = price - pos["atr"] * getattr(self.strategy.cfg, "trail_atr_mult", getattr(self.strategy.cfg, "chandelier_mult", 1.3))
                    if trail_stop > pos["trail_stop"]:
                        pos["trail_stop"] = trail_stop
                else:
                    # optional pattern gate on trail activation
                    trail_trigger = (price - pos["entry"]) >= self.strategy.cfg.trail_trigger_atr * pos["atr"]
                    if _apply_pattern_gate(_first_df_for_gate(locals()), bool(trail_trigger)) and trail_trigger:
                        pos["trail_active"] = True
                        pos["trail_stop"] = price - pos["atr"] * getattr(self.strategy.cfg, "trail_atr_mult", getattr(self.strategy.cfg, "chandelier_mult", 1.3))

                if price <= pos["stop"]:
                    exit_price = price * (1 - self.slippage); exit_reason = "stop"
                elif price >= pos["target"]:
                    exit_price = price * (1 - self.slippage); exit_reason = "target"
                elif pos.get("trail_active", False) and price <= pos["trail_stop"]:
                    exit_price = price * (1 - self.slippage); exit_reason = "trail"
                elif self.strategy.should_early_exit(df_sym, i):
                    exit_price = price * (1 - self.slippage); exit_reason = "early"

                if exit_price is not None:
                    shares = pos["shares"]
                    proceeds = exit_price * shares
                    entry_cost = pos["entry"] * shares
                    self.capital += proceeds - self.fee * proceeds - self.fee * entry_cost
                    self.trades.append({
                        "symbol": sym,
                        "entry_date": pos["entry_date"],
                        "entry_price": pos["entry"],
                        "exit_date": date,
                        "exit_price": exit_price,
                        "shares": shares,
                        "reason": exit_reason,
                    })
                    del self.portfolio[sym]

            # --- mark to market ---
            invested_value = 0.0
            for sym, pos in self.portfolio.items():
                df_sym = self.data.get(sym)
                if df_sym is None: continue
                j = _pos_at(df_sym, date)
                if j is None: continue
                invested_value += float(df_sym.iloc[j]["Close"]) * pos["shares"]
            total_value = float(self.capital + invested_value)
            self.equity_curve.append({"date": date, "value": total_value})

            # --- entries ---
            candidates: List[tuple[float, str, pd.Series]] = []
            for sym, df_sym in self.data.items():
                if sym in self.portfolio: continue
                k = _pos_at(df_sym, date)
                if k is None: continue
                row = df_sym.iloc[k]
                if k == 0 or bool(row.get("_WARMUP", False)): continue
                prev = df_sym.iloc[k - 1]

                # Step 1 gate: Breakout + RSI + Volume
                if not confirm_breakout_for_index(
                    df_sym, k,
                    high_n=self._HIGH_N,
                    rsi_len=self._RSI_LEN,
                    rsi_thr=self._RSI_THR,
                    vol_len=self._VOL_LEN,
                    vol_mult=self._VOL_MULT,
                    require_volume=self._REQ_VOL,
                ):
                    continue

                # Step 2 gate: Relative Strength vs Index (top percentile)
                if self._RS_FILTER_ON:
                    rs_pct = row.get("RS_PCT", np.nan)
                    if not (pd.notna(rs_pct) and float(rs_pct) >= self.rs_top_pct):
                        continue

                # Existing strategy entry check (kept)
                if self.strategy.is_entry(row, prev):
                    candidates.append((self.strategy.score(row), sym, row))

            self._cand_total += len(candidates)
            if candidates: self._cand_dates += 1
            candidates.sort(reverse=True, key=lambda x: x[0])
            for score, sym, row in candidates:
                self._orders_attempted += 1
                if len(self.portfolio) >= self.max_positions: break

                df_sym = self.data.get(sym)
                if df_sym is None: continue
                atr_val = float(row["ATR14"])
                if pd.isna(atr_val) or atr_val <= 0: continue

                close_px = float(row["Close"])
                stop   = close_px - self.strategy.cfg.atr_stop_mult   * atr_val
                target = close_px + self.strategy.cfg.atr_target_mult * atr_val

                # --- trend / 52w-high proximity filter (kept from your code) ---
                k = _pos_at(df_sym, date)
                if k is None or k < 252:
                    continue
                sma200 = df_sym['Close'].rolling(200, min_periods=200).mean().iloc[k]
                hh252  = df_sym['Close'].rolling(252, min_periods=252).max().iloc[k]
                px     = float(df_sym['Close'].iloc[k])
                near_hi_pct = float(os.environ.get('RS_WITHIN_HI','0.05'))  # within 5% of 52w high by default
                if (px < float(sma200)) or (px < float(hh252) * (1 - near_hi_pct)):
                    continue
                # ---------------------------------------

                risk_per_share = close_px - stop
                if risk_per_share <= 0: continue

                # pattern-gate before sizing (uses k from _pos_at)
                if USE_PATTERN_GATE and pattern_gate is not None and k is not None:
                    try:
                        g = pattern_gate(df_sym, now_idx=df_sym.index[k])
                        if os.environ.get("GATE_DEBUG", "0") == "1":
                            print("GATE", sym, str(date), g)
                        if not g.get("ok", False):
                            continue
                    except Exception as e:
                        if os.environ.get("GATE_DEBUG", "0") == "1":
                            print("GATE_ERR", sym, str(date), repr(e))
                        continue

                shares = int((self.per_trade_risk * float(os.environ.get("RISK_BOOST", "1"))) / risk_per_share)
                if shares <= 0: continue

                cost = shares * close_px

                # cap by cash/invested
                current_invested = 0.0
                for sym2, pos2 in self.portfolio.items():
                    df2 = self.data.get(sym2)
                    if df2 is None: continue
                    p2 = _pos_at(df2, date)
                    if p2 is None: continue
                    current_invested += float(df2.iloc[p2]["Close"]) * pos2["shares"]

                lev = float(os.environ.get("LEV", "1"))
                cap_by_cash = int((self.capital * lev) // close_px)
                cap_by_invested = int(max(0.0, (self.max_invested * lev - current_invested)) // close_px)
                shares = max(0, min(shares, cap_by_cash, cap_by_invested))
                cost = shares * close_px
                if shares < 1: continue
                if cost > self.capital: continue
                if current_invested + cost > self.max_invested * lev: continue

                entry_price = close_px * (1 + self.slippage)
                self.capital -= cost + self.fee * cost
                self.portfolio[sym] = {
                    "entry_date": date,
                    "entry": entry_price,
                    "shares": shares,
                    "stop": stop,
                    "target": target,
                    "atr": atr_val,
                    "trail_active": False,
                    "trail_stop": stop,
                }
                self._orders_filled += 1

        # --- force exit remaining ---
        if self.dates:
            last_date = self.dates[-1]
            for sym, pos in list(self.portfolio.items()):
                df_sym = self.data.get(sym)
                if df_sym is not None:
                    p_last = _pos_at(df_sym, last_date)
                    price = float(df_sym.iloc[p_last]["Close"]) if p_last is not None else float(df_sym["Close"].iloc[-1])
                else:
                    price = float(pos["entry"])
                exit_price = price * (1 - self.slippage)
                shares = pos["shares"]
                proceeds = exit_price * shares
                entry_cost = pos["entry"] * shares
                self.capital += proceeds - self.fee * proceeds - self.fee * entry_cost
                self.trades.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry"],
                    "exit_date": last_date,
                    "exit_price": exit_price,
                    "shares": shares,
                    "reason": "forced",
                })
                del self.portfolio[sym]
            self.equity_curve.append({"date": last_date, "value": float(self.capital)})

        # --- save outputs ---
        pd.DataFrame(self.trades).to_csv("backtest_trades.csv", index=False)
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv("equity_curve.csv", index=False)
        if not equity_df.empty:
            cagr_val = cagr(equity_df["date"], equity_df["value"])
            mdd = max_drawdown(equity_df["value"])
            print(f"CAGR: {cagr_val:.2%}, Max Drawdown: {mdd:.2%}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest engine for Breakout Momentum v3.0")
    p.add_argument("--config", type=str, default="backtest/config.yaml", help="Path to config YAML")
    p.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    p.add_argument("--universe", type=str, help="Path to CSV with symbols")
    p.add_argument("--max-pos", type=int, help="Override MAX_POSITIONS")
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    engine = BacktestEngine(
        config_path=args.config,
        start_date=start,
        end_date=end,
        universe_csv=args.universe,
        max_positions_override=args.max_pos,
    )
    engine.run()


if __name__ == "__main__":
    main()
