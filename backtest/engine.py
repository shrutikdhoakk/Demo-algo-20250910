"""Simple backtest engine for the Breakout Momentum strategy (Turbo, Pylance-safe)."""
from __future__ import annotations

# --- Safe guards for .iloc on optional objects ---
from typing import Optional, cast
import pandas as pd

def require_dataframe(x: Optional[object], name: str) -> pd.DataFrame:
    if x is None:
        raise ValueError(f"{name} is None")
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(x)}")
    if x.empty:
        raise ValueError(f"{name} is empty")
    return cast(pd.DataFrame, x)

def require_series(x: Optional[object], name: str) -> pd.Series:
    if x is None:
        raise ValueError(f"{name} is None")
    if not isinstance(x, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(x)}")
    if x.empty:
        raise ValueError(f"{name} is empty")
    return cast(pd.Series, x)

def iloc_last_df(x: Optional[pd.DataFrame], name: str) -> pd.Series:
    return require_dataframe(x, name).iloc[-1]

def iloc_last_series(x: Optional[pd.Series], name: str):
    return require_series(x, name).iloc[-1]


# --- pattern gate hook (auto-injected) ---
import os
try:
    from .pattern_gate import pattern_gate
except Exception:
    pattern_gate = None

USE_PATTERN_GATE = os.environ.get("USE_PATTERN_GATE", "0") == "1"

def _first_df_for_gate(ns):
    try:
        import pandas as pd
    except Exception:
        return None
    for k in ("df_sym","df","ohlc","bars","data","frame","window"):
        v = ns.get(k)
        try:
            if isinstance(v, pd.DataFrame):
                cols = {c.lower() for c in v.columns}
                if {"open","high","low","close"}.issubset(cols):
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
from typing import Dict, Any, List, Optional, cast
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from dataclasses import fields, MISSING
from types import SimpleNamespace as _SNS

from src.strategy.breakout_momentum_v3 import StrategyConfig, BreakoutMomentumV3
try:
    from src.strategy.breakout_momentum_v7 import StrategyConfigV7, BreakoutMomentumV7  # type: ignore
except Exception:
    StrategyConfigV7 = None  # type: ignore
    BreakoutMomentumV7 = None  # type: ignore

from src.data.ingest import load_csv
from .metrics import cagr, max_drawdown

from backtest.alpha_layers import build_momentum_mask

# --- helpers: selection, dtype, RSI, etc. ---
_NUMERIC_COL_CANDIDATES = [
    "Open","High","Low","Close","Adj Close","Volume",
    "open","high","low","close","adj close","volume",
    "ATR14","RSI14","VOL_SMA_N","HH_N_PREV","RS_PCT"
]

def _coerce_numeric_cols(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    cols = cols or _NUMERIC_COL_CANDIDATES
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df

def _pick_numeric(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").astype(float)

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").astype(float)
    delta = close.diff()
    gain = pd.Series(np.where(delta.gt(0), delta, 0.0), index=close.index)
    loss = pd.Series(np.where(delta.lt(0), -delta, 0.0), index=close.index)
    up   = gain.ewm(alpha=1/period, adjust=False).mean()
    dn   = loss.ewm(alpha=1/period, adjust=False).mean()
    rs   = up / dn.replace(0, np.nan)
    out  = 100 - (100 / (1 + rs))
    return out.bfill().fillna(50.0)

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
    if idx <= 0 or idx >= len(df):
        return False

    close = _pick_numeric(df, ["Close","close"])
    high  = _pick_numeric(df, ["High","high"])

    hh_prev_opt = df.get("HH_N_PREV")
    if not isinstance(hh_prev_opt, pd.Series):
        hh_prev = high.rolling(high_n, min_periods=high_n).max().shift(1)
    else:
        hh_prev = hh_prev_opt
        if pd.isna(hh_prev.iloc[idx]):
            hh_prev = high.rolling(high_n, min_periods=high_n).max().shift(1)

    hv = float(hh_prev.iloc[idx]) if pd.notna(hh_prev.iloc[idx]) else np.nan
    breakout = bool(pd.notna(hv) and (float(close.iloc[idx]) > hv))

    rsi_col_opt = df.get("RSI14")
    if not isinstance(rsi_col_opt, pd.Series) or pd.isna(rsi_col_opt.iloc[idx]):
        rsi_col = _rsi(close, rsi_len).bfill().fillna(50.0)
    else:
        rsi_col = pd.to_numeric(rsi_col_opt, errors="coerce").astype(float)
    rsi_ok = bool(float(rsi_col.iloc[idx]) >= float(rsi_thr))

    vol_ok = False
    vol_sma_opt = df.get("VOL_SMA_N")
    if isinstance(vol_sma_opt, pd.Series):
        vol_sma = pd.to_numeric(vol_sma_opt, errors="coerce").astype(float)
        if "Volume" in df.columns:
            vol = pd.to_numeric(df["Volume"], errors="coerce").astype(float)
        elif "volume" in df.columns:
            vol = pd.to_numeric(df["volume"], errors="coerce").astype(float)
        else:
            vol = None
        if vol is not None and pd.notna(vol_sma.iloc[idx]):
            vol_ok = bool(float(vol.iloc[idx]) >= float(vol_mult) * float(vol_sma.iloc[idx]))
        else:
            vol_ok = (not require_volume)
    else:
        vol_ok = (not require_volume)

    return bool(breakout and rsi_ok and vol_ok)
# --- end helpers ---

def _parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s: return None
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

def _pos_at(df: Optional[pd.DataFrame], date: pd.Timestamp) -> Optional[int]:
    if df is None or df.empty:
        return None
    idxer = df.index.get_indexer([date])
    if len(idxer) == 0:
        return None
    idx = int(idxer[0])
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
            scfg = self.cfg.get("strategycfg")
            if isinstance(scfg, dict):
                for _k, _v in scfg.items():
                    self.cfg.setdefault(_k, _v)
            print("USING_PARAMS","scope=engine_flat",
                  f"breakout_atr_buf={self.cfg.get('breakout_atr_buf')}",
                  f"trail_atr_mult={self.cfg.get('trail_atr_mult')}",
                  f"atr_pct_max={self.cfg.get('atr_pct_max')}",
                  sep=" | ")
        try:
            import json, hashlib
            _hash = hashlib.sha1(json.dumps(self.cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
        except Exception:
            _hash = "nohash"
        print("USING_PARAMS","scope=engine",f"hash={_hash}",
              f"breakout_atr_buf={self.cfg.get('breakout_atr_buf')}",
              f"trail_atr_mult={self.cfg.get('trail_atr_mult')}",
              f"atr_pct_max={self.cfg.get('atr_pct_max')}",
              sep=" | ")

        # risk/settings (defaults for 1L+ profile; override via YAML/ENV)
        self.total_capital: float  = float(self.cfg.get("TOTAL_CAPITAL", 100000.0))
        self.max_invested: float   = float(self.cfg.get("MAX_INVESTED", 100000.0))
        self.per_trade_risk: float = float(self.cfg.get("PER_TRADE_RISK", 2000.0))
        self.slippage: float       = float(self.cfg.get("SLIPPAGE", 0.001))
        self.fee: float            = float(self.cfg.get("FEE", 0.0003))

        default_max_pos = int(self.cfg.get("MAX_POSITIONS", 5))
        self.max_positions: int = int(max_positions_override or default_max_pos)

        if universe_csv:
            self.symbols: List[str] = _read_universe_csv(universe_csv)
        else:
            self.symbols = list(self.cfg.get("SYMBOLS", []))

        # choose strategy
        strategy_name = (self.cfg.get("STRATEGY") or self.cfg.get("STRATEGY_VERSION") or "v3").lower()
        if strategy_name == "v7" and StrategyConfigV7 is not None:
            dataclass_type = StrategyConfigV7
        else:
            dataclass_type = StrategyConfig

        base: Dict[str, Any] = {}
        try:
            for f in fields(dataclass_type):
                if f.default is not MISSING:
                    base[f.name] = f.default
                else:
                    dfac = getattr(f, "default_factory", MISSING)
                    if dfac is not MISSING:
                        try: base[f.name] = dfac()
                        except Exception: pass
        except Exception as e:
            print("WARN | could not read StrategyConfig defaults:", e)

        for k, v in self.cfg.items():
            if k in base: base[k] = v

        sc_view = _SNS(**base)
        if strategy_name == "v7" and BreakoutMomentumV7 is not None:
            print("USING_PARAMS","scope=engine->strategycfg(v7)",
                  " | ".join(f"{k}={getattr(sc_view,k,None)}" for k in base.keys()), sep=" | ")
            self.strategy = BreakoutMomentumV7(sc_view)
        else:
            for k in ("breakout_atr_buf","trail_atr_mult","atr_pct_max"):
                if k in self.cfg and self.cfg[k] is not None:
                    try: setattr(sc_view,k,float(self.cfg[k]))
                    except Exception: setattr(sc_view,k,self.cfg[k])
            print("USING_PARAMS","scope=engine->strategycfg(v3)",
                  f"breakout_atr_buf={getattr(sc_view,'breakout_atr_buf',None)}",
                  f"trail_atr_mult={getattr(sc_view,'trail_atr_mult',None)}",
                  f"atr_pct_max={getattr(sc_view,'atr_pct_max',None)}", sep=" | ")
            self.strategy = BreakoutMomentumV3(sc_view)

        # --- data prep (symbols) ---
        self.data: Dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            df = load_csv(sym)
            if start_date is not None: df = df[df.index >= start_date]
            if end_date   is not None: df = df[df.index <= end_date]
            if df.empty: continue

            df = _coerce_numeric_cols(df)
            df = self.strategy.compute_features(df)
            df = _coerce_numeric_cols(df)
            df = _normalize_index(df)

            close = _pick_numeric(df, ["Close","close"])
            high  = _pick_numeric(df, ["High","high"])

            df["RSI14"] = _rsi(close, int(os.environ.get("RSI_LEN","14"))).bfill().fillna(50.0)
            high_n = int(os.environ.get("HIGH_N","20"))
            df["HH_N_PREV"] = high.rolling(high_n, min_periods=high_n).max().shift(1)

            if "Volume" in df.columns:
                vol = pd.to_numeric(df["Volume"], errors="coerce").astype(float)
            elif "volume" in df.columns:
                vol = pd.to_numeric(df["volume"], errors="coerce").astype(float)
            else:
                vol = None
            if vol is not None:
                vol_len = int(os.environ.get("VOL_LEN","20"))
                df["VOL_SMA_N"] = vol.rolling(vol_len, min_periods=max(1, vol_len//2)).mean()

            self.data[sym] = df

        # union of dates
        all_dates = set()
        for df in self.data.values(): all_dates.update(df.index)
        dates_sorted = sorted(all_dates)
        if start_date is not None: dates_sorted = [d for d in dates_sorted if d >= start_date]
        if end_date   is not None: dates_sorted = [d for d in dates_sorted if d <= end_date]
        self.dates = dates_sorted

        # --- Relative Strength vs index (synthetic fallback) ---
        self.index_symbol = str(self.cfg.get("INDEX_SYMBOL") or os.environ.get("INDEX_SYMBOL") or "^NIFTY500")
        self.rs_top_pct   = float(os.environ.get("RS_TOP_PCT", self.cfg.get("RS_TOP_PCT", 0.60)))
        self.rs_lookback  = int(os.environ.get("RS_LOOKBACK_D", "63"))

        self.idx_close: Optional[pd.Series] = None
        self._synthetic_index = False
        try:
            idx_df = load_csv(self.index_symbol)
            if start_date is not None: idx_df = idx_df[idx_df.index >= start_date]
            if end_date   is not None: idx_df = idx_df[idx_df.index <= end_date]
            idx_df = _normalize_index(idx_df)
            idx_col = idx_df["Close"] if "Close" in idx_df.columns else idx_df.iloc[:,0]
            self.idx_close = pd.to_numeric(idx_col, errors="coerce").astype(float).reindex(self.dates)
            if self.idx_close.isna().all(): self.idx_close = None
        except Exception:
            self.idx_close = None

        close_panel = None
        if self.idx_close is None and len(self.data) > 0:
            close_panel = {}
            for sym, df in self.data.items():
                s = _pick_numeric(df, ["Close","close"]).reindex(self.dates)
                close_panel[sym] = s
            close_panel = pd.DataFrame(close_panel, index=self.dates)
            self.idx_close = close_panel.median(axis=1, skipna=True)
            self._synthetic_index = True
            print(f"RS: using synthetic index (median of {close_panel.shape[1]} symbols)")

        self._RS_FILTER_ON = self.idx_close is not None
        if self._RS_FILTER_ON:
            if close_panel is None:
                close_panel = {}
                for sym, df in self.data.items():
                    s = _pick_numeric(df, ["Close","close"]).reindex(self.dates)
                    close_panel[sym] = s
                close_panel = pd.DataFrame(close_panel, index=self.dates)

            L = max(1, self.rs_lookback)
            assert self.idx_close is not None  # type narrowing for pylance
            idx_close = cast(pd.Series, self.idx_close)
            idx_ret = (idx_close / idx_close.shift(L) - 1.0)
            sym_ret = (close_panel / close_panel.shift(L) - 1.0)
            rs_excess = sym_ret.sub(idx_ret, axis=0)
            rs_smoothed = rs_excess.rolling(5, min_periods=3).mean()
            rs_pct = rs_smoothed.rank(axis=1, pct=True, method="min")
            for sym, df in self.data.items():
                df["RS_PCT"] = rs_pct.get(sym, pd.Series(index=self.dates, dtype=float)).reindex(df.index)
            idx_name = "synthetic" if self._synthetic_index else self.index_symbol
            print(f"RS: filter ON | threshold={self.rs_top_pct:.2f} | lookback={L}d | index={idx_name}")

        # --- Market regime filter (index 200-DMA slope > 0) ---
        self.regime_on = os.environ.get("REGIME_ON", str(self.cfg.get("REGIME_ON", 1))) == "1"
        self.regime_len = int(os.environ.get("REGIME_LEN", self.cfg.get("REGIME_LEN", 200)))
        self.regime_slope_win = int(os.environ.get("REGIME_SLOPE_WIN", self.cfg.get("REGIME_SLOPE_WIN", 20)))
        if self.idx_close is not None and self.regime_on:
            idx_ma = self.idx_close.rolling(self.regime_len, min_periods=self.regime_len).mean()
            slope = idx_ma - idx_ma.shift(self.regime_slope_win)
            self.regime_bull = (slope > 0) & (self.idx_close > idx_ma)
            self.regime_bull = self.regime_bull.fillna(False)
            print(f"REGIME: ON | len={self.regime_len} | slope_win={self.regime_slope_win}")
        else:
            self.regime_bull = pd.Series(True, index=self.dates)
            print("REGIME: OFF (no index or disabled)")

        # --- Momentum overlay ---
        self.mom_on = os.environ.get("ENH_MOM_ON", "1") == "1"
        if self.mom_on:
            _pp: Dict[str, pd.DataFrame] = {}
            for _sym, _df in self.data.items():
                _c = _pick_numeric(_df, ["Close","close"]).reindex(self.dates)
                _pp[_sym] = pd.DataFrame({"Close": _c})
            self.mom_mask: Dict[str, pd.Series] = build_momentum_mask(_pp)
            print("MOMENTUM: ON", f"pctile={os.environ.get('MOM_PCTILE','0.70')}", f"lookbacks={os.environ.get('MOM_LOOKBACKS_D','126,63,21')}", sep=" | ")
        else:
            self.mom_mask = {s: pd.Series(True, index=self.dates) for s in self.data}
            print("MOMENTUM: OFF")

        # --- state ---
        self.capital: float = self.total_capital
        self.portfolio: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self._cand_total = 0
        self._cand_dates = 0
        self._orders_attempted = 0
        self._orders_filled = 0

        # tunables (ENV overrides)
        self._RSI_LEN  = int(os.environ.get("RSI_LEN", "14"))
        self._RSI_THR  = float(os.environ.get("RSI_THR", "55"))
        self._VOL_LEN  = int(os.environ.get("VOL_LEN", "20"))
        self._VOL_MULT = float(os.environ.get("VOL_MULT", "1.5"))
        self._HIGH_N   = int(os.environ.get("HIGH_N", "20"))
        self._REQ_VOL  = os.environ.get("REQUIRE_VOLUME", "1") == "1"

        # multi-stage dynamic trail
        self._TR1_AFTER_R = float(os.environ.get("TR1_AFTER_R","1.0"))
        self._TR1_MULT    = float(os.environ.get("TR1_MULT","0.8"))
        self._TR2_AFTER_R = float(os.environ.get("TR2_AFTER_R","2.0"))
        self._TR2_MULT    = float(os.environ.get("TR2_MULT","0.6"))

        # pyramiding: levels & fracs (comma-separated)
        self._PYR_ON   = os.environ.get("PYR_ON","1") == "1"
        def _parse_floats(key: str, default: str) -> List[float]:
            raw = os.environ.get(key, default)
            return [float(x) for x in raw.split(",") if x.strip()]
        self._PYR_LEVELS = _parse_floats("PYR_LEVELS", "0.5,1.0")
        self._PYR_FRACS  = _parse_floats("PYR_FRACS",  "0.5,0.33")
        self._PYR_MAX_ADDS = int(os.environ.get("PYR_MAX_ADDS","2"))

        # optional fixed take-profit (default OFF for trend riding)
        self._TP_ON = os.environ.get("TP_ON","0") == "1"

    def run(self) -> None:
        for date in self.dates:
            # exits & pyramids
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

                # --- multi-stage dynamic trail ---
                base_mult = float(getattr(self.strategy.cfg, "trail_atr_mult", getattr(self.strategy.cfg, "chandelier_mult", 1.3)))
                R_unit_tr = max(1e-12, float(pos["atr"]) * base_mult)
                open_r_tr = (price - float(pos["entry"])) / R_unit_tr
                eff_mult = base_mult
                if open_r_tr >= self._TR1_AFTER_R:
                    eff_mult = base_mult * self._TR1_MULT
                if open_r_tr >= self._TR2_AFTER_R:
                    eff_mult = base_mult * self._TR2_MULT

                if pos.get("trail_active", False):
                    trail_stop = price - float(pos["atr"]) * eff_mult
                    if trail_stop > pos["trail_stop"]:
                        pos["trail_stop"] = float(trail_stop)
                else:
                    trig = (price - float(pos["entry"])) >= float(self.strategy.cfg.trail_trigger_atr) * float(pos["atr"])
                    if _apply_pattern_gate(_first_df_for_gate(locals()), bool(trig)) and trig:
                        pos["trail_active"] = True
                        pos["trail_stop"] = float(price - float(pos["atr"]) * eff_mult)

                # --- pyramiding (multi-level) ---
                if self._PYR_ON:
                    adds_done = int(pos.get("adds",0))
                    pyr_idx = int(pos.get("pyr_idx",0))
                    if adds_done < self._PYR_MAX_ADDS and pyr_idx < len(self._PYR_LEVELS):
                        R_unit = max(1e-12, float(self.strategy.cfg.atr_stop_mult) * float(pos["atr"]))
                        open_r = (price - float(pos["entry"])) / R_unit
                        lev = float(os.environ.get("LEV","1"))
                        init_sh = int(pos.get("init_shares", pos["shares"]))
                        # allow chaining multiple thresholds hit on same day
                        while (adds_done < self._PYR_MAX_ADDS and pyr_idx < len(self._PYR_LEVELS) and open_r >= float(self._PYR_LEVELS[pyr_idx])):
                            frac = float(self._PYR_FRACS[min(pyr_idx, len(self._PYR_FRACS)-1)])
                            add_sh_plan = max(0, int(init_sh * frac))
                            if add_sh_plan < 1:
                                pyr_idx += 1
                                continue
                            # capacity checks
                            current_invested = 0.0
                            for s2, p2 in self.portfolio.items():
                                df2 = self.data.get(s2)
                                j = _pos_at(df2, date)
                                if j is None: continue
                                current_invested += float(df2.iloc[j]["Close"]) * int(p2["shares"])
                            cap_by_cash = int((self.capital * lev) // price)
                            cap_by_invested = int(max(0.0, (self.max_invested * lev - current_invested)) // price)
                            add_sh = max(0, min(add_sh_plan, cap_by_cash, cap_by_invested))
                            if add_sh >= 1:
                                add_price = price * (1 + self.slippage)
                                add_cost  = add_price * add_sh
                                if add_cost <= self.capital * lev and current_invested + add_cost <= self.max_invested * lev:
                                    self.capital -= add_cost + self.fee * add_cost
                                    pos["shares"] = int(pos["shares"]) + int(add_sh)
                                    pos["entry_cost"] = float(pos.get("entry_cost", float(pos["entry"]) * init_sh)) + float(add_price) * int(add_sh)
                                    adds_done += 1
                                    pyr_idx  += 1
                                else:
                                    break
                            else:
                                break
                        pos["adds"] = adds_done
                        pos["pyr_idx"] = pyr_idx

                # --- exits ---
                if price <= pos["stop"]:
                    exit_price = price * (1 - self.slippage); exit_reason = "stop"
                elif self._TP_ON and price >= pos["target"]:
                    exit_price = price * (1 - self.slippage); exit_reason = "target"
                elif pos.get("trail_active", False) and price <= pos["trail_stop"]:
                    exit_price = price * (1 - self.slippage); exit_reason = "trail"
                elif self.strategy.should_early_exit(df_sym, i):
                    exit_price = price * (1 - self.slippage); exit_reason = "early"

                if exit_price is not None:
                    shares = int(pos["shares"])
                    proceeds = float(exit_price) * shares
                    entry_cost = float(pos.get("entry_cost", float(pos["entry"]) * shares))
                    self.capital += proceeds - self.fee * proceeds - self.fee * entry_cost
                    self.trades.append({
                        "symbol": sym,
                        "entry_date": pos["entry_date"],
                        "entry_price": float(pos["entry"]),
                        "exit_date": date,
                        "exit_price": float(exit_price),
                        "shares": shares,
                        "reason": exit_reason,
                    })
                    del self.portfolio[sym]

            # mark to market
            invested_value = 0.0
            for sym, pos in self.portfolio.items():
                df_sym = self.data.get(sym)
                if df_sym is None: continue
                j = _pos_at(df_sym, date)
                if j is None: continue
                invested_value += float(df_sym.iloc[j]["Close"]) * int(pos["shares"])
            total_value = float(self.capital + invested_value)
            self.equity_curve.append({"date": date, "value": total_value})

            # entries
            if not bool(self.regime_bull.loc[date]):
                continue

            candidates: List[tuple[float, str, pd.Series]] = []
            for sym, df_sym in self.data.items():
                if sym in self.portfolio: continue
                k = _pos_at(df_sym, date)
                if k is None: continue
                row = df_sym.iloc[k]
                if k == 0 or bool(row.get("_WARMUP", False)): continue
                prev = df_sym.iloc[k - 1]

                # Gate 1: Breakout + RSI + Volume
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

                # Gate 2: RS vs index
                rs_pct = row.get("RS_PCT", np.nan)
                if not (pd.notna(rs_pct) and float(rs_pct) >= float(self.rs_top_pct)):
                    continue

                # Gate 3: Momentum overlay
                m_series = self.mom_mask.get(sym)
                try:
                    mom_ok = True if m_series is None else bool(m_series.loc[date])
                except KeyError:
                    mom_ok = True
                if not mom_ok:
                    continue

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
                stop   = close_px - float(self.strategy.cfg.atr_stop_mult)   * atr_val
                target = close_px + float(self.strategy.cfg.atr_target_mult) * atr_val

                # trend / 52w-high proximity
                k = _pos_at(df_sym, date)
                if k is None or k < 252: continue
                s_close = _pick_numeric(df_sym, ["Close","close"])
                sma200_val = float(s_close.rolling(200, min_periods=200).mean().iloc[k])
                hh252_val  = float(s_close.rolling(252, min_periods=252).max().iloc[k])
                px         = float(s_close.iloc[k])
                near_hi_pct = float(os.environ.get("RS_WITHIN_HI","0.05"))
                if (px < sma200_val) or (px < hh252_val * (1 - near_hi_pct)):
                    continue

                risk_per_share = close_px - float(stop)
                if risk_per_share <= 0: continue

                lev = float(os.environ.get("LEV","1"))
                shares = int((self.per_trade_risk * float(os.environ.get("RISK_BOOST","1"))) / risk_per_share)
                if shares <= 0: continue

                cost = shares * close_px
                current_invested = 0.0
                for sym2, pos2 in self.portfolio.items():
                    df2 = self.data.get(sym2)
                    p2 = _pos_at(df2, date)
                    if p2 is None or df2 is None: continue
                    current_invested += float(df2.iloc[p2]["Close"]) * int(pos2["shares"])

                cap_by_cash = int((self.capital * lev) // close_px)
                cap_by_invested = int(max(0.0, (self.max_invested * lev - current_invested)) // close_px)
                shares = max(0, min(shares, cap_by_cash, cap_by_invested))
                cost = shares * close_px
                if shares < 1: continue
                if cost > self.capital * lev: continue
                if current_invested + cost > self.max_invested * lev: continue

                entry_price = close_px * (1 + self.slippage)
                self.capital -= cost + self.fee * cost
                self.portfolio[sym] = {
                    "entry_date": date,
                    "entry": float(entry_price),
                    "shares": int(shares),
                    "init_shares": int(shares),
                    "entry_cost": float(entry_price) * int(shares),
                    "stop": float(stop),
                    "target": float(target),
                    "atr": float(atr_val),
                    "trail_active": False,
                    "trail_stop": float(stop),
                    "adds": 0,
                    "pyr_idx": 0,
                }
                self._orders_filled += 1

        # force exit remaining
        if self.dates:
            last_date = self.dates[-1]
            for sym, pos in list(self.portfolio.items()):
                df_sym = self.data.get(sym)
                if df_sym is not None:
                    p_last = _pos_at(df_sym, last_date)
                    if p_last is not None:
                        price = float(df_sym.iloc[p_last]["Close"])
                    else:
                        price = float(df_sym["Close"].iloc[-1])
                else:
                    price = float(pos["entry"])
                exit_price = price * (1 - self.slippage)
                shares = int(pos["shares"])
                proceeds = float(exit_price) * shares
                entry_cost = float(pos.get("entry_cost", float(pos["entry"]) * shares))
                self.capital += proceeds - self.fee * proceeds - self.fee * entry_cost
                self.trades.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "entry_price": float(pos["entry"]),
                    "exit_date": last_date,
                    "exit_price": float(exit_price),
                    "shares": shares,
                    "reason": "forced",
                })
                del self.portfolio[sym]
            self.equity_curve.append({"date": last_date, "value": float(self.capital)})

        # save outputs
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
