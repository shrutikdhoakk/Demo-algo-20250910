"""Breakout + Momentum Swing v3.0 strategy implementation.

This module defines a dataclass for configuration and a concrete
strategy class that computes required features, evaluates entry
conditions, assigns a composite score and determines early exit
conditions.  The strategy is designed to be used with the backtest
engine provided in ``backtest/engine.py``.
"""

from __future__ import annotations
from typing import Any

# --- runtime YAML override merge (strategy knobs) ---
def _merge_yaml_strategy_overrides(cfg):
    try:
        from pathlib import Path
        ypath = Path("backtest/config.yaml")
        if not ypath.exists():
            return
        y = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
        # accept both: top-level keys or nested under STRATEGY:
        src = y.get("STRATEGY") or y
        allow = {
            "breakout_atr_buf","atr_pct_max","gap_guard_atr","ext_cap_atr",
            "trail_lookback","trail_atr_mult","atr_pct_spike",
            "shock_atr_mult","gap_atr_mult"
        }
        for k in allow:
            if k in src:
                setattr(cfg, k, src[k])
    except Exception:
        # be silent in backtest runs
        pass


from dataclasses import dataclass
from typing import Tuple
from typing import Any

import numpy as np
import yaml
import pandas as pd
from typing import Any

from ..features.indicators import rsi, atr, adx, hhv, sma, bb_width, zscore
from ..patterns.candlesticks import bullish_on_bar
from ..patterns.chart_patterns import (
    bullish_pattern_confirm,
    pattern_score,
    bearish_reversal_detect,
)


@dataclass(frozen=True)
@dataclass
class StrategyConfig:
    """Parameter container for the breakout momentum strategy."""
    breakout_atr_buf: float = 0.45
    trail_atr_mult: float = 1.30
    atr_pct_max: float = 0.055
    adx_min: float = 25.0
    rsi_min: float = 55.0
    vol_mult_min: float = 2.0
    vol_z_min: float = 1.5
    beta_min: float = 0.9
    beta_max: float = 1.5
    delta_min: float = 0.7
    close_top_pct: float = 0.30
    hh_lookbacks: Tuple[int, int] = (20, 50)
    bb_width_percentile: int = 5
    atr_compression_window: int = 20
    squeeze_window: int = 20
    atr_stop_mult: float = 1.5
    atr_target_mult: float = 2.0
    trail_trigger_atr: float = 1.5
    trail_method: str = 'chandelier'
    chandelier_mult: float = 3.0
    early_sma20_below_days: int = 2
    early_rsi14_below: float = 35.0
    early_rsi14_bars: int = 2


class BreakoutMomentumV3:
    """Encapsulates the logic for the Breakout + Momentum Swing v3.0 strategy."""

    def __init__(self, cfg: Any):
        self.cfg = cfg

    # ---- internal helper -------------------------------------------------
    def _avg_rsi_nan_safe(self, row: pd.Series) -> float:
        """
        Return average of RSI7/14/21 as float. If all are NaN, return NaN.
        """
        raw_vals = [row.get("RSI7"), row.get("RSI14"), row.get("RSI21")]
        arr = pd.to_numeric(pd.Series(raw_vals, dtype="float64"), errors="coerce").to_numpy()
        if np.isnan(arr).all():
            return np.nan
        return float(np.nanmean(arr))

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        try:
            _merge_yaml_strategy_overrides(self.cfg)
        except Exception:
            pass

        # --- SMAs ---
        df['SMA20']  = df['Close'].rolling(20,  min_periods=1).mean()
        df['SMA50']  = df['Close'].rolling(50,  min_periods=1).mean()
        df['SMA200'] = df['Close'].rolling(200, min_periods=1).mean()

        # --- Prior highs for breakout checks (shifted to use previous bars) ---
        df['HH20'] = df['High'].rolling(20, min_periods=1).max().shift(1)
        df['HH50'] = df['High'].rolling(50, min_periods=1).max().shift(1)

        # --- ATR14 (simple mean TR) ---
        tr1 = (df['High'] - df['Low']).abs()
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low']  - df['Close'].shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14, min_periods=1).mean()

        # --- Volume stats ---
        df['VolAvg20'] = df['Volume'].rolling(20, min_periods=1).mean()
        _vol_std = df['Volume'].rolling(20, min_periods=1).std(ddof=0)
        df['VolZ20']   = (df['Volume'] - df['VolAvg20']) / _vol_std.replace({0: np.nan})

        # --- ADX14 (lightweight) ---
        up = df['High'].diff()
        dn = -df['Low'].diff()
        plus_dm  = up.where((up > dn) & (up > 0), 0.0)
        minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
        atr_s = tr.rolling(14, min_periods=1).mean().replace(0, np.nan)
        plus_di  = 100 * (plus_dm.rolling(14, min_periods=1).mean()  / atr_s)
        minus_di = 100 * (minus_dm.rolling(14, min_periods=1).mean() / atr_s)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        df['ADX14'] = dx.rolling(14, min_periods=1).mean().fillna(0)

        # --- RSIs (7/14/21) ---
        for p in (7, 14, 21):
            delta = df['Close'].diff()
            gain  = delta.clip(lower=0).rolling(p, min_periods=1).mean()
            loss  = (-delta.clip(upper=0)).rolling(p, min_periods=1).mean()
            rs    = gain / loss.replace(0, np.nan)
            df[f'RSI{p}'] = 100 - (100 / (1 + rs))

        # --- Candles / patterns / scores ---
        rng   = (df['High'] - df['Low']).abs().replace(0, np.nan)
        bodyU = (df['Close'] - df['Open']).clip(lower=0)
        bodyD = (df['Open']  - df['Close']).clip(lower=0)
        df['BullCandle']       = (df['Close'] > df['Open'])
        df['BearCandle']       = (df['Close'] < df['Open'])
        df['PatternConfirmed'] = df['BullCandle'] & ((df['Close'] - df['Open']).abs() >= 0.6 * rng)
        df['BearishPattern']   = df['BearCandle'] & ((df['Open'] - df['Close']).abs() >= 0.6 * rng)
        df['PatternScore']     = np.where(df['BullCandle'], 100.0 * bodyU / rng, 0.0)
        df['BearPatternScore'] = np.where(df['BearCandle'], 100.0 * bodyD / rng, 0.0)
        df['PatternScore']     = df['PatternScore'].fillna(0.0).clip(0, 100)
        df['BearPatternScore'] = df['BearPatternScore'].fillna(0.0).clip(0, 100)

        # --- Risk metrics / trailing / events ---
        df['ATRPCT14']   = (df['ATR14'] / df['Close']).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df['RollMaxHigh']  = df['High'].rolling(int(getattr(self.cfg, 'trail_lookback', 8)), min_periods=1).max()
        df['RollMaxClose'] = df['Close'].rolling(int(getattr(self.cfg, 'trail_lookback', 8)), min_periods=1).max()
        df['TrailLine']    = df['RollMaxHigh'] - float(getattr(self.cfg, 'trail_atr_mult', 1.3)) * df['ATR14']

        prev_close = df['Close'].shift(1); prev_low = df['Low'].shift(1)
        df['DownShock'] = (prev_close - df['Close']) > (float(getattr(self.cfg, 'shock_atr_mult', 1.0)) * df['ATR14'])
        df['GapDown']   = (df['Open'] < prev_low) | (df['Open'] < (prev_close - float(getattr(self.cfg,'gap_atr_mult',0.75)) * df['ATR14']))

        # NEW: over-extension vs SMA20 measured in ATRs, and gap-up size in ATRs
        df['ExtOver20'] = ((df['Close'] - df['SMA20']) / df['ATR14']).replace([np.inf,-np.inf], np.nan)
        df['GapUpATR']  = ((df['Open'] - prev_close) / df['ATR14']).replace([np.inf,-np.inf], np.nan)

        return df

    def is_entry(self, row: pd.Series, prev: pd.Series) -> bool:
        cfg = self.cfg
        s20  = row.get("SMA20",  float("nan"))
        s50  = row.get("SMA50",  float("nan"))
        s200 = row.get("SMA200", float("nan"))
        trend_ok = (row.get("Close", float("nan")) > s50) and (s50 > s200) and (s20 > s50)

        hh20_prev = prev.get("HH20", row.get("HH20", float("nan")))
        hh50_prev = prev.get("HH50", row.get("HH50", float("nan")))
        hh_prev   = max(hh20_prev, hh50_prev)
        atr14     = row.get("ATR14", float("nan"))
        buf_mult  = float(getattr(cfg, "breakout_atr_buf", 0.45))  # tighter default 0.45 ATR
        brk = (atr14 == atr14) and (hh_prev == hh_prev) and (row["Close"] > (hh_prev + buf_mult * atr14))

        day_rng = (row.get("High", float("nan")) - row.get("Low", float("nan")))
        close_top_ok = (day_rng == day_rng) and (day_rng > 0) and ((row["High"] - row["Close"]) <= day_rng * 0.20)

        # Anti-gap chase & overextension: guard against vertical moves
        gap_guard = float(getattr(cfg, "gap_guard_atr", 0.75))
        ext_cap   = float(getattr(cfg, "ext_cap_atr", 2.5))
        gap_ok    = (row.get("GapUpATR", 0) <= gap_guard) if (row.get("GapUpATR", float("nan")) == row.get("GapUpATR", float("nan"))) else True
        ext_ok    = (row.get("ExtOver20", 0) <= ext_cap)  if (row.get("ExtOver20", float("nan")) == row.get("ExtOver20", float("nan"))) else True

        # Volume
        vol_ok = False
        base = row.get("VolAvg20", float("nan"))
        if base == base and base != 0:
            v1 = (row.get("Volume", 0) >= 1.5 * base)
            v2 = (row.get("Volume", 0) >= cfg.vol_mult_min * base) and (row.get("VolZ20", -1e9) >= cfg.vol_z_min)
            vol_ok = bool(v1 or v2)

        # Momentum: ADX + any 2 of 3 RSIs
        rsis = [row.get("RSI7", 0), row.get("RSI14", 0), row.get("RSI21", 0)]
        mom_ok = (row.get("ADX14", 0) > cfg.adx_min) and (sum(v > cfg.rsi_min for v in rsis) >= 2)

        # Pattern
        patt_ok = bool(row.get("PatternConfirmed", False) or (row.get("PatternScore", 0) >= 60))

        # Volatility guard (tighter default 5.5%)
        atr_pct_max = float(getattr(cfg, "atr_pct_max", 0.055))
        atr_pct_ok  = row.get("ATRPCT14", 1.0) <= atr_pct_max

        # Optional beta/delta gates
        beta_ok = True
        if "Beta1Y" in row.index and row["Beta1Y"] == row["Beta1Y"]:
            beta_ok = cfg.beta_min <= row["Beta1Y"] <= cfg.beta_max
        delta_ok = True
        if "Delta" in row.index and row["Delta"] == row["Delta"]:
            delta_ok = row["Delta"] >= cfg.delta_min

        return all([trend_ok, brk, close_top_ok, vol_ok, mom_ok, patt_ok, atr_pct_ok, gap_ok, ext_ok, beta_ok, delta_ok])

    def score(self, row: pd.Series) -> float:
        """Compute the composite score for a candidate bar."""
        hh = np.nanmax([row.get("HH20", np.nan), row.get("HH50", np.nan)])
        if np.isnan(hh) or row["ATR14"] == 0 or np.isnan(row["ATR14"]):
            brk_z = 0.0
        else:
            brk_z = (row["Close"] - hh) / row["ATR14"]
        avg_rsi = self._avg_rsi_nan_safe(row)
        rsi_component = 0.0 if np.isnan(avg_rsi) else avg_rsi
        _ps = row.get('PatternScore', float('nan'))
        patt = float(_ps) if pd.notna(_ps) else 0.0
        return 0.3 * brk_z + 0.25 * row["ADX14"] + 0.25 * rsi_component + 0.2 * patt

    def exit_levels(self, entry_price: float, atr_val: float) -> tuple[float, float]:
        """Return the stop and target prices given the entry and ATR."""
        stop = entry_price - self.cfg.atr_stop_mult * atr_val
        target = entry_price + self.cfg.atr_target_mult * atr_val
        return stop, target
    def should_early_exit(self, df: pd.DataFrame, idx: int) -> bool:
        if idx <= 0:
            return False
        row   = df.iloc[idx]; prev = df.iloc[idx-1]
        close = row.get("Close", float("nan"))
        sma20 = row.get("SMA20", float("nan"))
        adx14 = row.get("ADX14", 0)
        rsi7  = row.get("RSI7", 0)
        trail = row.get("TrailLine", float("nan"))
        bear  = bool(row.get("BearishPattern", False))
        atr   = row.get("ATR14", float("nan"))
        atrpct= row.get("ATRPCT14", 0.0)
        shock = bool(row.get("DownShock", False) or row.get("GapDown", False))

        trail_break   = (close == close) and (trail == trail) and (close < trail)
        rsi_fail      = (rsi7 < 45) and (close == close) and (sma20 == sma20) and (close < sma20)
        adx_collapse  = (adx14 < 12) and (close == close) and (sma20 == sma20) and (close < sma20)
        atr_spike     = (atrpct >= float(getattr(self.cfg, "atr_pct_spike", 0.10))) and (close < sma20)

        # NEW: fail-safe ? drop > 1.0 ATR from the recent 8-bar close max
        drop_from_hi = (df['RollMaxClose'].iloc[idx] - close) if ((close == close) and (df['RollMaxClose'].iloc[idx] == df['RollMaxClose'].iloc[idx])) else 0
        drop_fail    = (atr == atr) and (drop_from_hi > 1.0 * atr)

        return bool(bear or shock or trail_break or rsi_fail or adx_collapse or atr_spike or drop_fail)




