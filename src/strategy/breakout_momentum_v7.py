"""Breakout + Momentum Swing v7.0 strategy implementation.

This module implements a simplified variant of the Breakout + Momentum Swing
strategy that draws inspiration from the research‑driven version outlined
earlier in the conversation.  The aim of this implementation is to be
compatible with the existing backtest engine while providing upgraded entry
criteria, momentum confirmation, volume filters and scoring.  It does not
attempt to exactly replicate every nuance of the earlier pseudo‑code (for
example, it omits the regime filter based on an external index and the
residual momentum computation) because the provided dataset does not include
an index series.  Instead the focus is on price/volume action, trend
confirmation via moving averages and momentum gauges, candlestick pattern
detection and a composite scoring function.

Key features:

* Detects breakouts above recent highs (20 and 50 periods) with a minimum
  thrust measured in ATR multiples and a volume surge relative to the
  rolling mean.
* Confirms breakouts with momentum filters: ADX14, RSI7 and RSI14 must
  exceed configurable thresholds and the price must be in a healthy uptrend
  relative to its moving averages.
* Requires a bullish candlestick structure on the entry bar (engulfing,
  hammer, marubozu or morning star) to reduce false signals.
* Assigns a per‑bar score based on the z‑score of the breakout thrust,
  relative volume, composite RSI and the candlestick pattern flag.  Higher
  scores indicate stronger candidates.
* Provides an early exit signal when the price closes below the 20‑period
  SMA for a configurable number of consecutive bars, when the RSI14
  deteriorates or when a bearish candlestick pattern appears.

The class exposes a ``compute_features`` method that augments the input
DataFrame with the necessary columns and an ``is_entry`` method that
evaluates whether a bar qualifies as an entry.  The ``score`` method
returns the precomputed composite score and ``should_early_exit`` flags
potential early exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..features.indicators import (
    rsi,
    atr,
    adx,
    hhv,
    sma,
    zscore,
)
from ..patterns.candlesticks import bullish_on_bar


@dataclass(frozen=True)
class StrategyConfigV7:
    """Parameter container for the Breakout Momentum v7.0 strategy.

    The values defined here act as sensible defaults.  They can be
    overridden by keys of the same name in the YAML configuration file.
    """

    # Lookback periods for highest highs
    hh_lookbacks: tuple[int, int] = (20, 50)
    # Minimum multiple of rolling average volume required to qualify a breakout
    vol_mult_min: float = 1.5
    # Minimum thrust as a fraction of ATR (absolute close‑open divided by ATR)
    thrust_atr: float = 0.2
    # Momentum thresholds
    adx_min: float = 20.0
    rsi7_min: float = 55.0
    rsi14_min: float = 50.0
    # Moving average trend confirmation: price above SMA50, SMA50 above SMA200
    require_trend: bool = True
    # Candlestick confirmation: require bullish pattern on entry bar
    require_bullish_pattern: bool = True
    # Scoring parameters (weights for z‑scored components)
    score_weights: dict[str, float] = None  # Set in __post_init__
    # Early exit configuration
    early_sma20_below_days: int = 2
    early_rsi14_below: float = 35.0
    early_rsi14_bars: int = 2
    # Stop/target multipliers (used by engine)
    atr_stop_mult: float = 1.2
    atr_target_mult: float = 3.5
    # Trailing stop multiplier (chandelier)
    chandelier_mult: float = 3.0
    # Trigger to activate the trailing stop.  Once the open profit
    # exceeds trail_trigger_atr * ATR, the trailing stop is enabled.
    trail_trigger_atr: float = 1.5

    # --- New fields for alpha/beta integration ---
    # Rolling window length (in bars) used to estimate the average
    # return (alpha) for each symbol.  A positive alpha indicates
    # that the stock has been drifting higher on average over this
    # period.  A shorter window makes the alpha more responsive,
    # whereas a longer window smooths the noise.
    alpha_lookback: int = 20
    # Rolling window length for the beta estimate.  In this simplified
    # implementation we approximate beta as 1.0 because we do not
    # have a benchmark index series available.  These fields are
    # provided for completeness and future extensions.  They can be
    # overridden via the YAML config but will not materially change
    # the behaviour in this implementation.
    beta_lookback: int = 60
    # Beta filter range.  Candidates whose beta estimate falls
    # outside this band will be rejected.  Since beta is 1.0 in
    # practice, this range simply acts as a sanity check.
    beta_min: float = 0.6
    beta_max: float = 1.8

    def __post_init__(self) -> None:
        """Initialize default score weights.

        The strategy combines multiple z‑scored components into a single
        composite score.  If the user does not provide explicit
        ``score_weights`` the defaults below are applied.  The
        components are:

        * ``thrust`` – breakout thrust measured in ATR units
        * ``rsi`` – composite RSI across multiple timeframes
        * ``vol`` – relative volume z‑score
        * ``pattern`` – binary indicator of a bullish candlestick
        * ``alpha`` – rolling average of returns as a proxy for stock
          specific alpha

        The weights sum to one to maintain a consistent scale.  Users
        may override any subset of these via the YAML configuration;
        the missing weights will be filled with the defaults and the
        final set will be normalized.
        """
        if object.__getattribute__(self, "score_weights") is None:
            default_weights = {
                "thrust": 0.25,
                "rsi":    0.25,
                "vol":    0.20,
                "pattern":0.10,
                "alpha":  0.20,
            }
            object.__setattr__(self, "score_weights", default_weights)


class BreakoutMomentumV7:
    """Encapsulates the logic for the Breakout + Momentum Swing v7.0 strategy."""

    def __init__(self, cfg: Any) -> None:
        # Accept either a dataclass instance or a simple namespace.  We avoid
        # mutating the passed object and instead keep a reference.  The
        # ``compute_features`` method will not attempt to merge YAML
        # overrides; that is handled upstream in the engine.
        self.cfg = cfg

    # ------------------------------------------------------------------
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators and intermediate columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Input OHLCV data with columns ``Open``, ``High``, ``Low``,
            ``Close`` and ``Volume``.  The index should be datetime‑like.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with additional columns used
            downstream by ``is_entry``, ``score`` and ``should_early_exit``.
        """
        # Work on a copy to avoid mutating caller's data
        df = df.copy()

        # Basic moving averages
        df['SMA20']  = sma(df['Close'], 20)
        df['SMA50']  = sma(df['Close'], 50)
        df['SMA200'] = sma(df['Close'], 200)

        # Highest highs over the lookback windows, shifted by one to avoid
        # lookahead bias.  Use min_periods=1 so early bars are not NaN.
        n_short, n_long = self.cfg.hh_lookbacks if hasattr(self.cfg, 'hh_lookbacks') else (20, 50)
        df['HH_SHORT'] = hhv(df['High'], n_short).shift(1)
        df['HH_LONG']  = hhv(df['High'], n_long).shift(1)

        # ATR14
        df['ATR14'] = atr(df, 14)

        # Volume mean and z‑score (20‑period rolling)
        df['VolAvg20'] = df['Volume'].rolling(20, min_periods=1).mean()
        # Replace zero std with NaN to avoid divide by zero
        vol_std = df['Volume'].rolling(20, min_periods=1).std(ddof=0)
        df['VolZ20'] = (df['Volume'] - df['VolAvg20']) / vol_std.replace(0, np.nan)

        # ADX and RSIs
        df['ADX14'] = adx(df, 14)
        df['RSI7']  = rsi(df['Close'], 7)
        df['RSI14'] = rsi(df['Close'], 14)
        df['RSI21'] = rsi(df['Close'], 21)
        # Composite RSI: weighted average emphasising shorter lookbacks
        df['RSIComp'] = 0.5 * df['RSI7'] + 0.3 * df['RSI14'] + 0.2 * df['RSI21']

        # Candlestick patterns: bullish flag
        df['BullishPattern'] = bullish_on_bar(df)

        # Bearish pattern for early exit: define a simple bearish candle
        rng = (df['High'] - df['Low']).abs().replace(0, np.nan)
        bodyD = (df['Open'] - df['Close']).clip(lower=0)
        df['BearPattern'] = (df['Close'] < df['Open']) & (bodyD >= 0.6 * rng)

        # ------------------------------------------------------------------
        # Alpha/beta estimation.  We compute a simple rolling mean of the
        # daily percentage returns as a proxy for stock specific alpha.
        # A positive alpha implies that the stock has been trending higher
        # on average over the specified lookback window.  In a fully
        # fledged implementation one would regress the stock's returns
        # against a benchmark index to estimate both alpha and beta.  Since
        # this dataset does not include an index series we approximate
        # beta as unity.  These columns are still computed and can be
        # used to filter candidates based on the configured beta range and
        # to incorporate alpha into the scoring function.
        try:
            df['RET'] = df['Close'].pct_change().fillna(0.0)
        except Exception:
            df['RET'] = 0.0
        lookback_alpha = int(getattr(self.cfg, 'alpha_lookback', 20))
        df['Alpha'] = df['RET'].rolling(lookback_alpha, min_periods=lookback_alpha).mean()
        lookback_beta = int(getattr(self.cfg, 'beta_lookback', 60))
        ret_std = df['RET'].rolling(lookback_beta, min_periods=lookback_beta).std(ddof=0)
        df['Beta'] = ret_std / ret_std.replace(0, np.nan)
        df['ResidMom'] = df['Alpha']

        # Breakout flag: close breaks above recent highs with volume surge and thrust
        thr = (df['Close'] - df['Open']).abs() / df['ATR14'].replace(0, np.nan)
        df['ThrustATR'] = thr
        vol_ok = df['Volume'] >= (self.cfg.vol_mult_min if hasattr(self.cfg, 'vol_mult_min') else 1.5) * df['VolAvg20']
        thrust_ok = thr >= (self.cfg.thrust_atr if hasattr(self.cfg, 'thrust_atr') else 0.2)
        breakout = (
            (df['Close'] > df['HH_SHORT']) | (df['Close'] > df['HH_LONG'])
        ) & vol_ok & thrust_ok
        df['Breakout'] = breakout

        # Trend filter: price above SMA50, SMA50 above SMA200 and SMA20 above SMA50
        trend_ok = (
            (df['Close'] > df['SMA50']) &
            (df['SMA50'] > df['SMA200']) &
            (df['SMA20'] > df['SMA50'])
        ) if getattr(self.cfg, 'require_trend', True) else True

        # Momentum filters
        adx_ok = df['ADX14'] >= (self.cfg.adx_min if hasattr(self.cfg, 'adx_min') else 20)
        rsi7_ok  = df['RSI7']  >= (self.cfg.rsi7_min if hasattr(self.cfg, 'rsi7_min') else 55)
        rsi14_ok = df['RSI14'] >= (self.cfg.rsi14_min if hasattr(self.cfg, 'rsi14_min') else 50)

        # Candlestick confirmation (optional)
        pattern_ok = df['BullishPattern'] if getattr(self.cfg, 'require_bullish_pattern', True) else True

        # Alpha/beta gates: require positive alpha and beta within configured bounds
        alpha_ok = df['Alpha'] > 0
        beta_min = float(getattr(self.cfg, 'beta_min', 0.6))
        beta_max = float(getattr(self.cfg, 'beta_max', 1.8))
        beta_ok = (df['Beta'] >= beta_min) & (df['Beta'] <= beta_max)

        df['EntrySignal'] = breakout & trend_ok & adx_ok & rsi7_ok & rsi14_ok & pattern_ok & alpha_ok & beta_ok

        # Scoring: compute z‑scores for thrust, composite RSI, volume, and alpha.  The
        # pattern contributes a fixed value.  We merge the user supplied
        # weights with defaults and normalise the result.
        thrust_z = zscore(df['ThrustATR'], 60).fillna(0)
        rsi_z    = zscore(df['RSIComp'], 60).fillna(0)
        vol_z    = zscore(df['VolZ20'], 60).fillna(0)
        alpha_z  = zscore(df['Alpha'], 60).fillna(0)
        patt_val = df['BullishPattern'].astype(float)
        default_w = {'thrust': 0.25, 'rsi': 0.25, 'vol': 0.20, 'pattern': 0.10, 'alpha': 0.20}
        w_cfg = getattr(self.cfg, 'score_weights', None)
        if isinstance(w_cfg, dict):
            w = {**default_w, **{k: w_cfg[k] for k in w_cfg if k in default_w}}
            s = sum(float(v) for v in w.values())
            if s > 0:
                w = {k: float(v) / s for k, v in w.items()}
        else:
            w = default_w
        df['Score'] = (
            w.get('thrust', 0.25) * thrust_z +
            w.get('rsi',    0.25) * rsi_z +
            w.get('vol',    0.20) * vol_z +
            w.get('pattern',0.10) * patt_val +
            w.get('alpha',  0.20) * alpha_z
        ) * 100.0

        return df

    # ------------------------------------------------------------------
    def is_entry(self, row: pd.Series, prev: pd.Series) -> bool:
        """Evaluate whether the current bar qualifies as an entry.

        The engine calls this for each symbol on each bar.  We inspect
        the precomputed ``EntrySignal`` flag on the row.  A bar cannot
        qualify if it is the first row or when indicators have not yet
        converged and return NaN.

        Parameters
        ----------
        row : pandas.Series
            Current bar (with computed features).
        prev : pandas.Series
            Previous bar; included for interface compatibility but not
            explicitly used by this strategy.

        Returns
        -------
        bool
            True if the bar is a valid entry, False otherwise.
        """
        # Avoid entries on the very first bar or when features are NaN
        if row.isnull().any():
            return False
        return bool(row.get('EntrySignal', False))

    # ------------------------------------------------------------------
    def score(self, row: pd.Series) -> float:
        """Return the composite score for the current bar.

        Parameters
        ----------
        row : pandas.Series
            Row containing the computed 'Score' column.

        Returns
        -------
        float
            Score value; higher indicates a more attractive candidate.
        """
        sc = row.get('Score')
        return float(sc) if sc == sc else float('nan')

    # ------------------------------------------------------------------
    def should_early_exit(self, df: pd.DataFrame, idx: int) -> bool:
        """Determine whether a position should be exited early.

        An early exit is triggered if either of the following conditions
        occurs:

        * The closing price has been below the 20‑period SMA for
          ``early_sma20_below_days`` consecutive bars.
        * The RSI14 has been below ``early_rsi14_below`` for
          ``early_rsi14_bars`` consecutive bars.
        * A bearish candlestick pattern appears on the current bar.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with computed features for a single symbol.
        idx : int
            Integer position within the DataFrame corresponding to the
            current bar.

        Returns
        -------
        bool
            True if the position should be exited early, False otherwise.
        """
        # If not enough history, never exit early
        if idx is None or idx <= 0:
            return False

        # Ensure we have at least the required number of bars
        n_sma = int(getattr(self.cfg, 'early_sma20_below_days', 2))
        n_rsi = int(getattr(self.cfg, 'early_rsi14_bars', 2))
        # Price below SMA20 for n_sma consecutive bars
        if n_sma > 0 and idx >= n_sma - 1:
            window = df.iloc[idx - n_sma + 1: idx + 1]
            below = window['Close'] < window['SMA20']
            if below.all():
                return True
        # RSI14 below threshold for n_rsi consecutive bars
        if n_rsi > 0 and idx >= n_rsi - 1:
            window = df.iloc[idx - n_rsi + 1: idx + 1]
            below_rsi = window['RSI14'] < float(getattr(self.cfg, 'early_rsi14_below', 35.0))
            if below_rsi.all():
                return True
        # Bearish candlestick pattern on current bar
        cur_bear = bool(df.iloc[idx].get('BearPattern', False))
        if cur_bear:
            return True
        return False