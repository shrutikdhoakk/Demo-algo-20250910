import pandas as pd

from src.strategy.breakout_momentum_v3 import BreakoutMomentumV3, StrategyConfig


def test_compute_and_entry():
    # Create a small synthetic dataset
    df = pd.DataFrame(
        {
            'Open': [10, 11, 12, 13, 14, 15],
            'High': [11, 12, 13, 14, 15, 16],
            'Low': [9, 10, 11, 12, 13, 14],
            'Close': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            'Volume': [100, 110, 120, 130, 140, 150],
        },
        index=pd.date_range('2025-01-01', periods=6),
    )
    strat = BreakoutMomentumV3(StrategyConfig())
    feats = strat.compute_features(df)
    # Check that required columns exist
    for col in ['ATR14', 'ADX14', 'RSI7', 'RSI14', 'RSI21', 'SMA20', 'PatternConfirmed', 'PatternScore']:
        assert col in feats.columns
    # is_entry should return a boolean
    if len(feats) >= 2:
        row = feats.iloc[-1]
        prev = feats.iloc[-2]
        res = strat.is_entry(row, prev)
        assert isinstance(res, bool)
        # score returns float
        s = strat.score(row)
        assert isinstance(s, float)