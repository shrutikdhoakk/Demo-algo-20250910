import pandas as pd

from src.patterns.chart_patterns import bullish_pattern_confirm, pattern_score, bearish_reversal_detect


def test_chart_patterns():
    # Generate a simple trending dataset
    dates = pd.date_range('2025-01-01', periods=25)
    opens = pd.Series(range(100, 125))
    highs = opens + 2
    lows = opens - 2
    closes = opens + 1
    volumes = pd.Series([100000 + i * 500 for i in range(25)])
    df = pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': volumes}, index=dates)
    # Chart pattern boolean
    bp = bullish_pattern_confirm(df)
    assert bp.dtype == bool
    # Pattern score bounded
    ps = pattern_score(df)
    assert ((ps >= 0) & (ps <= 1)).all()
    # Bearish reversal returns boolean
    br = bearish_reversal_detect(df)
    assert br.dtype == bool