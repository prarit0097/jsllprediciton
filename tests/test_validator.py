from datetime import date

import numpy as np
import pandas as pd

from jeena_sikho_tournament.validator import validate_ohlcv_quality


def _sample_df(start: str, periods: int, freq: str = "60min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    base = np.linspace(100, 102, periods)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.1,
            "volume": np.full(periods, 1000.0),
        },
        index=idx,
    )


def test_validator_detects_duplicate_timestamp():
    df = _sample_df("2026-02-10 03:45:00", 5)
    dup = pd.concat([df, df.iloc[-1:]]).sort_index()
    report = validate_ohlcv_quality(dup, 60, nse_mode=False, holidays=set(), max_missing_ratio=1.0)
    assert report.ok is False
    assert any("duplicate_timestamps" in e for e in report.errors)


def test_validator_nse_boundary_violation():
    # 08:00 IST is outside NSE session
    idx = pd.DatetimeIndex([pd.Timestamp("2026-02-10 02:30:00+00:00")])
    df = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000.0]},
        index=idx,
    )
    report = validate_ohlcv_quality(df, 60, nse_mode=True, holidays=set(), max_missing_ratio=1.0)
    assert report.ok is False
    assert any("nse_session_boundary_violation" in e for e in report.errors)
