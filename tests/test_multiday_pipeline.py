import pandas as pd

from jeena_sikho_tournament.data_sources import _aggregate_ohlcv, _expected_nse_slots
from jeena_sikho_tournament.market_calendar import IST


def test_expected_nse_slots_for_2d_uses_every_second_trading_close():
    start = pd.Timestamp("2026-02-09 00:00:00+00:00")  # Monday
    end = pd.Timestamp("2026-02-13 23:59:00+00:00")    # Friday
    slots = _expected_nse_slots(start, end, 2880, holidays=set())
    got = [ts.tz_convert(IST).strftime("%Y-%m-%d %H:%M") for ts in slots]
    assert got == [
        "2026-02-09 15:30",
        "2026-02-11 15:30",
        "2026-02-13 15:30",
    ]


def test_aggregate_ohlcv_builds_full_2d_bars_only():
    idx = pd.DatetimeIndex(
        [
            "2026-02-09 10:00:00+00:00",  # Mon 15:30 IST
            "2026-02-10 10:00:00+00:00",  # Tue
            "2026-02-11 10:00:00+00:00",  # Wed
            "2026-02-12 10:00:00+00:00",  # Thu
            "2026-02-13 10:00:00+00:00",  # Fri (tail partial for 2d)
        ]
    )
    hourly_like = pd.DataFrame(
        {
            "timestamp_utc": idx,
            "open": [100, 110, 120, 130, 140],
            "high": [101, 111, 121, 131, 141],
            "low": [99, 109, 119, 129, 139],
            "close": [100.5, 110.5, 120.5, 130.5, 140.5],
            "volume": [1, 2, 3, 4, 5],
            "source": ["yfinance"] * 5,
        }
    )
    out = _aggregate_ohlcv(hourly_like, 2880, symbol="JSLL.NS")
    assert len(out) == 2
    got_ts = [ts.tz_convert(IST).strftime("%Y-%m-%d %H:%M") for ts in pd.to_datetime(out["timestamp_utc"], utc=True)]
    assert got_ts == ["2026-02-10 15:30", "2026-02-12 15:30"]
    assert list(out["open"]) == [100, 120]
    assert list(out["close"]) == [110.5, 130.5]
    assert list(out["volume"]) == [3, 7]
