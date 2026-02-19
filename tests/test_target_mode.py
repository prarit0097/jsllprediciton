import os

import numpy as np
import pandas as pd

from jeena_sikho_tournament.features import make_supervised


def test_make_supervised_adds_volnorm_target():
    os.environ["RETURN_TARGET_MODE"] = "volnorm_logret"
    idx = pd.date_range("2026-01-01", periods=500, freq="60min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 700 + np.cumsum(rng.normal(0, 0.8, size=len(idx)))
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.2, size=len(idx)),
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": rng.uniform(1000, 5000, size=len(idx)),
        },
        index=idx,
    )
    sup = make_supervised(df, candle_minutes=60, feature_windows_hours=[2, 4, 8, 12, 24])
    assert "y_ret_raw" in sup.columns
    assert "y_ret_model" in sup.columns
    assert "target_scale" in sup.columns
    # y_ret remains raw target for metric/trading consistency.
    assert np.allclose(sup["y_ret"].to_numpy(), sup["y_ret_raw"].to_numpy())


def test_multiday_nse_targets_use_correct_future_day_close(monkeypatch):
    monkeypatch.setenv("MARKET_YFINANCE_SYMBOL", "JSLL.NS")
    monkeypatch.setenv("EVENT_DAY_DROP_FROM_TRAIN", "0")
    monkeypatch.setenv("EXOGENOUS_REQUIRED", "0")
    idx = pd.date_range("2026-01-05 10:00:00+00:00", periods=7, freq="24h", tz="UTC")
    close = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0], dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )

    sup_2d = make_supervised(df, candle_minutes=2880, feature_windows_hours=[24, 48, 72])
    sup_3d = make_supervised(df, candle_minutes=4320, feature_windows_hours=[24, 48, 72])
    assert "y_ret_2d" in sup_2d.columns
    assert "y_ret_3d" in sup_3d.columns

    first_idx = sup_2d.index.min()
    first_pos = list(idx).index(first_idx)
    expected_2d = np.log(close[first_pos + 2] / close[first_pos])
    got_2d = float(sup_2d.loc[first_idx, "y_ret_2d"])
    assert np.isclose(got_2d, expected_2d, rtol=1e-9, atol=1e-12)

    first_idx_3d = sup_3d.index.min()
    first_pos_3d = list(idx).index(first_idx_3d)
    expected_3d = np.log(close[first_pos_3d + 3] / close[first_pos_3d])
    got_3d = float(sup_3d.loc[first_idx_3d, "y_ret_3d"])
    assert np.isclose(got_3d, expected_3d, rtol=1e-9, atol=1e-12)


def test_1d_nse_target_can_use_next_day_open(monkeypatch):
    monkeypatch.setenv("MARKET_YFINANCE_SYMBOL", "JSLL.NS")
    monkeypatch.setenv("EVENT_DAY_DROP_FROM_TRAIN", "0")
    monkeypatch.setenv("EXOGENOUS_REQUIRED", "0")
    monkeypatch.setenv("DAILY_TARGET_POINT", "open")
    idx = pd.date_range("2026-01-05 10:00:00+00:00", periods=6, freq="24h", tz="UTC")
    opens = np.array([100.0, 105.0, 110.0, 120.0, 130.0, 140.0], dtype=float)
    closes = np.array([101.0, 106.0, 111.0, 121.0, 131.0, 141.0], dtype=float)
    df = pd.DataFrame(
        {
            "open": opens,
            "high": closes + 1.0,
            "low": opens - 1.0,
            "close": closes,
            "volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )
    sup = make_supervised(df, candle_minutes=1440, feature_windows_hours=[24, 48, 72])
    first_idx = sup.index.min()
    pos = list(idx).index(first_idx)
    expected = np.log(opens[pos + 1] / closes[pos])
    got = float(sup.loc[first_idx, "y_ret_1d"])
    assert np.isclose(got, expected, rtol=1e-9, atol=1e-12)


def test_exogenous_required_auto_falls_back_when_exo_unavailable(monkeypatch):
    monkeypatch.setenv("EXOGENOUS_ENABLE", "0")
    monkeypatch.setenv("EXOGENOUS_REQUIRED", "auto")
    monkeypatch.setenv("EVENT_DAY_DROP_FROM_TRAIN", "0")
    idx = pd.date_range("2026-01-01", periods=200, freq="60min", tz="UTC")
    base = np.linspace(100.0, 120.0, len(idx))
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.2,
            "volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )
    sup = make_supervised(df, candle_minutes=60, feature_windows_hours=[2, 4, 8, 12, 24])
    assert not sup.empty
