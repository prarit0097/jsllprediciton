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
