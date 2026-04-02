import unittest

import numpy as np
import pandas as pd

from jeena_sikho_tournament.features import compute_features, make_supervised


class TestFeatureQuality(unittest.TestCase):
    def test_wilder_rsi_tracks_trend_direction(self):
        idx = pd.date_range("2026-01-01", periods=80, freq="60min", tz="UTC")
        up_close = pd.Series([100.0 + (i * 0.5) for i in range(len(idx))], index=idx)
        down_close = pd.Series([100.0 - (i * 0.5) for i in range(len(idx))], index=idx)

        up = pd.DataFrame(
            {
                "open": up_close,
                "high": up_close + 0.2,
                "low": up_close - 0.2,
                "close": up_close,
                "volume": np.full(len(idx), 1000.0),
            },
            index=idx,
        )
        down = pd.DataFrame(
            {
                "open": down_close,
                "high": down_close + 0.2,
                "low": down_close - 0.2,
                "close": down_close,
                "volume": np.full(len(idx), 1000.0),
            },
            index=idx,
        )

        up_rsi = compute_features(up, candle_minutes=60)["rsi_14"].dropna()
        down_rsi = compute_features(down, candle_minutes=60)["rsi_14"].dropna()

        self.assertFalse(up_rsi.empty)
        self.assertFalse(down_rsi.empty)
        self.assertGreater(float(up_rsi.iloc[-1]), 60.0)
        self.assertLess(float(down_rsi.iloc[-1]), 40.0)

    def test_make_supervised_drops_zero_volume_rows(self):
        idx = pd.date_range("2026-01-01", periods=500, freq="60min", tz="UTC")
        base = np.linspace(100.0, 120.0, len(idx))
        volume = np.full(len(idx), 1000.0)
        volume[300] = 0.0

        df = pd.DataFrame(
            {
                "open": base,
                "high": base + 0.4,
                "low": base - 0.4,
                "close": base + 0.1,
                "volume": volume,
            },
            index=idx,
        )

        sup = make_supervised(df, candle_minutes=60, feature_windows_hours=[2, 4, 8, 12, 24])
        self.assertFalse(sup.empty)
        self.assertNotIn(idx[300], sup.index)


if __name__ == "__main__":
    unittest.main()
