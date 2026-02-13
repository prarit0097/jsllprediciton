import unittest
import numpy as np
import pandas as pd

from jeena_sikho_tournament.features import compute_features, make_supervised


class TestLeakage(unittest.TestCase):
    def test_target_shift(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
        data = pd.DataFrame(
            {
                "open": np.arange(30) + 1,
                "high": np.arange(30) + 2,
                "low": np.arange(30),
                "close": np.arange(30) + 1.5,
                "volume": np.ones(30),
            },
            index=idx,
        )
        sup = make_supervised(data)
        orig = np.log(data["close"]).diff().shift(-1)
        aligned = orig.loc[sup.index]
        self.assertTrue(np.allclose(sup["y_ret"].values, aligned.values, equal_nan=False))

    def test_strict_no_future(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "open": rng.normal(100, 1, size=len(idx)),
                "high": rng.normal(101, 1, size=len(idx)),
                "low": rng.normal(99, 1, size=len(idx)),
                "close": rng.normal(100, 1, size=len(idx)),
                "volume": rng.uniform(100, 200, size=len(idx)),
            },
            index=idx,
        )
        full = compute_features(data).dropna()
        check_points = full.index[50:55]
        for ts in check_points:
            truncated = data.loc[:ts]
            feat_trunc = compute_features(truncated).dropna()
            self.assertTrue(ts in feat_trunc.index)
            full_row = full.loc[ts]
            trunc_row = feat_trunc.loc[ts]
            self.assertTrue(np.allclose(full_row.values, trunc_row.values, equal_nan=True))


if __name__ == "__main__":
    unittest.main()

