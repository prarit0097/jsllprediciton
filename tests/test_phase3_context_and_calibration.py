import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jeena_sikho_dashboard.services import _fit_return_calibrator, _get_recent_bias
from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.features import compute_features, feature_sets


class Phase3ContextAndCalibrationTests(unittest.TestCase):
    def test_context_features_are_emitted(self):
        idx = pd.date_range("2026-01-01", periods=24 * 20, freq="60min", tz="UTC")
        close = 100 + np.cumsum(np.linspace(-0.2, 0.3, len(idx)))
        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": np.full(len(idx), 1000.0),
            },
            index=idx,
        )

        feats = compute_features(df, candle_minutes=60)
        for col in [
            "body_pct",
            "range_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "close_location",
            "day_open_dist",
            "prev_day_high_break",
            "prev_day_low_break",
            "vol_ratio_24_168",
        ]:
            self.assertIn(col, feats.columns)
        self.assertIn("context", feature_sets(feats))

    def test_bias_prefers_matching_regime_when_enough_samples_exist(self):
        cfg = TournamentConfig()
        cfg.bias_window = 18
        cfg.bias_max_abs = 1.0
        opening_rows = [
            {
                "predicted_return": 0.0,
                "current_price": 100.0,
                "actual_price_1h": 110.0,
                "predicted_price": 100.0,
                "match_percent": 90.0,
                "regime": "opening",
            }
            for _ in range(6)
        ]
        closing_rows = [
            {
                "predicted_return": 0.0,
                "current_price": 100.0,
                "actual_price_1h": 90.0,
                "predicted_price": 100.0,
                "match_percent": 90.0,
                "regime": "closing",
            }
            for _ in range(6)
        ]
        rows = opening_rows + closing_rows
        with patch("jeena_sikho_dashboard.services.get_recent_ready_predictions", return_value=rows):
            opening_bias = _get_recent_bias(cfg, "1h", regime="opening")
            closing_bias = _get_recent_bias(cfg, "1h", regime="closing")

        self.assertGreater(opening_bias, 0.05)
        self.assertLess(closing_bias, -0.05)

    def test_calibrator_uses_matching_regime_samples(self):
        cfg = TournamentConfig()
        rows = []
        for x in [0.01, 0.02, 0.03, 0.04, 0.05]:
            y = 0.02 + (1.5 * x)
            rows.append(
                {
                    "predicted_return": x,
                    "current_price": 100.0,
                    "actual_price_1h": float(100.0 * np.exp(y)),
                    "predicted_price": 100.0,
                    "match_percent": 90.0,
                    "regime": "opening",
                }
            )
        for x in [0.01, 0.02, 0.03, 0.04, 0.05]:
            y = -0.01 + (0.5 * x)
            rows.append(
                {
                    "predicted_return": x,
                    "current_price": 100.0,
                    "actual_price_1h": float(100.0 * np.exp(y)),
                    "predicted_price": 100.0,
                    "match_percent": 90.0,
                    "regime": "closing",
                }
            )

        with patch.dict(os.environ, {"CALIBRATION_MIN_SAMPLES": "5", "CALIBRATION_LOOKBACK": "10"}, clear=False):
            with patch("jeena_sikho_dashboard.services.get_recent_ready_predictions", return_value=rows):
                calibrator = _fit_return_calibrator(cfg, "1h", regime="opening")

        self.assertTrue(calibrator["active"])
        self.assertAlmostEqual(calibrator["alpha"], 0.02, places=2)
        self.assertAlmostEqual(calibrator["beta"], 1.5, places=1)


if __name__ == "__main__":
    unittest.main()
