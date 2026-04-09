import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from jeena_sikho_tournament.validator import assess_freshness


class Phase5PredictionContractTests(unittest.TestCase):
    def test_freshness_marks_stale_when_latest_bar_lags_watermark(self):
        idx = pd.date_range("2026-02-10 03:45:00+00:00", periods=3, freq="60min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0, 1000.0, 1000.0],
            },
            index=idx,
        )
        now_utc = datetime(2026, 2, 10, 10, 30, tzinfo=timezone.utc)
        freshness = assess_freshness(df, 60, nse_mode=False, holidays=set(), now_utc=now_utc)

        self.assertTrue(freshness["stale"])
        self.assertIsNotNone(freshness["expected_latest_timestamp"])

    def test_pending_prediction_does_not_settle_from_live_spot_when_target_bar_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_env = {
                "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
                "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
                "MARKET_YFINANCE_SYMBOL": os.environ.get("MARKET_YFINANCE_SYMBOL"),
            }
            try:
                os.environ["APP_DATA_DIR"] = tmp
                os.environ["APP_MARKET_DB_FILE"] = "test.sqlite3"
                os.environ["MARKET_YFINANCE_SYMBOL"] = "BTC-USD"

                from jeena_sikho_dashboard.db import get_latest_prediction_for_timeframe, insert_prediction
                from jeena_sikho_dashboard.services import update_pending_predictions
                from jeena_sikho_tournament.config import TournamentConfig

                insert_prediction(
                    {
                        "predicted_at": "2026-01-01T00:00:00+00:00",
                        "current_price": 100.0,
                        "predicted_return": 0.01,
                        "predicted_price": 101.0,
                        "predicted_price_low": 99.0,
                        "predicted_price_high": 103.0,
                        "actual_price_1h": None,
                        "match_percent": None,
                        "status": "pending",
                        "model_name": "model",
                        "feature_set": "base",
                        "run_id": 1,
                        "prediction_target": "y_ret_1h",
                        "prediction_horizon_min": 60,
                        "timeframe": "1h",
                        "timeframe_minutes": 60,
                        "confidence_pct": 80.0,
                        "low_confidence": False,
                        "regime": "always",
                    }
                )

                with patch("jeena_sikho_dashboard.services.get_live_price", side_effect=AssertionError("live spot fallback used")):
                    update_pending_predictions(TournamentConfig())

                row = get_latest_prediction_for_timeframe("1h")
                self.assertIsNotNone(row)
                self.assertEqual(row["status"], "pending")
                self.assertIsNone(row["actual_price_1h"])
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_provenance_fields_are_added_non_breaking(self):
        from jeena_sikho_dashboard.services import _decorate_prediction_provenance

        row = {
            "predicted_at": "2026-02-10T04:15:00+00:00",
            "current_price": 123.45,
            "regime": "opening",
        }
        _decorate_prediction_provenance(row, holdout_metric=1.25, calibration_regime="opening")
        self.assertEqual(row["forecast_anchor_at"], row["predicted_at"])
        self.assertEqual(row["forecast_anchor_price"], row["current_price"])
        self.assertTrue(row["generated_from_completed_bar"])
        self.assertEqual(row["selection_basis"], "holdout")
        self.assertEqual(row["holdout_primary_metric"], 1.25)
        self.assertEqual(row["calibration_regime"], "opening")


if __name__ == "__main__":
    unittest.main()
