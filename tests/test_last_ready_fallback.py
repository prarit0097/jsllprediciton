import os
import tempfile
import unittest


class LastReadyFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old_env = {
            "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
            "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
        }
        os.environ["APP_DATA_DIR"] = self.tmp.name
        os.environ["APP_MARKET_DB_FILE"] = "test.sqlite3"

        from jeena_sikho_dashboard import db

        self.db = db
        db.ensure_tables()

    def tearDown(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _insert_ready_row(self, **overrides):
        base = {
            "predicted_at": "2026-02-17T03:45:00+00:00",
            "current_price": 600.0,
            "predicted_return": 0.01,
            "predicted_price": 606.0,
            "predicted_price_low": 601.0,
            "predicted_price_high": 611.0,
            "actual_price_1h": 603.0,
            "match_percent": 99.5,
            "status": "ready",
            "model_name": "model",
            "feature_set": "base",
            "run_id": 1,
            "prediction_target": "y_ret_2d",
            "prediction_horizon_min": 2880,
            "timeframe": "2d",
            "timeframe_minutes": 2880,
            "confidence_pct": 80.0,
            "low_confidence": False,
            "regime": "mid_session",
        }
        base.update(overrides)
        return self.db.insert_prediction(base)

    def test_fallback_prefers_matching_horizon_when_exact_timeframe_missing(self):
        self._insert_ready_row(timeframe="legacy_2d", prediction_horizon_min=2880, timeframe_minutes=2880)

        row = self.db.get_latest_ready_prediction_fallback("2d", 2880)

        self.assertIsNotNone(row)
        self.assertEqual(row["prediction_horizon_min"], 2880)
        self.assertEqual(row["timeframe_minutes"], 2880)

    def test_fallback_returns_latest_ready_when_only_generic_history_exists(self):
        self._insert_ready_row(timeframe="older", prediction_horizon_min=60, timeframe_minutes=60)

        row = self.db.get_latest_ready_prediction_fallback("7d", 10080)

        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "ready")
        self.assertEqual(row["actual_price_1h"], 603.0)


if __name__ == "__main__":
    unittest.main()
