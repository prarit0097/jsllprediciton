import os
import unittest
from unittest.mock import patch

from jeena_sikho_dashboard.services import (
    _summarize_ready_metrics,
    get_tournament_summary,
    latest_prediction,
)
from jeena_sikho_tournament.config import TournamentConfig


class Phase7TrustSurfaceMetricTests(unittest.TestCase):
    def test_summarize_ready_metrics_includes_trust_surface_fields(self):
        rows = [
            {
                "predicted_price": 100.0,
                "predicted_price_low": 98.5,
                "predicted_price_high": 101.5,
                "actual_price_1h": 99.5,
                "predicted_return": 0.01,
                "current_price": 99.0,
            },
            {
                "predicted_price": 103.0,
                "predicted_price_low": 101.0,
                "predicted_price_high": 104.0,
                "actual_price_1h": 102.5,
                "predicted_return": 0.02,
                "current_price": 101.0,
            },
            {
                "predicted_price": 101.0,
                "predicted_price_low": 100.0,
                "predicted_price_high": 102.0,
                "actual_price_1h": 101.2,
                "predicted_return": -0.005,
                "current_price": 102.0,
            },
        ]

        summary = _summarize_ready_metrics(rows)

        self.assertIsNotNone(summary)
        for key in (
            "price_rmse",
            "median_abs_error",
            "p90_abs_error",
            "signed_bias_rs",
            "band_80_coverage",
            "sample_count",
        ):
            self.assertIn(key, summary)
        self.assertEqual(summary["sample_count"], 3.0)


class Phase7LatestPredictionContractTests(unittest.TestCase):
    def test_latest_prediction_adds_trust_surface_fields(self):
        prediction_row = {
            "predicted_at": "2026-04-13T09:15:00+00:00",
            "current_price": 666.0,
            "predicted_return": 0.01,
            "predicted_price": 672.66,
            "predicted_price_low": 668.0,
            "predicted_price_high": 676.0,
            "actual_price_1h": None,
            "match_percent": None,
            "status": "pending",
            "model_name": "champion",
            "feature_set": "context",
            "run_id": 7,
            "prediction_target": "y_ret_1d",
            "prediction_horizon_min": 1440,
            "timeframe": "1d",
            "timeframe_minutes": 1440,
            "confidence_pct": 78.0,
            "low_confidence": False,
            "regime": "closing",
        }
        metrics_by_horizon = [
            {
                "timeframe": "1d",
                "target": "y_ret_1d",
                "horizon_minutes": 1440,
                "metrics": {
                    "samples": 64,
                    "sample_count": 64.0,
                    "price_mae": 2.4,
                    "median_abs_error": 1.7,
                    "p90_abs_error": 4.8,
                    "signed_bias_rs": 0.5,
                    "band_80_coverage": 79.0,
                },
                "live_metrics_20": {"price_mae": 2.1},
                "live_mae_20": 2.1,
            }
        ]
        backtest_report = [
            {
                "timeframe": "1d",
                "production_ready": True,
                "holdout_price_mae": 2.2,
                "median_abs_error": 1.7,
                "p90_abs_error": 4.8,
                "signed_bias_rs": 0.5,
                "band_80_coverage": 79.0,
                "sample_count": 64.0,
                "quality_gate_reasons": [],
            }
        ]
        drift_status = {"alert": False, "details": [{"timeframe": "1d", "status": "ok"}]}

        with patch.dict(os.environ, {"MARKET_TIMEFRAMES": "1d"}, clear=False):
            with patch("jeena_sikho_dashboard.services.get_latest_prediction_for_timeframe", return_value=dict(prediction_row)):
                with patch("jeena_sikho_dashboard.services._latest_feature_snapshot", return_value={"freshness": {"stale": False}}):
                    with patch("jeena_sikho_dashboard.services._load_run_artifact", return_value={"served_artifact": {"holdout_metrics": {"price_mae": 2.2}}}):
                        with patch("jeena_sikho_dashboard.services._resolve_last_ready", return_value=None):
                            with patch("jeena_sikho_dashboard.services._collect_metrics_by_horizon", return_value=metrics_by_horizon):
                                with patch("jeena_sikho_dashboard.services._build_backtest_report", return_value=backtest_report):
                                    with patch("jeena_sikho_dashboard.services._compute_drift_status", return_value=drift_status):
                                        with patch("jeena_sikho_dashboard.services._latest_runs_by_timeframe", return_value={"1d": {"run_finished_at": "2026-04-13T06:00:00+00:00"}}):
                                            payload = latest_prediction(TournamentConfig(), update_pending=False)

        row = payload["predictions"][0]
        self.assertEqual(payload["primary_business_timeframe"], "1d")
        self.assertEqual(row["quality_badge"], "trusted")
        self.assertEqual(row["promotion_state"], "active")
        self.assertEqual(row["shadow_status"], "active_incumbent")
        self.assertEqual(row["holdout_price_mae"], 2.2)
        self.assertEqual(row["live_mae_20"], 2.1)
        self.assertEqual(row["confidence_proxy_label"], "model confidence proxy")
        self.assertTrue(row["is_primary_business_horizon"])
        self.assertIn("served_horizon_statuses", payload)
        self.assertEqual(payload["served_horizon_statuses"][0]["timeframe"], "1d")


class Phase7SummaryContractTests(unittest.TestCase):
    def test_summary_includes_served_horizon_statuses(self):
        metrics_by_horizon = [
            {
                "timeframe": "1d",
                "target": "y_ret_1d",
                "horizon_minutes": 1440,
                "metrics": {"samples": 72, "sample_count": 72.0, "median_abs_error": 1.8, "signed_bias_rs": 0.4, "band_80_coverage": 80.0},
                "live_metrics_20": {"price_mae": 2.3},
                "live_mae_20": 2.3,
            }
        ]
        backtest_report = [
            {
                "timeframe": "1d",
                "target": "y_ret_1d",
                "production_ready": True,
                "holdout_price_mae": 2.4,
                "p90_abs_error": 5.5,
                "sample_count": 72.0,
                "quality_gate_reasons": [],
            }
        ]
        champions_by_horizon = {
            "1d": {
                "return": {
                    "model_id": "daily_champion",
                    "champion_age_hours": 6.0,
                    "promotion_state": "active",
                }
            }
        }
        drift_status = {"alert": False, "details": [{"timeframe": "1d", "status": "ok"}]}
        completeness = [{"timeframe": "1d", "lookback_days": 30, "expected": 100, "actual": 99, "completeness_pct": 99.0}]

        with patch.dict(os.environ, {"MARKET_TIMEFRAMES": "1d"}, clear=False):
            with patch("jeena_sikho_dashboard.services.get_latest_run", return_value=None):
                with patch("jeena_sikho_dashboard.services.get_champions", return_value={}):
                    with patch("jeena_sikho_dashboard.services._load_registry", return_value={}):
                        with patch("jeena_sikho_dashboard.services._latest_feature_snapshot", return_value={"freshness": {"stale": False}}):
                            with patch("jeena_sikho_dashboard.services._load_run_artifact", return_value={"served_artifact": {"holdout_metrics": {"price_mae": 2.4}}}):
                                with patch("jeena_sikho_dashboard.services._collect_metrics_by_horizon", return_value=metrics_by_horizon):
                                    with patch("jeena_sikho_dashboard.services._build_backtest_report", return_value=backtest_report):
                                        with patch("jeena_sikho_dashboard.services._compute_drift_status", return_value=drift_status):
                                            with patch("jeena_sikho_dashboard.services._completeness_by_horizon", return_value=completeness):
                                                with patch("jeena_sikho_dashboard.services._champion_detail_from_registry", return_value=champions_by_horizon["1d"]["return"]):
                                                    payload = get_tournament_summary(TournamentConfig())

        self.assertEqual(payload["primary_business_timeframe"], "1d")
        self.assertIn("served_horizon_statuses", payload)
        status = payload["served_horizon_statuses"][0]
        self.assertEqual(status["timeframe"], "1d")
        self.assertEqual(status["quality_badge"], "trusted")
        self.assertTrue(status["is_primary_business_horizon"])


if __name__ == "__main__":
    unittest.main()
