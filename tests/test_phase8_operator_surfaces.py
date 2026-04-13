import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from jeena_sikho_dashboard.services import (
    _runtime_capability_status,
    get_tournament_summary,
    latest_prediction,
)
from jeena_sikho_tournament.config import TournamentConfig


class Phase8CapabilityStatusTests(unittest.TestCase):
    def test_runtime_capability_status_reports_degraded_when_ml_stack_missing(self):
        def fake_find_spec(name: str):
            if name in {"yfinance", "joblib"}:
                return object()
            return None

        with patch("importlib.util.find_spec", side_effect=fake_find_spec):
            status = _runtime_capability_status()

        self.assertEqual(status["status"], "baseline-only")
        self.assertTrue(status["degraded"])
        self.assertFalse(status["core_ml_ready"])
        self.assertIn("sklearn", status["missing_modules"])


class Phase8OperatorSurfaceSummaryTests(unittest.TestCase):
    def test_summary_surfaces_capability_baseline_shadow_and_matched_history(self):
        metrics_by_horizon = [
            {
                "timeframe": "1d",
                "target": "y_ret_1d",
                "horizon_minutes": 1440,
                "metrics": {
                    "samples": 72,
                    "sample_count": 72.0,
                    "price_mae": 2.4,
                    "price_rmse": 3.1,
                    "median_abs_error": 1.8,
                    "p90_abs_error": 5.4,
                    "mape": 0.5,
                    "smape": 0.45,
                    "signed_bias_rs": 0.4,
                    "direction_hit_rate": 58.0,
                    "band_80_coverage": 80.0,
                },
                "live_metrics_20": {"price_mae": 2.3},
                "live_mae_20": 2.3,
            }
        ]
        backtest_report = [
            {
                "timeframe": "1d",
                "target": "y_ret_1d",
                "production_ready": True,
                "holdout_price_mae": 2.2,
                "price_mae": 2.4,
                "price_rmse": 3.1,
                "median_abs_error": 1.8,
                "p90_abs_error": 5.4,
                "signed_bias_rs": 0.4,
                "band_80_coverage": 80.0,
                "sample_count": 72.0,
                "quality_gate_reasons": [],
                "holdout_window": {
                    "start": "2026-03-01T00:00:00+00:00",
                    "end": "2026-04-01T00:00:00+00:00",
                },
            }
        ]
        drift_status = {"alert": False, "details": [{"timeframe": "1d", "status": "ok"}]}
        completeness = [{"timeframe": "1d", "lookback_days": 30, "expected": 100, "actual": 99, "completeness_pct": 99.0}]
        registry = {
            "champions": {"return": {"model_id": "daily_champion", "promotion_state": "active"}},
            "promotion_state": {
                "return": {
                    "state": "blocked_by_shadow",
                    "reason": "shadow_window_pending",
                    "updated_at": "2026-04-13T06:30:00+00:00",
                    "model_id": "shadow_return",
                }
            },
            "shadow_candidates": {
                "return": {
                    "model_id": "shadow_return",
                    "baseline_comparison": {
                        "naive_last_close": {"passed": True},
                        "incumbent": {"passed": True},
                    },
                }
            },
        }
        run_artifact = {
            "run_at": "2026-04-13T06:00:00+00:00",
            "holdout_window": {
                "start": "2026-03-01T00:00:00+00:00",
                "end": "2026-04-01T00:00:00+00:00",
            },
            "served_artifact": {
                "task": "return",
                "artifact_id": "daily_champion",
                "holdout_metrics": {"price_mae": 2.2},
                "baseline_comparison": {
                    "naive_last_close": {"passed": True},
                    "incumbent": {"passed": True},
                },
                "promotion_state": "blocked_by_shadow",
                "promotion_reason": "shadow_window_pending",
                "shadow_candidate": {"model_id": "shadow_return"},
            },
        }

        with patch.dict(os.environ, {"MARKET_TIMEFRAMES": "1d"}, clear=False):
            with patch("jeena_sikho_dashboard.services.get_latest_run", return_value=None):
                with patch("jeena_sikho_dashboard.services.get_champions", return_value={}):
                    with patch("jeena_sikho_dashboard.services._load_registry", return_value=registry):
                        with patch("jeena_sikho_dashboard.services._latest_feature_snapshot", return_value={"freshness": {"stale": False}}):
                            with patch("jeena_sikho_dashboard.services._load_run_artifact", return_value=run_artifact):
                                with patch("jeena_sikho_dashboard.services._collect_metrics_by_horizon", return_value=metrics_by_horizon):
                                    with patch("jeena_sikho_dashboard.services._build_backtest_report", return_value=backtest_report):
                                        with patch("jeena_sikho_dashboard.services._compute_drift_status", return_value=drift_status):
                                            with patch("jeena_sikho_dashboard.services._completeness_by_horizon", return_value=completeness):
                                                with patch("jeena_sikho_dashboard.services._champion_detail_from_registry", return_value={"model_id": "daily_champion", "champion_age_hours": 6.0, "promotion_state": "active"}):
                                                    payload = get_tournament_summary(TournamentConfig())

        self.assertIn("capability_status", payload)
        self.assertIn("baseline_accuracy_snapshot", payload)
        self.assertIn("shadow_promotion_report", payload)
        self.assertIn("promotion_decision_log", payload)
        self.assertIn("matched_history_sufficiency", payload)
        self.assertEqual(payload["matched_history_sufficiency"]["status"], "sufficient")
        self.assertTrue(payload["matched_history_sufficiency"]["ready_for_continual_learning"])
        baseline_row = payload["baseline_accuracy_snapshot"]["timeframes"][0]
        self.assertEqual(baseline_row["timeframe"], "1d")
        self.assertEqual(baseline_row["matched_history_status"], "sufficient")
        shadow_row = payload["shadow_promotion_report"]["timeframes"][0]
        self.assertEqual(shadow_row["promotion_state"], "blocked_by_shadow")
        self.assertEqual(shadow_row["shadow_candidate_model_id"], "shadow_return")
        log_states = {entry["state"] for entry in payload["promotion_decision_log"]}
        self.assertIn("blocked_by_shadow", log_states)
        served = payload["served_horizon_statuses"][0]
        self.assertEqual(served["matched_history_status"], "sufficient")
        self.assertTrue(served["matched_history_sufficient"])


class Phase8OperatorSurfaceLatestTests(unittest.TestCase):
    def test_latest_prediction_surfaces_operator_reports_and_matched_history(self):
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
                "holdout_window": {
                    "start": "2026-03-01T00:00:00+00:00",
                    "end": "2026-04-01T00:00:00+00:00",
                },
            }
        ]
        drift_status = {"alert": False, "details": [{"timeframe": "1d", "status": "ok"}]}
        registry = {
            "champions": {"return": {"model_id": "daily_champion", "promotion_state": "active"}},
            "promotion_state": {
                "return": {
                    "state": "active",
                    "reason": "holdout_passed",
                    "updated_at": "2026-04-13T06:30:00+00:00",
                    "model_id": "daily_champion",
                }
            },
        }
        run_artifact = {
            "run_at": "2026-04-13T06:00:00+00:00",
            "served_artifact": {
                "holdout_metrics": {"price_mae": 2.2},
                "promotion_state": "active",
                "promotion_reason": "holdout_passed",
                "baseline_comparison": {
                    "naive_last_close": {"passed": True},
                    "incumbent": {"passed": True},
                },
            },
        }
        anchor_ts = datetime(2026, 4, 13, 9, 15, tzinfo=timezone.utc)

        with patch.dict(os.environ, {"MARKET_TIMEFRAMES": "1d"}, clear=False):
            with patch("jeena_sikho_dashboard.services.get_latest_prediction_for_timeframe", return_value=dict(prediction_row)):
                with patch("jeena_sikho_dashboard.services._latest_feature_snapshot", return_value={"freshness": {"stale": False}, "anchor_ts": anchor_ts, "latest_row": pd.DataFrame({"close": [666.0]}), "source_df": pd.DataFrame({"close": [666.0]})}):
                    with patch("jeena_sikho_dashboard.services._load_run_artifact", return_value=run_artifact):
                        with patch("jeena_sikho_dashboard.services._resolve_last_ready", return_value=None):
                            with patch("jeena_sikho_dashboard.services._collect_metrics_by_horizon", return_value=metrics_by_horizon):
                                with patch("jeena_sikho_dashboard.services._build_backtest_report", return_value=backtest_report):
                                    with patch("jeena_sikho_dashboard.services._compute_drift_status", return_value=drift_status):
                                        with patch("jeena_sikho_dashboard.services._latest_runs_by_timeframe", return_value={"1d": {"run_finished_at": "2026-04-13T06:00:00+00:00"}}):
                                            with patch("jeena_sikho_dashboard.services._load_registry", return_value=registry):
                                                payload = latest_prediction(TournamentConfig(), update_pending=False)

        self.assertIn("capability_status", payload)
        self.assertIn("baseline_accuracy_snapshot", payload)
        self.assertIn("shadow_promotion_report", payload)
        self.assertIn("promotion_decision_log", payload)
        self.assertIn("matched_history_sufficiency", payload)
        row = payload["predictions"][0]
        self.assertEqual(row["matched_history_status"], "sufficient")
        self.assertTrue(row["matched_history_sufficient"])
        self.assertEqual(row["matched_history_required_samples"], 60.0)
        self.assertEqual(payload["shadow_promotion_report"]["timeframes"][0]["promotion_state"], "active")


if __name__ == "__main__":
    unittest.main()
