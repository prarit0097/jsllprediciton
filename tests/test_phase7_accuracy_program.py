import os
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from jeena_sikho_dashboard.db import get_latest_prediction_for_timeframe, insert_prediction
from jeena_sikho_dashboard.services import (
    _prediction_target_timestamp,
    _summarize_ready_metrics,
    refresh_prediction,
    update_pending_predictions,
)
from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.features import make_inference_frame
from jeena_sikho_tournament.tournament import _update_predictions_safe


def _base_ohlcv_with_exogenous(periods: int = 24 * 25) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=periods, freq="60min", tz="UTC")
    close = 650.0 + np.cumsum(np.linspace(-0.35, 0.45, len(idx)))
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": np.full(len(idx), 1000.0),
            "nifty_close": np.linspace(24000.0, 24300.0, len(idx)),
            "sector_close": np.linspace(5200.0, 5300.0, len(idx)),
            "india_vix": np.linspace(13.0, 17.0, len(idx)),
            "usdinr": np.linspace(86.0, 86.6, len(idx)),
            "event_flag": np.zeros(len(idx), dtype=int),
            "exogenous_freshness_minutes": np.full(len(idx), 5.0),
        },
        index=idx,
    )
    return df


class AccuracyProgramSettlementTests(unittest.TestCase):
    def test_tournament_prediction_sync_refreshes_fresh_forecasts_after_settlement(self):
        config = TournamentConfig()

        with ExitStack() as stack:
            update_mock = stack.enter_context(
                patch("jeena_sikho_dashboard.services.update_pending_predictions")
            )
            refresh_mock = stack.enter_context(
                patch("jeena_sikho_dashboard.services.refresh_prediction")
            )

            _update_predictions_safe(config)

        update_mock.assert_called_once_with(config)
        refresh_mock.assert_called_once_with(config)

    def test_daily_horizon_targets_next_nse_close_not_next_open(self):
        pred_at = datetime(2026, 2, 10, 4, 25, tzinfo=timezone.utc)  # 09:55 IST
        expected = datetime(2026, 2, 11, 10, 0, tzinfo=timezone.utc)  # 15:30 IST next trading day

        with patch("jeena_sikho_dashboard.services._is_indian_equity", return_value=True):
            target = _prediction_target_timestamp(pred_at, 1440, 1440)

        self.assertEqual(target, expected)

    def test_pending_daily_prediction_settles_only_from_exact_target_close_bar(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "APP_DATA_DIR": tmp,
                    "APP_MARKET_DB_FILE": "phase7_accuracy.sqlite3",
                },
                clear=False,
            ):
                insert_prediction(
                    {
                        "predicted_at": "2026-02-09T05:00:00+00:00",
                        "current_price": 660.0,
                        "predicted_return": 0.01,
                        "predicted_price": 666.6,
                        "predicted_price_low": 662.0,
                        "predicted_price_high": 671.0,
                        "actual_price_1h": None,
                        "match_percent": None,
                        "status": "pending",
                        "model_name": "daily_model",
                        "feature_set": "base",
                        "run_id": 7,
                        "prediction_target": "y_ret_1d",
                        "prediction_horizon_min": 1440,
                        "timeframe": "1d",
                        "timeframe_minutes": 1440,
                        "confidence_pct": 81.0,
                        "low_confidence": False,
                        "regime": "closing",
                    }
                )

                expected_target_iso = "2026-02-10T10:00:00+00:00"

                def fake_get_close(target_iso: str, table: str = "ohlcv"):
                    self.assertEqual(table, "ohlcv_1440m")
                    self.assertEqual(target_iso, expected_target_iso)
                    return 666.0

                with ExitStack() as stack:
                    stack.enter_context(patch("jeena_sikho_dashboard.services._is_indian_equity", return_value=True))
                    stack.enter_context(patch("jeena_sikho_dashboard.services.get_ohlcv_close_at", side_effect=fake_get_close))
                    update_pending_predictions(TournamentConfig())

                row = get_latest_prediction_for_timeframe("1d")
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(row["status"], "ready")
                self.assertEqual(row["actual_price_1h"], 666.0)


class AccuracyProgramHorizonStatusTests(unittest.TestCase):
    def test_refresh_prediction_blocks_stale_horizon_without_generating_new_forecast(self):
        anchor_ts = datetime(2026, 2, 10, 9, 0, tzinfo=timezone.utc)
        snapshot = {
            "source_df": pd.DataFrame({"close": [660.0]}, index=pd.DatetimeIndex([anchor_ts], tz="UTC")),
            "feature_df": pd.DataFrame({"close": [660.0]}, index=pd.DatetimeIndex([anchor_ts], tz="UTC")),
            "latest_row": pd.DataFrame({"close": [660.0]}, index=pd.DatetimeIndex([anchor_ts], tz="UTC")),
            "anchor_ts": anchor_ts,
            "anchor_price": 660.0,
            "freshness": {
                "stale": True,
                "latest_timestamp": anchor_ts.isoformat(),
                "expected_latest_timestamp": "2026-02-10T10:00:00+00:00",
                "lag_slots": 1,
            },
        }

        with ExitStack() as stack:
            stack.enter_context(patch("jeena_sikho_dashboard.services.update_pending_predictions"))
            stack.enter_context(patch("jeena_sikho_dashboard.services.get_timeframes", return_value=["1d"]))
            stack.enter_context(patch("jeena_sikho_dashboard.services._load_registry", return_value={"champions": {"return": {"model_id": "champ"}}}))
            stack.enter_context(patch("jeena_sikho_dashboard.services._latest_feature_snapshot", return_value=snapshot))
            stack.enter_context(patch("jeena_sikho_dashboard.services.get_latest_prediction_for_timeframe", return_value=None))
            stack.enter_context(patch("jeena_sikho_dashboard.services._predict_return_from_ensemble", side_effect=AssertionError("prediction should be blocked on stale data")))
            stack.enter_context(patch("jeena_sikho_dashboard.services._predict_return_from_champion", side_effect=AssertionError("prediction should be blocked on stale data")))
            stack.enter_context(patch("jeena_sikho_dashboard.services._predict_return_from_direction", side_effect=AssertionError("prediction should be blocked on stale data")))
            stack.enter_context(patch("jeena_sikho_dashboard.services._collect_metrics_by_horizon", return_value=[]))
            stack.enter_context(patch("jeena_sikho_dashboard.services._build_backtest_report", return_value=[]))
            stack.enter_context(patch("jeena_sikho_dashboard.services._compute_drift_status", return_value={"alert": False, "details": []}))
            stack.enter_context(patch("jeena_sikho_dashboard.services.get_latest_run", return_value=None))
            payload = refresh_prediction(TournamentConfig())

        self.assertEqual(len(payload["predictions"]), 1)
        row = payload["predictions"][0]
        self.assertEqual(row["status"], "stale_data")
        self.assertEqual(row["prediction_target"], "y_ret_1d")
        self.assertTrue(row["data_freshness"]["stale"])


class AccuracyProgramMetricContractTests(unittest.TestCase):
    def test_ready_metric_summary_includes_extended_price_bias_and_band_metrics(self):
        rows = [
            {
                "predicted_price": 101.0,
                "predicted_price_low": 100.0,
                "predicted_price_high": 102.0,
                "actual_price_1h": 102.0,
                "predicted_return": np.log(101.0 / 100.0),
                "current_price": 100.0,
            },
            {
                "predicted_price": 99.0,
                "predicted_price_low": 99.0,
                "predicted_price_high": 100.0,
                "actual_price_1h": 97.0,
                "predicted_return": np.log(99.0 / 100.0),
                "current_price": 100.0,
            },
            {
                "predicted_price": 103.0,
                "predicted_price_low": 100.0,
                "predicted_price_high": 104.0,
                "actual_price_1h": 101.0,
                "predicted_return": np.log(103.0 / 100.0),
                "current_price": 100.0,
            },
        ]

        summary = _summarize_ready_metrics(rows)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertAlmostEqual(summary["mae"], 5.0 / 3.0, places=6)
        self.assertIn("price_rmse", summary)
        self.assertIn("median_abs_error", summary)
        self.assertIn("p90_abs_error", summary)
        self.assertIn("signed_bias_rs", summary)
        self.assertIn("band_80_coverage", summary)
        self.assertAlmostEqual(summary["price_rmse"], np.sqrt(3.0), places=6)
        self.assertAlmostEqual(summary["median_abs_error"], 2.0, places=6)
        self.assertAlmostEqual(summary["p90_abs_error"], 2.0, places=6)
        self.assertAlmostEqual(summary["signed_bias_rs"], 1.0, places=6)
        self.assertAlmostEqual(summary["band_80_coverage"], (2.0 / 3.0) * 100.0, places=6)


class AccuracyProgramPromotionSurfaceTests(unittest.TestCase):
    def test_refresh_prediction_surfaces_shadow_and_promotion_metadata(self):
        pred_row = {
            "id": 1,
            "predicted_at": "2026-02-11T10:15:00+00:00",
            "current_price": 660.0,
            "predicted_return": 0.01,
            "predicted_price": 666.6,
            "predicted_price_low": 662.0,
            "predicted_price_high": 671.0,
            "actual_price_1h": None,
            "match_percent": None,
            "status": "pending",
            "model_name": "shadow_candidate",
            "feature_set": "base",
            "run_id": 9,
            "prediction_target": "y_ret_1d",
            "prediction_horizon_min": 1440,
            "timeframe": "1d",
            "timeframe_minutes": 1440,
            "confidence_pct": 83.0,
            "low_confidence": False,
            "regime": "closing",
        }
        anchor_ts = datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc)
        snapshot = {
            "source_df": pd.DataFrame({"close": [660.0]}, index=pd.DatetimeIndex([anchor_ts], tz="UTC")),
            "feature_df": pd.DataFrame({"close": [660.0]}, index=pd.DatetimeIndex([anchor_ts], tz="UTC")),
            "latest_row": pd.DataFrame({"close": [660.0]}, index=pd.DatetimeIndex([anchor_ts], tz="UTC")),
            "anchor_ts": anchor_ts,
            "anchor_price": 660.0,
            "freshness": {"stale": False, "lag_slots": 0},
        }
        run_artifact = {
            "served_artifact": {
                "holdout_metrics": {"price_mae": 1.8},
                "baseline_comparison": {"naive_last_close": "pass", "incumbent": "pass"},
            },
            "shadow_status": "shadow",
            "promotion_state": "blocked_by_shadow",
            "confidence_proxy_label": "provisional",
        }

        with ExitStack() as stack:
            stack.enter_context(patch("jeena_sikho_dashboard.services.update_pending_predictions"))
            stack.enter_context(patch("jeena_sikho_dashboard.services.get_timeframes", return_value=["1d"]))
            stack.enter_context(patch("jeena_sikho_dashboard.services.get_latest_run", return_value=None))
            stack.enter_context(patch("jeena_sikho_dashboard.services.get_latest_prediction_for_timeframe", return_value=pred_row.copy()))
            stack.enter_context(patch("jeena_sikho_dashboard.services._load_registry", return_value={"champions": {"return": {"model_id": "champ"}}}))
            stack.enter_context(patch("jeena_sikho_dashboard.services._latest_feature_snapshot", return_value=snapshot))
            stack.enter_context(patch("jeena_sikho_dashboard.services._load_run_artifact", return_value=run_artifact))
            stack.enter_context(patch("jeena_sikho_dashboard.services._resolve_last_ready", return_value=None))
            stack.enter_context(patch("jeena_sikho_dashboard.services._collect_metrics_by_horizon", return_value=[]))
            stack.enter_context(patch("jeena_sikho_dashboard.services._build_backtest_report", return_value=[]))
            stack.enter_context(patch("jeena_sikho_dashboard.services._compute_drift_status", return_value={"alert": False, "details": []}))
            payload = refresh_prediction(TournamentConfig())

        row = payload["predictions"][0]
        self.assertEqual(row["holdout_primary_metric"], 1.8)
        self.assertIn("shadow_status", row)
        self.assertIn("promotion_state", row)
        self.assertIn("confidence_proxy_label", row)
        self.assertIn("baseline_comparison", row)
        self.assertEqual(row["shadow_status"], "shadow")
        self.assertEqual(row["promotion_state"], "blocked_by_shadow")
        self.assertEqual(row["confidence_proxy_label"], "provisional")
        self.assertEqual(row["baseline_comparison"]["incumbent"], "pass")


class AccuracyProgramExogenousJoinTests(unittest.TestCase):
    def test_inference_frame_preserves_joined_exogenous_columns_when_fresh(self):
        df = _base_ohlcv_with_exogenous()

        frame = make_inference_frame(df, candle_minutes=60)

        self.assertFalse(frame.empty)
        for column in [
            "nifty_close",
            "sector_close",
            "india_vix",
            "usdinr",
            "event_flag",
            "exogenous_freshness_minutes",
        ]:
            self.assertIn(column, frame.columns)
        self.assertAlmostEqual(float(frame.iloc[-1]["nifty_close"]), float(df.iloc[-1]["nifty_close"]))
        self.assertAlmostEqual(float(frame.iloc[-1]["india_vix"]), float(df.iloc[-1]["india_vix"]))

    def test_inference_frame_drops_rows_with_missing_exogenous_join_values(self):
        df = _base_ohlcv_with_exogenous()
        missing_ts = df.index[-1]
        df.loc[missing_ts, "nifty_close"] = np.nan

        frame = make_inference_frame(df, candle_minutes=60)

        self.assertFalse(frame.empty)
        self.assertNotIn(missing_ts, frame.index)


if __name__ == "__main__":
    unittest.main()
