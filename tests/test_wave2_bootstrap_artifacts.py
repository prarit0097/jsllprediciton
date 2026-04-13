import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from jeena_sikho_dashboard.db import ensure_tables, insert_prediction
from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.exogenous import load_event_calendar_features
from jeena_sikho_tournament.multi_timeframe import config_for_timeframe
from jeena_sikho_tournament.run_hourly import main
from jeena_sikho_tournament.storage import Storage


class EventCalendarContractTests(unittest.TestCase):
    def test_event_calendar_json_contract_supports_required_and_optional_fields(self):
        idx = pd.date_range("2026-04-10", periods=6, freq="12h", tz="UTC")
        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "events.json"
            event_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {"event_date": "2026-04-10", "symbol": "JSLL.NS"},
                            {"event_date": "2026-04-11", "symbol": "JSLL.NS", "severity": 3},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "EVENT_FEATURES_ENABLE": "1",
                    "EVENT_CALENDAR_FILE": str(event_path),
                    "MARKET_YFINANCE_SYMBOL": "JSLL.NS",
                },
                clear=False,
            ):
                frame, meta = load_event_calendar_features(idx)

        self.assertFalse(frame.empty)
        self.assertTrue(meta["available"])
        self.assertEqual(meta["record_count"], 2)
        self.assertIn("event_severity", frame.columns)
        self.assertEqual(float(frame["event_severity"].max()), 3.0)
        self.assertEqual(float(frame["has_known_event"].max()), 1.0)


class Wave2BootstrapArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.old_env = {
            "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
            "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
            "MARKET_TIMEFRAMES": os.environ.get("MARKET_TIMEFRAMES"),
            "MARKET_YFINANCE_SYMBOL": os.environ.get("MARKET_YFINANCE_SYMBOL"),
            "EXOGENOUS_FEEDS_ENABLE": os.environ.get("EXOGENOUS_FEEDS_ENABLE"),
        }
        os.environ["APP_DATA_DIR"] = self.tmp
        os.environ["APP_MARKET_DB_FILE"] = "wave2.sqlite3"
        os.environ["MARKET_TIMEFRAMES"] = "1d"
        os.environ["MARKET_YFINANCE_SYMBOL"] = "JSLL.NS"
        os.environ["EXOGENOUS_FEEDS_ENABLE"] = "1"
        ensure_tables()

    def tearDown(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _seed_ready_predictions(self, timeframe: str, count: int):
        horizon_min = 1440 if timeframe == "1d" else 60
        base_ts = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)
        for idx in range(count):
            predicted_at = base_ts + timedelta(days=idx)
            current_price = 660.0 + idx
            predicted_price = current_price + 1.0
            actual_price = current_price + (0.5 if idx % 2 == 0 else 1.5)
            insert_prediction(
                {
                    "predicted_at": predicted_at.isoformat(),
                    "current_price": current_price,
                    "predicted_return": 0.01,
                    "predicted_price": predicted_price,
                    "predicted_price_low": predicted_price - 1.0,
                    "predicted_price_high": predicted_price + 1.0,
                    "actual_price_1h": actual_price,
                    "match_percent": 99.0,
                    "status": "ready",
                    "model_name": "champ",
                    "feature_set": "base",
                    "run_id": 11,
                    "prediction_target": f"y_ret_{timeframe}",
                    "prediction_horizon_min": horizon_min,
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "confidence_pct": 82.0,
                    "low_confidence": False,
                    "regime": "closing",
                }
            )

    def test_bootstrap_artifacts_include_baseline_snapshot_and_lineage_when_ready_history_exists(self):
        from jeena_sikho_tournament.bootstrap_artifacts import write_runtime_bootstrap_artifacts

        cfg = config_for_timeframe(TournamentConfig(), "1d")
        cfg.registry_path.write_text(
            json.dumps({"champions": {"return": {"model_id": "champ", "timestamp": "2026-04-10T00:00:00+00:00"}}}),
            encoding="utf-8",
        )
        (cfg.data_dir / f"run_artifact_{cfg.candle_minutes}m.json").write_text(
            json.dumps(
                {
                    "served_artifact": {"holdout_metrics": {"price_mae": 2.4}},
                    "shadow_status": "shadow",
                    "promotion_state": "blocked_by_shadow",
                }
            ),
            encoding="utf-8",
        )
        storage = Storage(cfg.db_path, cfg.ohlcv_table)
        storage.upsert(
            pd.DataFrame(
                {
                    "open": [660.0, 661.0],
                    "high": [662.0, 663.0],
                    "low": [659.0, 660.0],
                    "close": [661.0, 662.0],
                    "volume": [1000.0, 1100.0],
                    "source": ["primary_feed", "primary_feed"],
                },
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-04-10T10:00:00+00:00"), pd.Timestamp("2026-04-13T10:00:00+00:00")],
                    tz="UTC",
                ),
            )
        )
        exo_dir = Path(self.tmp) / "exogenous"
        exo_dir.mkdir(parents=True, exist_ok=True)
        (exo_dir / "nifty_1440m.meta.json").write_text(
            json.dumps(
                {
                    "alias": "nifty",
                    "available": True,
                    "stale": False,
                    "latest_timestamp": "2026-04-11T10:00:00+00:00",
                    "cache_status": "refreshed",
                }
            ),
            encoding="utf-8",
        )
        self._seed_ready_predictions("1d", 60)

        report = write_runtime_bootstrap_artifacts(TournamentConfig(), ["1d"])

        baseline_path = Path(self.tmp) / "baseline_accuracy_snapshot.json"
        lineage_path = Path(self.tmp) / "source_lineage_summary.json"
        self.assertTrue(baseline_path.exists())
        self.assertTrue(lineage_path.exists())
        self.assertIn("1d", report["baseline_accuracy_snapshot"])
        snapshot = report["baseline_accuracy_snapshot"]["1d"]
        self.assertEqual(snapshot["sample_count"], 60)
        self.assertTrue(snapshot["continual_learning_ready"])
        self.assertEqual(snapshot["shadow_status"], "shadow")
        self.assertEqual(snapshot["promotion_state"], "blocked_by_shadow")
        self.assertIn("frozen_window", snapshot)

        lineage = report["source_lineage_summary"]["1d"]
        self.assertEqual(lineage["ohlcv_sources"]["primary_feed"], 2)
        self.assertIn("nifty", lineage["exogenous_signals"])
        self.assertFalse(lineage["data_freshness"]["stale"])


class RunHourlyBootstrapOrderingTests(unittest.TestCase):
    def test_main_refreshes_exogenous_inputs_before_training_and_writes_bootstrap_artifacts_after(self):
        calls = []
        with patch.dict(
            os.environ,
            {
                "APP_DATA_DIR": self._tmpdir(),
                "APP_MARKET_DB_FILE": "ordering.sqlite3",
                "MARKET_TIMEFRAMES": "1h",
                "AUTO_RETRAIN_ON_DRIFT": "1",
            },
            clear=False,
        ):
            with patch("jeena_sikho_tournament.run_hourly.load_env"):
                with patch("jeena_sikho_tournament.run_hourly._is_running", return_value=False):
                    with patch("jeena_sikho_tournament.run_hourly._is_indian_equity", return_value=False):
                        with patch("jeena_sikho_tournament.run_hourly.should_retrain_on_drift", return_value=(True, "ok")):
                            with patch("jeena_sikho_tournament.run_hourly.acquire_run_lock", return_value=(True, {"owner": "scheduler"})):
                                with patch("jeena_sikho_tournament.run_hourly.release_run_lock"):
                                    with patch("jeena_sikho_tournament.run_hourly.resolve_timeframes", return_value=["1h"]):
                                        with patch(
                                            "jeena_sikho_tournament.run_hourly.refresh_public_signal_caches",
                                            side_effect=lambda *args, **kwargs: calls.append("refresh"),
                                        ):
                                            with patch(
                                                "jeena_sikho_tournament.run_hourly.run_tournament",
                                                side_effect=lambda *args, **kwargs: calls.append("tournament"),
                                            ):
                                                with patch(
                                                    "jeena_sikho_tournament.run_hourly._verify_served_horizons",
                                                    side_effect=lambda *args, **kwargs: calls.append("verify"),
                                                ):
                                                    with patch(
                                                        "jeena_sikho_tournament.run_hourly.write_runtime_bootstrap_artifacts",
                                                        side_effect=lambda *args, **kwargs: calls.append("artifact"),
                                                    ):
                                                        main()

        self.assertEqual(calls, ["refresh", "tournament", "verify", "artifact"])

    def _tmpdir(self) -> str:
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        return tmp


if __name__ == "__main__":
    unittest.main()
