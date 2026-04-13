import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.models_zoo import ModelSpec, ZeroRegressor
from jeena_sikho_tournament.registry import load_registry, update_champion
from jeena_sikho_tournament.tournament import (
    _baseline_comparison,
    _run_artifact_payload,
)
from jeena_sikho_tournament.drift import should_retrain_on_drift


class Wave34GatingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.old_env = {
            "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
            "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
            "MARKET_TIMEFRAMES": os.environ.get("MARKET_TIMEFRAMES"),
            "SHADOW_PROMOTION_MIN_SETTLED_1D": os.environ.get("SHADOW_PROMOTION_MIN_SETTLED_1D"),
            "PROD_CAPABILITY_EXPECTATION": os.environ.get("PROD_CAPABILITY_EXPECTATION"),
            "PROD_MAX_CROSS_HORIZON_REGRESSION": os.environ.get("PROD_MAX_CROSS_HORIZON_REGRESSION"),
            "DRIFT_ROLLBACK_BREACHES": os.environ.get("DRIFT_ROLLBACK_BREACHES"),
        }
        os.environ["APP_DATA_DIR"] = self.tmp
        os.environ["APP_MARKET_DB_FILE"] = "wave34.sqlite3"
        os.environ["MARKET_TIMEFRAMES"] = "1h,2h,1d"
        os.environ["PROD_MAX_CROSS_HORIZON_REGRESSION"] = "0.03"
        os.environ["DRIFT_ROLLBACK_BREACHES"] = "2"

    def tearDown(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _config(self, timeframe: str = "1d") -> TournamentConfig:
        cfg = TournamentConfig()
        cfg.timeframe = timeframe
        if timeframe == "1d":
            cfg.candle_minutes = 1440
            cfg.registry_path = Path(self.tmp) / "registry_1440m.json"
        elif timeframe == "2h":
            cfg.candle_minutes = 120
            cfg.registry_path = Path(self.tmp) / "registry_120m.json"
        else:
            cfg.candle_minutes = 60
            cfg.registry_path = Path(self.tmp) / "registry_60m.json"
        return cfg

    def _task_results(self):
        best = {
            "spec": ModelSpec("candidate", ZeroRegressor(), "return", {"family": "zero"}),
            "feature_set_id": "minimal",
            "holdout_metrics": {"price_mae": 2.4, "sample_count": 90},
            "metrics": {"price_mae": 2.4},
        }
        naive = {
            "spec": ModelSpec("naive_last", ZeroRegressor(), "return", {"family": "naive", "baseline": True}),
            "feature_set_id": "minimal",
            "holdout_metrics": {"price_mae": 3.2, "sample_count": 90},
            "metrics": {"price_mae": 3.2},
        }
        return best, [best, naive]

    def test_baseline_comparison_enforces_shadow_settled_minimum_from_env(self):
        os.environ["SHADOW_PROMOTION_MIN_SETTLED_1D"] = "10"
        cfg = self._config("1d")
        best, task_results = self._task_results()

        rows = [{"predicted_price": 100.0, "actual_price_1h": 100.5} for _ in range(9)]
        with patch("jeena_sikho_tournament.tournament.get_recent_ready_predictions", return_value=rows):
            comparison = _baseline_comparison("return", best, task_results, {}, cfg)

        report = comparison["shadow_promotion_report"]
        self.assertEqual(report["settled_count"], 9)
        self.assertEqual(report["required_min_settled"], 10)
        self.assertFalse(report["passed"])
        self.assertFalse(comparison["shadow_window_passed"])

    def test_update_champion_blocks_runtime_capability_and_cross_horizon_regression(self):
        registry = {}
        challenger = {
            "model_id": "challenger",
            "timestamp": "2026-04-14T00:00:00+00:00",
            "final_score": 1.0,
            "metrics": {"price_mae": 2.4},
            "holdout_metrics": {
                "price_mae": 2.4,
                "sample_count": 90,
                "p90_abs_error": 5.0,
                "direction_hit_rate": 60.0,
                "signed_bias_rs": 0.1,
            },
            "val_points": 600,
            "baseline_comparison": {
                "runtime_capability": {
                    "status": "baseline-only",
                    "expected": "core-ml",
                    "passed": False,
                    "missing_modules": ["sklearn", "joblib"],
                }
            },
        }

        decision = update_champion(
            registry,
            "return",
            challenger,
            min_val_points=50,
            margin=0.02,
            margin_override=0.05,
            cooldown_hours=0,
            timeframe="1d",
        )

        self.assertEqual(decision.promotion_state, "blocked_by_dq")
        self.assertEqual(decision.reason, "runtime_capability_insufficient")
        self.assertTrue(registry["promotion_decision_log"])

        challenger["baseline_comparison"] = {
            "runtime_capability": {"status": "core-ml", "expected": "core-ml", "passed": True},
            "cross_horizon": {
                "passed": False,
                "entries": [{"timeframe": "1h", "regression_ratio": 0.08, "passed": False}],
            },
        }
        decision = update_champion(
            registry,
            "return",
            challenger,
            min_val_points=50,
            margin=0.02,
            margin_override=0.05,
            cooldown_hours=0,
            timeframe="1d",
        )
        self.assertEqual(decision.promotion_state, "blocked_by_holdout")
        self.assertEqual(decision.reason, "cross_horizon_regression_blocked")

    def test_run_artifact_payload_includes_shadow_and_promotion_reports(self):
        cfg = self._config("1d")
        registry = {
            "promotion_decision_log": [{"task": "return", "promotion_state": "shadow", "reason": "shadow_window_gate_failed"}],
            "shadow_promotion_report": {
                "return": {"settled_count": 6, "required_min_settled": 10, "passed": False}
            },
        }
        task_results = {
            "return": {
                "best": {
                    "holdout_metrics": {"price_mae": 2.2},
                    "baseline_comparison": {"shadow_window_passed": False},
                },
                "promotion_state": "shadow",
                "decision": "shadow_window_gate_failed",
                "served_artifact_id": "candidate",
                "shadow_candidate": {"model_id": "candidate"},
            }
        }

        payload = _run_artifact_payload(
            cfg,
            task_results,
            holdout_train_df=None,
            holdout_eval_df=None,
            run_at="2026-04-14T00:00:00+00:00",
            registry=registry,
            capability_status={"status": "baseline-only", "expected": "baseline-only", "passed": True},
        )

        self.assertIn("promotion_decision_log", payload)
        self.assertIn("shadow_promotion_report", payload)
        self.assertEqual(payload["shadow_promotion_report"]["required_min_settled"], 10)

    def test_drift_marks_registry_rolled_back_after_repeated_breaches(self):
        os.environ["MARKET_TIMEFRAMES"] = "1h"
        cfg = self._config("1h")
        cfg.registry_path.write_text(
            json.dumps(
                {
                    "champions": {
                        "return": {
                            "model_id": "champ",
                            "timestamp": "2026-04-10T00:00:00+00:00",
                            "promotion_state": "active",
                        }
                    }
                }
            ),
            encoding="utf8",
        )
        with sqlite3.connect(cfg.db_path) as con:
            con.execute(
                """
                CREATE TABLE market_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_price REAL,
                    actual_price_1h REAL,
                    status TEXT,
                    timeframe TEXT
                )
                """
            )
            rows = []
            for i in range(24):
                actual = 100.0 + i
                pred = actual + (0.5 if i < 12 else 3.0)
                rows.append((pred, actual, "ready", "1h"))
            con.executemany(
                "INSERT INTO market_predictions (predicted_price, actual_price_1h, status, timeframe) VALUES (?, ?, ?, ?)",
                rows,
            )

        should_run, reason = should_retrain_on_drift(cfg)
        self.assertTrue(should_run)
        self.assertEqual(reason, "drift_alert_1h")

        should_run, reason = should_retrain_on_drift(cfg)
        self.assertTrue(should_run)
        self.assertEqual(reason, "drift_alert_1h")

        registry = load_registry(cfg.registry_path)
        self.assertEqual(registry["promotion_state"]["return"]["state"], "rolled_back")
        self.assertEqual(registry["champions"]["return"]["promotion_state"], "rolled_back")


if __name__ == "__main__":
    unittest.main()
