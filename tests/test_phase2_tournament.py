import importlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.features import feature_sets, make_supervised
from jeena_sikho_tournament.models_zoo import ModelSpec, ZeroRegressor
from jeena_sikho_tournament.registry import update_champion
from jeena_sikho_tournament.tournament import (
    _activate_served_ensemble,
    _cap_candidates_by_task_budget,
    _fit_final_model,
    _price_metrics_from_returns,
    _select_diverse_ensemble_candidates,
)


class Phase2TournamentTests(unittest.TestCase):
    def test_config_uses_holdout_by_default(self):
        tracked = {"USE_TEST", "MARKET_TIMEFRAME", "MARKET_TIMEFRAMES"}
        saved = {key: os.environ.get(key) for key in tracked}
        try:
            for key in tracked:
                os.environ.pop(key, None)
            import jeena_sikho_tournament.config as config_module

            config_module = importlib.reload(config_module)
            cfg = config_module.TournamentConfig()
            self.assertTrue(cfg.use_test)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_fit_final_model_returns_trained_model(self):
        idx = pd.date_range("2026-01-01", periods=24 * 140, freq="60min", tz="UTC")
        close = 100 + np.cumsum(np.full(len(idx), 0.05))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": np.full(len(idx), 1000.0),
            },
            index=idx,
        )

        cfg = TournamentConfig()
        cfg.candle_minutes = 60
        cfg.timeframe = "1h"
        sup = make_supervised(df, candle_minutes=60, feature_windows_hours=[2, 4, 8, 12, 24])
        cols = feature_sets(sup)["minimal"]
        spec = ModelSpec("zero_reg", ZeroRegressor(), "return", {"family": "zero", "group": "fast"})

        model = _fit_final_model(spec, "return", cols, sup, cfg)
        preds = model.predict(sup[cols].tail(3))
        self.assertEqual(len(preds), 3)

    def test_task_budgeting_prioritizes_return(self):
        candidates = []
        for task, family_count in [("return", 6), ("direction", 6), ("range", 6)]:
            for idx in range(family_count):
                family = f"fam_{idx}"
                spec = ModelSpec(f"{task}_{idx}", ZeroRegressor(), task, {"family": family, "group": "fast"})
                candidates.append((spec, f"fs_{idx % 2}", ["ret_1c"]))

        capped = _cap_candidates_by_task_budget(candidates, 6, seed=42)
        counts = {"return": 0, "direction": 0, "range": 0}
        for spec, _, _ in capped:
            counts[spec.task] += 1

        self.assertGreaterEqual(counts["return"], counts["direction"])
        self.assertGreaterEqual(counts["direction"], counts["range"])
        self.assertEqual(sum(counts.values()), 6)

    def test_served_ensemble_updates_only_on_replace(self):
        registry = {"ensembles": {"return": {"k": 1, "members": [{"model_id": "old"}]}}}
        candidate = {"k": 2, "members": [{"model_id": "new"}]}

        _activate_served_ensemble(registry, "return", candidate, replaced=False)
        self.assertEqual(registry["ensembles"]["return"]["members"][0]["model_id"], "old")

        _activate_served_ensemble(registry, "return", candidate, replaced=True)
        self.assertEqual(registry["ensembles"]["return"]["members"][0]["model_id"], "new")

    def test_diverse_selection_filters_near_duplicate_predictions(self):
        rows = [
            {
                "spec": ModelSpec("m1", ZeroRegressor(), "return", {"family": "zero", "group": "fast"}),
                "feature_set_id": "a",
                "y_pred": np.array([0.1, 0.2, 0.3, 0.4]),
            },
            {
                "spec": ModelSpec("m2", ZeroRegressor(), "return", {"family": "zero", "group": "fast"}),
                "feature_set_id": "b",
                "y_pred": np.array([0.1, 0.2, 0.3, 0.4]),
            },
            {
                "spec": ModelSpec("m3", ZeroRegressor(), "return", {"family": "zero", "group": "fast"}),
                "feature_set_id": "c",
                "y_pred": np.array([-0.2, 0.1, -0.3, 0.5]),
            },
        ]

        selected = _select_diverse_ensemble_candidates(rows, top_k=2)
        ids = {(row["spec"].name, row["feature_set_id"]) for row in selected}
        self.assertEqual(len(selected), 2)
        self.assertIn(("m1", "a"), ids)
        self.assertIn(("m3", "c"), ids)

    def test_price_metrics_include_accuracy_program_fields(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="60min", tz="UTC")
        eval_df = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
        y_pred = np.array([0.01, 0.0, -0.01])
        y_true = np.array([0.008, -0.002, -0.012])

        metrics = _price_metrics_from_returns(eval_df, y_pred, y_true)

        for key in [
            "price_rmse",
            "median_abs_error",
            "p90_abs_error",
            "smape",
            "signed_bias_rs",
            "band_80_coverage",
            "sample_count",
        ]:
            self.assertIn(key, metrics)
        self.assertEqual(metrics["sample_count"], 3.0)

    def test_update_champion_keeps_shadow_state_when_holdout_gate_fails(self):
        registry = {
            "champions": {
                "return": {
                    "model_id": "incumbent",
                    "timestamp": "2026-04-10T00:00:00+00:00",
                    "final_score": 0.90,
                    "metrics": {"price_mae": 5.0},
                    "holdout_metrics": {"price_mae": 5.0, "sample_count": 90},
                    "promotion_state": "active",
                }
            }
        }
        challenger = {
            "model_id": "challenger",
            "timestamp": "2026-04-11T00:00:00+00:00",
            "final_score": 0.95,
            "metrics": {"price_mae": 4.9},
            "holdout_metrics": {"price_mae": 4.9, "sample_count": 90},
            "val_points": 120,
            "baseline_comparison": {
                "naive_last_close": {"price_mae": 5.5},
                "incumbent": {"price_mae": 5.0},
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
            timeframe="1h",
        )

        self.assertFalse(decision.replaced)
        self.assertEqual(decision.promotion_state, "shadow")
        self.assertTrue(decision.baseline_comparison["naive_last_close"]["passed"])
        self.assertEqual(registry["champions"]["return"]["model_id"], "incumbent")
        self.assertEqual(challenger["promotion_state"], "shadow")

    def test_update_champion_promotes_when_holdout_and_baselines_pass(self):
        registry = {
            "champions": {
                "return": {
                    "model_id": "incumbent",
                    "timestamp": "2026-04-10T00:00:00+00:00",
                    "final_score": 0.90,
                    "metrics": {"price_mae": 5.0},
                    "holdout_metrics": {"price_mae": 5.0, "sample_count": 90},
                    "promotion_state": "active",
                }
            }
        }
        challenger = {
            "model_id": "challenger",
            "timestamp": "2026-04-11T00:00:00+00:00",
            "final_score": 0.95,
            "metrics": {"price_mae": 4.0},
            "holdout_metrics": {
                "price_mae": 2.5,
                "median_abs_error": 1.0,
                "p90_abs_error": 2.0,
                "direction_hit_rate": 60.0,
                "signed_bias_rs": 0.2,
                "sample_count": 90,
            },
            "val_points": 120,
            "baseline_comparison": {
                "naive_last_close": {"price_mae": 5.5},
                "incumbent": {"price_mae": 5.0},
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

        self.assertTrue(decision.replaced)
        self.assertEqual(decision.promotion_state, "active")
        self.assertAlmostEqual(decision.baseline_comparison["incumbent"]["improvement_ratio"], 0.5, places=6)
        self.assertEqual(registry["champions"]["return"]["model_id"], "challenger")


if __name__ == "__main__":
    unittest.main()
