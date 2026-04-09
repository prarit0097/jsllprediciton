import unittest

import numpy as np
import pandas as pd

from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.features import feature_sets, make_supervised
from jeena_sikho_tournament.models_zoo import ModelSpec, ZeroRegressor
from jeena_sikho_tournament.tournament import (
    _activate_served_ensemble,
    _cap_candidates_by_task_budget,
    _fit_final_model,
)


class Phase2TournamentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
