import unittest
from unittest.mock import patch

from jeena_sikho_tournament.models_zoo import ModelSpec, ZeroRegressor, _filter_specs_by_horizon, get_candidates


class Phase4ModelZooTests(unittest.TestCase):
    def test_horizon_filter_keeps_robust_return_families(self):
        specs = [
            ModelSpec("huber_demo", ZeroRegressor(), "return", {"family": "huber", "group": "fast"}),
            ModelSpec("qreg_demo", ZeroRegressor(), "return", {"family": "qreg", "group": "fast"}),
        ]

        kept = _filter_specs_by_horizon(specs, "return", 60, True)
        families = {spec.meta.get("family") for spec in kept}
        self.assertIn("huber", families)
        self.assertIn("qreg", families)

    def test_get_candidates_can_surface_robust_families(self):
        robust_specs = [
            ModelSpec("huber_demo", ZeroRegressor(), "return", {"family": "huber", "group": "fast"}),
            ModelSpec("qreg_demo", ZeroRegressor(), "return", {"family": "qreg", "group": "fast"}),
        ]
        with patch("jeena_sikho_tournament.models_zoo._sklearn_candidates", return_value=robust_specs):
            with patch("jeena_sikho_tournament.models_zoo._optional_boosters", return_value=[]):
                specs = get_candidates("return", max_candidates=10, enable_dl=False, candle_minutes=60, strict_horizon_pool=True)

        families = {spec.meta.get("family") for spec in specs}
        self.assertIn("huber", families)
        self.assertIn("qreg", families)


if __name__ == "__main__":
    unittest.main()
