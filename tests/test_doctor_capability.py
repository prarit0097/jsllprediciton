from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from jeena_sikho_tournament import diagnostics


class DoctorCapabilityTests(TestCase):
    def _module_map(self, **overrides):
        defaults = {
            "pandas": True,
            "numpy": True,
            "sklearn": False,
            "joblib": False,
            "xgboost": False,
            "lightgbm": False,
            "catboost": False,
            "ccxt": False,
            "yfinance": True,
        }
        defaults.update(overrides)
        return defaults

    def _patch_modules(self, mapping):
        return patch(
            "jeena_sikho_tournament.diagnostics._has_module",
            side_effect=lambda name: mapping.get(name, True),
        )

    def test_inspect_runtime_capabilities_baseline_only_when_core_missing(self):
        mapping = self._module_map()
        with self._patch_modules(mapping):
            report = diagnostics.inspect_runtime_capabilities(required_capability="core-ml")

        self.assertEqual(report["capability_status"], "baseline-only")
        self.assertEqual(report["dependency_health"], "blocked")
        self.assertFalse(report["promotion_ready"])
        self.assertEqual(report["data_provider_status"], "degraded")
        self.assertIn("sklearn", report["missing_core_ml"])
        self.assertIn("joblib", report["missing_core_ml"])
        self.assertIn("ccxt", report["missing_data_providers"])

    def test_inspect_runtime_capabilities_core_ml_when_boosters_missing(self):
        mapping = self._module_map(sklearn=True, joblib=True)
        with self._patch_modules(mapping):
            report = diagnostics.inspect_runtime_capabilities(required_capability="core-ml")

        self.assertEqual(report["capability_status"], "core-ml")
        self.assertEqual(report["dependency_health"], "degraded")
        self.assertTrue(report["promotion_ready"])
        self.assertCountEqual(report["missing_ensemble"], ["xgboost", "lightgbm", "catboost"])

    def test_inspect_runtime_capabilities_full_ensemble_when_everything_installed(self):
        mapping = self._module_map(
            sklearn=True,
            joblib=True,
            xgboost=True,
            lightgbm=True,
            catboost=True,
            ccxt=True,
            yfinance=True,
        )
        with self._patch_modules(mapping):
            report = diagnostics.inspect_runtime_capabilities(required_capability="full-ensemble")

        self.assertEqual(report["capability_status"], "full-ensemble")
        self.assertEqual(report["dependency_health"], "healthy")
        self.assertEqual(report["data_provider_status"], "ready")
        self.assertTrue(report["promotion_ready"])
        self.assertEqual(report["missing_ensemble"], [])
        self.assertEqual(report["missing_data_providers"], [])

    def test_check_dependencies_fails_when_required_capability_unavailable(self):
        mapping = self._module_map(sklearn=True, joblib=True)
        with self._patch_modules(mapping):
            result, install_cmds, capability = diagnostics.check_dependencies(required_capability="full-ensemble")

        self.assertEqual(result.status, "FAIL")
        self.assertEqual(capability["capability_status"], "core-ml")
        self.assertFalse(capability["promotion_ready"])
        self.assertIn("full-ensemble", result.message)
        self.assertTrue(any("xgboost" in cmd for cmd in install_cmds))

    def test_build_doctor_report_exposes_report_friendly_capability_fields(self):
        mapping = self._module_map(sklearn=True, joblib=True)

        def _pass(name):
            return diagnostics.CheckResult(name)

        with self._patch_modules(mapping), \
            patch("jeena_sikho_tournament.diagnostics.check_structure", return_value=_pass("Structure")), \
            patch("jeena_sikho_tournament.diagnostics.check_storage", return_value=_pass("Storage")), \
            patch("jeena_sikho_tournament.diagnostics.check_data", return_value=(_pass("Data"), {"rows": "0"})), \
            patch("jeena_sikho_tournament.diagnostics.check_features", return_value=_pass("Features")), \
            patch("jeena_sikho_tournament.diagnostics.check_tests", return_value=_pass("Tests")), \
            patch("jeena_sikho_tournament.diagnostics.check_model_zoo", return_value=(_pass("Model Zoo"), {"total_candidates": "12"})), \
            patch("jeena_sikho_tournament.diagnostics.check_dry_run", return_value=_pass("Dry Run")), \
            patch("jeena_sikho_tournament.diagnostics.check_registry_predictor", return_value=_pass("Registry + Predictor")):
            report = diagnostics.build_doctor_report(Path("."), required_capability="core-ml")

        self.assertIn("capability_status", report)
        self.assertIn("capability_report", report)
        self.assertIn("results", report)
        self.assertEqual(report["capability_status"], "core-ml")
        self.assertEqual(report["capability_report"]["required_capability"], "core-ml")
        self.assertTrue(report["capability_report"]["promotion_ready"])
