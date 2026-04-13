import importlib
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import django
import numpy as np
import pandas as pd
from django.test import RequestFactory, SimpleTestCase

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "btcsite.settings")
django.setup()

from jeena_sikho_dashboard import db as dashboard_db
from jeena_sikho_dashboard import views as dashboard_views
from jeena_sikho_dashboard.services import get_scoreboard
from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.features import feature_sets, make_supervised
from jeena_sikho_tournament.models_zoo import ModelSpec, ZeroRegressor
from jeena_sikho_tournament.tournament import _score_candidate_once


class Phase6TournamentHardeningTests(unittest.TestCase):
    def test_return_candidate_scoring_runs_without_name_error(self):
        idx = pd.date_range("2026-01-01", periods=24 * 40, freq="60min", tz="UTC")
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
        sup = make_supervised(df, candle_minutes=60, feature_windows_hours=[2, 4, 8, 12, 24])
        cols = feature_sets(sup)["minimal"]
        spec = ModelSpec("zero_reg", ZeroRegressor(), "return", {"family": "zero", "group": "fast"})

        result = _score_candidate_once(spec, "return", cols, sup.iloc[:-24], sup.iloc[-24:], TournamentConfig())

        self.assertIn("primary", result)
        self.assertTrue(np.isfinite(float(result["primary"])))


class Phase6MutationAuthTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_tournament_run_requires_admin_token(self):
        request = self.factory.post(
            "/api/jeena-sikho/tournament/run",
            data="{}",
            content_type="application/json",
        )
        with patch.dict(os.environ, {"APP_ADMIN_TOKEN": "secret-token"}, clear=False):
            response = dashboard_views.api_tournament_run(request)
        self.assertEqual(response.status_code, 403)

    def test_tournament_run_accepts_valid_admin_token(self):
        request = self.factory.post(
            "/api/jeena-sikho/tournament/run",
            data="{}",
            content_type="application/json",
            HTTP_X_APP_ADMIN_TOKEN="secret-token",
        )
        with patch.dict(os.environ, {"APP_ADMIN_TOKEN": "secret-token"}, clear=False):
            with patch("jeena_sikho_dashboard.views.run_tournament_async", return_value={"status": "started"}):
                response = dashboard_views.api_tournament_run(request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.content.decode("utf-8"))["status"], "started")

    def test_prediction_refresh_requires_admin_token(self):
        request = self.factory.post(
            "/api/jeena-sikho/prediction/refresh",
            data="{}",
            content_type="application/json",
        )
        with patch.dict(os.environ, {"APP_ADMIN_TOKEN": "secret-token"}, clear=False):
            response = dashboard_views.api_prediction_refresh(request)
        self.assertEqual(response.status_code, 403)

    def test_prediction_refresh_accepts_valid_admin_token(self):
        request = self.factory.post(
            "/api/jeena-sikho/prediction/refresh",
            data="{}",
            content_type="application/json",
            HTTP_X_APP_ADMIN_TOKEN="secret-token",
        )
        with patch.dict(os.environ, {"APP_ADMIN_TOKEN": "secret-token"}, clear=False):
            with patch("jeena_sikho_dashboard.views.refresh_prediction", return_value={"predictions": []}):
                response = dashboard_views.api_prediction_refresh(request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.content.decode("utf-8"))["predictions"], [])


class Phase6RunLockTests(unittest.TestCase):
    def test_run_lock_blocks_second_owner_until_release(self):
        from jeena_sikho_tournament.run_lock import acquire_run_lock, release_run_lock

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            first_acquired, _ = acquire_run_lock("web", data_dir=data_dir)
            second_acquired, _ = acquire_run_lock("scheduler", data_dir=data_dir)

            self.assertTrue(first_acquired)
            self.assertFalse(second_acquired)

            self.assertTrue(release_run_lock("web", data_dir=data_dir))

            third_acquired, _ = acquire_run_lock("scheduler", data_dir=data_dir)
            self.assertTrue(third_acquired)


class Phase6ScoreboardTests(unittest.TestCase):
    def test_scoreboard_includes_latest_rows_for_each_timeframe(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_env = {
                "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
                "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
                "MARKET_TIMEFRAMES": os.environ.get("MARKET_TIMEFRAMES"),
            }
            try:
                os.environ["APP_DATA_DIR"] = tmp
                os.environ["APP_MARKET_DB_FILE"] = "scoreboard.sqlite3"
                os.environ["MARKET_TIMEFRAMES"] = "1h,2h"

                run_1h = dashboard_db.insert_run(
                    "2026-04-13T09:00:00+00:00",
                    "daily",
                    10,
                    timeframe="1h",
                    candle_minutes=60,
                )
                dashboard_db.insert_scores(
                    run_1h,
                    [
                        {
                            "rank": 1,
                            "target": "return",
                            "feature_set": "base",
                            "model_name": "ReturnModel1H",
                            "family": "linear",
                            "final_score": 0.91,
                            "primary_metric_name": "holdout_price_mae",
                            "primary_metric_value": 1.2,
                            "trading_score": 0.6,
                            "stability_penalty": 0.02,
                            "is_champion": True,
                            "run_at": "2026-04-13T09:00:00+00:00",
                        }
                    ],
                )

                run_2h = dashboard_db.insert_run(
                    "2026-04-13T10:00:00+00:00",
                    "daily",
                    12,
                    timeframe="2h",
                    candle_minutes=120,
                )
                dashboard_db.insert_scores(
                    run_2h,
                    [
                        {
                            "rank": 1,
                            "target": "return",
                            "feature_set": "context",
                            "model_name": "ReturnModel2H",
                            "family": "boosting",
                            "final_score": 0.88,
                            "primary_metric_name": "holdout_price_mae",
                            "primary_metric_value": 1.5,
                            "trading_score": 0.55,
                            "stability_penalty": 0.03,
                            "is_champion": True,
                            "run_at": "2026-04-13T10:00:00+00:00",
                        }
                    ],
                )

                rows = get_scoreboard(TournamentConfig(), limit=20)
                timeframes = {row["timeframe"] for row in rows}
                models = {row["model_name"] for row in rows}

                self.assertEqual(timeframes, {"1h", "2h"})
                self.assertEqual(models, {"ReturnModel1H", "ReturnModel2H"})
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class Phase6LegacyCleanupTests(unittest.TestCase):
    def test_config_defaults_are_jsll_focused(self):
        tracked = {
            "MARKET_SYMBOL",
            "MARKET_YFINANCE_SYMBOL",
            "MARKET_TIMEFRAME",
            "MARKET_TIMEFRAMES",
            "MARKET_CANDLE_MINUTES",
        }
        saved = {key: os.environ.get(key) for key in tracked}
        try:
            for key in tracked:
                os.environ.pop(key, None)
            import jeena_sikho_tournament.config as config_module

            config_module = importlib.reload(config_module)
            config = config_module.TournamentConfig()

            self.assertEqual(config.symbol, "JSLL/INR")
            self.assertEqual(config.yfinance_symbol, "JSLL.NS")
            self.assertEqual(config.timeframe, "1h")
            self.assertEqual(config.candle_minutes, 60)
            self.assertEqual(config.ohlcv_table, "ohlcv")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_ensure_tables_renames_legacy_btc_predictions_table(self):
        tmp = tempfile.mkdtemp()
        old_env = {
            "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
            "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
        }
        try:
            os.environ["APP_DATA_DIR"] = tmp
            os.environ["APP_MARKET_DB_FILE"] = "legacy.sqlite3"
            db_path = Path(tmp) / "legacy.sqlite3"
            with sqlite3.connect(db_path) as con:
                con.execute(
                    """
                    CREATE TABLE btc_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        predicted_at TEXT,
                        current_price REAL,
                        predicted_return REAL,
                        predicted_price REAL,
                        predicted_price_low REAL,
                        predicted_price_high REAL,
                        actual_price_1h REAL,
                        match_percent REAL,
                        status TEXT,
                        model_name TEXT,
                        feature_set TEXT,
                        run_id INTEGER,
                        prediction_target TEXT,
                        prediction_horizon_min INTEGER,
                        timeframe TEXT,
                        timeframe_minutes INTEGER,
                        confidence_pct REAL,
                        low_confidence INTEGER,
                        regime TEXT
                    )
                    """
                )
                con.execute(
                    """
                    INSERT INTO btc_predictions (
                        predicted_at, current_price, predicted_return, predicted_price,
                        predicted_price_low, predicted_price_high, actual_price_1h, match_percent,
                        status, model_name, feature_set, run_id, prediction_target,
                        prediction_horizon_min, timeframe, timeframe_minutes, confidence_pct,
                        low_confidence, regime
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "2026-04-13T09:00:00+00:00",
                        100.0,
                        0.01,
                        101.0,
                        99.0,
                        103.0,
                        None,
                        None,
                        "pending",
                        "model",
                        "base",
                        1,
                        "y_ret_1h",
                        60,
                        "1h",
                        60,
                        75.0,
                        0,
                        "opening",
                    ),
                )

            dashboard_db.ensure_tables()

            with sqlite3.connect(db_path) as con:
                tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                count = con.execute("SELECT COUNT(*) FROM market_predictions").fetchone()[0]

            self.assertIn("market_predictions", tables)
            self.assertNotIn("btc_predictions", tables)
            self.assertEqual(count, 1)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
