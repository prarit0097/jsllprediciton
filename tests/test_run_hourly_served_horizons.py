import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.multi_timeframe import config_for_timeframe
from jeena_sikho_tournament.run_hourly import _verify_served_horizons
from jeena_sikho_tournament.storage import Storage


class RunHourlyServedHorizonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.old_env = {
            "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
            "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
            "MARKET_TIMEFRAMES": os.environ.get("MARKET_TIMEFRAMES"),
            "MARKET_YFINANCE_SYMBOL": os.environ.get("MARKET_YFINANCE_SYMBOL"),
        }
        os.environ["APP_DATA_DIR"] = self.tmp
        os.environ["APP_MARKET_DB_FILE"] = "served.sqlite3"
        os.environ["MARKET_TIMEFRAMES"] = "1h,2h,1d"
        os.environ["MARKET_YFINANCE_SYMBOL"] = "JSLL.NS"

    def tearDown(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _seed_horizon(self, timeframe: str):
        cfg = config_for_timeframe(TournamentConfig(), timeframe)
        cfg.registry_path.write_text(
            '{"champions":{"return":{"model_id":"champ","timestamp":"2026-04-13T00:00:00+00:00"}}}',
            encoding="utf-8",
        )
        (cfg.data_dir / f"run_artifact_{cfg.candle_minutes}m.json").write_text(
            '{"served_artifact":{"holdout_metrics":{"price_mae":2.1}}}',
            encoding="utf-8",
        )
        if cfg.candle_minutes >= 1440:
            ts = pd.Timestamp("2026-04-13T10:00:00+00:00")
        else:
            ts = pd.Timestamp("2026-04-13T09:45:00+00:00")
        storage = Storage(cfg.db_path, cfg.ohlcv_table)
        storage.upsert(
            pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1000.0],
                    "source": ["test"],
                },
                index=pd.DatetimeIndex([ts], tz="UTC"),
            )
        )

    def test_verify_served_horizons_passes_when_artifacts_champions_and_data_exist(self):
        for timeframe in ["1h", "2h", "1d"]:
            self._seed_horizon(timeframe)

        with patch("jeena_sikho_tournament.run_hourly.assess_freshness", return_value={"stale": False}):
            _verify_served_horizons(TournamentConfig(), ["1h", "2h", "1d"])

    def test_verify_served_horizons_fails_when_artifact_missing(self):
        self._seed_horizon("1h")
        with patch("jeena_sikho_tournament.run_hourly.assess_freshness", return_value={"stale": False}):
            with self.assertRaises(RuntimeError) as ctx:
                _verify_served_horizons(TournamentConfig(), ["1h", "2h"])
        self.assertIn("2h:missing_run_artifact", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
