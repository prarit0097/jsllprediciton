import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.drift import should_retrain_on_drift


class DriftTableSourceTests(unittest.TestCase):
    def test_drift_reads_active_market_predictions_table(self):
        tmp = tempfile.mkdtemp()
        try:
            data_dir = Path(tmp)
            db_path = data_dir / "market.sqlite3"
            reg_path = data_dir / "registry_60m.json"
            reg_path.write_text(
                '{"champions":{"return":{"model_id":"champ","timestamp":"2026-04-10T00:00:00+00:00"}}}',
                encoding="utf8",
            )
            with sqlite3.connect(db_path) as con:
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
                    pred = actual + (0.5 if i < 12 else 0.4)
                    rows.append((pred, actual, "ready", "1h"))
                con.executemany(
                    "INSERT INTO market_predictions (predicted_price, actual_price_1h, status, timeframe) VALUES (?, ?, ?, ?)",
                    rows,
                )
            old_env = {
                "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
                "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
                "MARKET_TIMEFRAMES": os.environ.get("MARKET_TIMEFRAMES"),
            }
            try:
                os.environ["APP_DATA_DIR"] = tmp
                os.environ["APP_MARKET_DB_FILE"] = "market.sqlite3"
                os.environ["MARKET_TIMEFRAMES"] = "1h"
                config = TournamentConfig()
                should_run, reason = should_retrain_on_drift(config)
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertFalse(should_run)
        self.assertEqual(reason, "stable_no_drift")


if __name__ == "__main__":
    unittest.main()
