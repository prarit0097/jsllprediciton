import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from jeena_sikho_dashboard.services import _latest_feature_snapshot
from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.features import make_supervised
from jeena_sikho_tournament.storage import Storage


class Phase1AlignmentTests(unittest.TestCase):
    def test_make_supervised_targets_do_not_change_with_future_outlier(self):
        idx = pd.date_range("2026-01-01", periods=500, freq="60min", tz="UTC")
        close = np.linspace(100.0, 112.0, len(idx))
        close[-1] = 220.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(len(idx), 1000.0),
            },
            index=idx,
        )

        full = make_supervised(df, candle_minutes=60, feature_windows_hours=[2, 4, 8, 12, 24])
        truncated = make_supervised(df.iloc[:-1], candle_minutes=60, feature_windows_hours=[2, 4, 8, 12, 24])
        common = full.index.intersection(truncated.index)

        self.assertFalse(common.empty)
        self.assertTrue(
            np.allclose(
                full.loc[common, "y_ret_raw"].to_numpy(),
                truncated.loc[common, "y_ret_raw"].to_numpy(),
                equal_nan=False,
            )
        )

    def test_latest_feature_snapshot_uses_latest_completed_feature_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_env = {
                "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
                "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
                "MARKET_TIMEFRAME": os.environ.get("MARKET_TIMEFRAME"),
                "MARKET_TIMEFRAMES": os.environ.get("MARKET_TIMEFRAMES"),
                "MARKET_YFINANCE_SYMBOL": os.environ.get("MARKET_YFINANCE_SYMBOL"),
            }
            try:
                os.environ["APP_DATA_DIR"] = tmp
                os.environ["APP_MARKET_DB_FILE"] = "phase1.sqlite3"
                os.environ["MARKET_TIMEFRAME"] = "1h"
                os.environ["MARKET_TIMEFRAMES"] = "1h"
                os.environ["MARKET_YFINANCE_SYMBOL"] = "BTC-USD"

                config = TournamentConfig()
                end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
                idx = pd.date_range(end=end, periods=500, freq="60min", tz="UTC")
                close = 100 + np.cumsum(np.full(len(idx), 0.1))
                df = pd.DataFrame(
                    {
                        "open": close,
                        "high": close + 0.5,
                        "low": close - 0.5,
                        "close": close,
                        "volume": np.full(len(idx), 1000.0),
                        "source": ["test"] * len(idx),
                    },
                    index=idx,
                )
                storage = Storage(config.db_path, config.ohlcv_table)
                storage.upsert(df)

                snapshot = _latest_feature_snapshot(config)
                self.assertIsNotNone(snapshot)
                assert snapshot is not None

                sup = make_supervised(df, candle_minutes=60, feature_windows_hours=config.feature_windows)
                self.assertLess(sup.index.max(), snapshot["anchor_ts"])
                self.assertEqual(snapshot["anchor_ts"], df.index.max().to_pydatetime())
                self.assertAlmostEqual(snapshot["anchor_price"], float(df.iloc[-1]["close"]))
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
