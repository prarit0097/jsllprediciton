from datetime import date
import os
import sqlite3
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.repair import repair_timeframe_data
from jeena_sikho_tournament.storage import Storage
from jeena_sikho_tournament.validator import validate_ohlcv_quality


def _sample_df(start: str, periods: int, freq: str = "60min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    base = np.linspace(100, 102, periods)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.1,
            "volume": np.full(periods, 1000.0),
        },
        index=idx,
    )


def test_validator_detects_duplicate_timestamp():
    df = _sample_df("2026-02-10 03:45:00", 5)
    dup = pd.concat([df, df.iloc[-1:]]).sort_index()
    report = validate_ohlcv_quality(dup, 60, nse_mode=False, holidays=set(), max_missing_ratio=1.0)
    assert report.ok is False
    assert any("duplicate_timestamps" in e for e in report.errors)


def test_validator_nse_boundary_violation():
    # 08:00 IST is outside NSE session
    idx = pd.DatetimeIndex([pd.Timestamp("2026-02-10 02:30:00+00:00")])
    df = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000.0]},
        index=idx,
    )
    report = validate_ohlcv_quality(df, 60, nse_mode=True, holidays=set(), max_missing_ratio=1.0)
    assert report.ok is False
    assert any("nse_session_boundary_violation" in e for e in report.errors)


class RepairSanitizationTests(unittest.TestCase):
    def test_repair_removes_existing_invalid_nse_session_rows(self):
        tmp = tempfile.mkdtemp()
        old_env = {
            "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
            "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
            "MARKET_YFINANCE_SYMBOL": os.environ.get("MARKET_YFINANCE_SYMBOL"),
        }
        try:
            os.environ["APP_DATA_DIR"] = tmp
            os.environ["APP_MARKET_DB_FILE"] = "repair.sqlite3"
            os.environ["MARKET_YFINANCE_SYMBOL"] = "JSLL.NS"

            cfg = TournamentConfig()
            storage = Storage(cfg.db_path, cfg.ohlcv_table)
            storage.init_db()

            valid_ts = pd.Timestamp("2026-02-10T03:45:00+00:00")
            invalid_ts = pd.Timestamp("2026-02-10T02:30:00+00:00")

            storage.upsert(
                pd.DataFrame(
                    {
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.5],
                        "volume": [1000.0],
                        "source": ["seed"],
                    },
                    index=pd.DatetimeIndex([valid_ts], tz="UTC"),
                )
            )

            with sqlite3.connect(cfg.db_path) as con:
                con.execute(
                    f"""
                    INSERT OR REPLACE INTO {cfg.ohlcv_table}
                    (timestamp_utc, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (invalid_ts.isoformat(), 99.0, 100.0, 98.0, 99.5, 900.0, "legacy_bad"),
                )

            with patch("jeena_sikho_tournament.repair.fetch_and_stitch", return_value=(pd.DataFrame(), None)):
                report = repair_timeframe_data(cfg, lookback_days=120)

            repaired = storage.load()
            self.assertNotIn(invalid_ts, repaired.index)
            self.assertIn(valid_ts, repaired.index)
            self.assertEqual(report["purged_invalid_session_rows"], 1)
            self.assertNotIn("nse_session_boundary_violation", report["dq_errors"])
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            shutil.rmtree(tmp, ignore_errors=True)

    def test_repair_removes_existing_holiday_rows(self):
        tmp = tempfile.mkdtemp()
        old_env = {
            "APP_DATA_DIR": os.environ.get("APP_DATA_DIR"),
            "APP_MARKET_DB_FILE": os.environ.get("APP_MARKET_DB_FILE"),
            "MARKET_YFINANCE_SYMBOL": os.environ.get("MARKET_YFINANCE_SYMBOL"),
        }
        try:
            os.environ["APP_DATA_DIR"] = tmp
            os.environ["APP_MARKET_DB_FILE"] = "repair.sqlite3"
            os.environ["MARKET_YFINANCE_SYMBOL"] = "JSLL.NS"

            cfg = TournamentConfig()
            storage = Storage(cfg.db_path, cfg.ohlcv_table)
            storage.init_db()

            valid_ts = pd.Timestamp("2026-04-13T09:45:00+00:00")
            holiday_ts = pd.Timestamp("2026-04-14T09:45:00+00:00")

            storage.upsert(
                pd.DataFrame(
                    {
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.5],
                        "volume": [1000.0],
                        "source": ["seed"],
                    },
                    index=pd.DatetimeIndex([valid_ts], tz="UTC"),
                )
            )

            with sqlite3.connect(cfg.db_path) as con:
                con.execute(
                    f"""
                    INSERT OR REPLACE INTO {cfg.ohlcv_table}
                    (timestamp_utc, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (holiday_ts.isoformat(), 99.0, 100.0, 98.0, 99.5, 900.0, "legacy_holiday"),
                )

            with patch("jeena_sikho_tournament.repair.fetch_and_stitch", return_value=(pd.DataFrame(), None)):
                report = repair_timeframe_data(cfg, lookback_days=120)

            repaired = storage.load()
            self.assertNotIn(holiday_ts, repaired.index)
            self.assertIn(valid_ts, repaired.index)
            self.assertEqual(report["purged_invalid_session_rows"], 1)
            self.assertNotIn("nse_session_boundary_violation", report["dq_errors"])
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            shutil.rmtree(tmp, ignore_errors=True)
