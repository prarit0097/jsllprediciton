import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from jeena_sikho_tournament.exogenous import PublicSignalSpec, load_public_signal_frame
from jeena_sikho_tournament.features import compute_features, make_supervised


def _fake_yfinance(symbol, start, end, auto_adjust=False, source_name=None):  # noqa: ARG001
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    idx = pd.date_range(start=start_ts, end=end_ts, freq="60min", tz="UTC")
    if idx.empty:
        return pd.DataFrame()

    base_map = {
        "^NSEI": 24000.0,
        "^INDIAVIX": 12.0,
        "INR=X": 83.0,
        "^CNXAUTO": 18000.0,
    }
    slope_map = {
        "^NSEI": 4.0,
        "^INDIAVIX": 0.02,
        "INR=X": 0.005,
        "^CNXAUTO": 3.0,
    }
    base = base_map.get(symbol, 100.0)
    slope = slope_map.get(symbol, 0.5)
    steps = np.arange(len(idx), dtype=float)
    close = base + (steps * slope) + np.sin(steps / 6.0)
    out = pd.DataFrame(
        {
            "timestamp_utc": idx,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(len(idx), 1000.0),
            "source": source_name or "fake_yf",
        }
    )
    return out


class ExogenousFeatureTests(unittest.TestCase):
    def test_public_signal_frame_caches_and_reports_freshness(self):
        idx = pd.date_range("2026-04-01", periods=72, freq="60min", tz="UTC")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"APP_DATA_DIR": tmp}, clear=False):
                spec = PublicSignalSpec(alias="nifty", symbol="^NSEI", market="nse")
                with patch("jeena_sikho_tournament.exogenous.fetch_yfinance", side_effect=_fake_yfinance):
                    frame, meta = load_public_signal_frame(spec, idx, candle_minutes=60)

                self.assertFalse(frame.empty)
                self.assertEqual(meta["cache_status"], "refreshed")
                self.assertTrue(Path(meta["cache_path"]).exists())
                self.assertTrue(Path(meta["meta_path"]).exists())
                self.assertIn("latest_timestamp", meta)

                with patch("jeena_sikho_tournament.exogenous.fetch_yfinance", return_value=pd.DataFrame()):
                    cached_frame, cached_meta = load_public_signal_frame(spec, idx, candle_minutes=60)

                self.assertFalse(cached_frame.empty)
                self.assertEqual(cached_meta["cache_status"], "cache_hit")

    def test_feature_pipeline_adds_exogenous_and_event_columns(self):
        idx = pd.date_range("2026-03-01", periods=24 * 25, freq="60min", tz="UTC")
        base = 680.0 + np.cumsum(np.linspace(-0.4, 0.6, len(idx)))
        price_df = pd.DataFrame(
            {
                "open": base - 0.2,
                "high": base + 0.5,
                "low": base - 0.5,
                "close": base,
                "volume": np.full(len(idx), 1200.0),
            },
            index=idx,
        )

        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "events.csv"
            pd.DataFrame(
                [
                    {"event_date": "2026-03-10", "symbol": "JSLL.NS", "severity": 2.0},
                    {"event_date": "2026-03-18", "symbol": "OTHER.NS", "severity": 1.0},
                ]
            ).to_csv(event_path, index=False)

            env = {
                "APP_DATA_DIR": tmp,
                "MARKET_YFINANCE_SYMBOL": "JSLL.NS",
                "EXOGENOUS_FEEDS_ENABLE": "1",
                "EXOGENOUS_SECTOR_SYMBOL": "^CNXAUTO",
                "EVENT_FEATURES_ENABLE": "1",
                "EVENT_CALENDAR_FILE": str(event_path),
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("jeena_sikho_tournament.exogenous.fetch_yfinance", side_effect=_fake_yfinance):
                    feats = compute_features(price_df, candle_minutes=60)
                    sup = make_supervised(price_df, candle_minutes=60)

            for col in [
                "exo_nifty_ret_1c",
                "exo_vix_high_regime",
                "exo_usdinr_ret_24h",
                "rel_nifty_strength_1c",
                "rel_sector_strength_24h",
                "fx_pressure_1c",
                "has_known_event",
                "days_to_known_event",
                "known_event_window",
            ]:
                self.assertIn(col, feats.columns)

            self.assertIn("exogenous_metadata", feats.attrs)
            metadata = feats.attrs["exogenous_metadata"]
            self.assertEqual(sorted(metadata["signals"].keys()), ["nifty", "sector", "usdinr", "vix"])
            self.assertTrue(metadata["event_calendar"]["available"])
            self.assertGreater(metadata["event_calendar"]["matched_days"], 0)
            self.assertEqual(int(sup["is_event_day"].max()), 1)

            meta_path = Path(metadata["signals"]["nifty"]["meta_path"])
            self.assertTrue(meta_path.exists())
            stored_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_meta["alias"], "nifty")


if __name__ == "__main__":
    unittest.main()
