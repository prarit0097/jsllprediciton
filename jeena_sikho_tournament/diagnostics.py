import importlib.util
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import TournamentConfig
from .features import compute_features, make_supervised
from .models_zoo import get_candidates
from .features import feature_sets
from .market_calendar import IST, NSE_CLOSE_MIN, load_nse_holidays, is_nse_market_open, is_nse_trading_day
from .storage import Storage
from .validator import validate_ohlcv_quality
from .registry import load_registry
from .predict import predict_latest
from .sample_run import run_dry_tournament

REQUIRED_FILES = [
    "jeena_sikho_tournament/config.py",
    "jeena_sikho_tournament/storage.py",
    "jeena_sikho_tournament/features.py",
    "jeena_sikho_tournament/models_zoo.py",
    "jeena_sikho_tournament/tournament.py",
    "jeena_sikho_tournament/registry.py",
    "jeena_sikho_tournament/run_hourly.py",
    "jeena_sikho_tournament/predict.py",
    "jeena_sikho_tournament/README.md",
]


class CheckResult:
    def __init__(self, name: str):
        self.name = name
        self.status = "PASS"
        self.message = ""

    def warn(self, msg: str):
        if self.status != "FAIL":
            self.status = "WARN"
            self.message = msg

    def fail(self, msg: str):
        self.status = "FAIL"
        self.message = msg


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_structure(base: Path) -> CheckResult:
    res = CheckResult("Structure")
    missing = []
    for rel in REQUIRED_FILES:
        path = base / rel
        if not path.exists():
            missing.append(rel)
    data_fetcher = base / "jeena_sikho_tournament" / "data_fetcher.py"
    data_sources = base / "jeena_sikho_tournament" / "data_sources.py"
    if not data_fetcher.exists() and not data_sources.exists():
        missing.append("jeena_sikho_tournament/data_fetcher.py OR jeena_sikho_tournament/data_sources.py")
    if missing:
        res.fail(f"Missing files: {', '.join(missing)}")
    return res


def check_dependencies() -> Tuple[CheckResult, List[str]]:
    res = CheckResult("Dependencies")
    missing = []
    warns = []

    required = ["pandas", "numpy", "sklearn", "joblib"]
    recommended = ["ccxt", "yfinance"]
    optional = ["lightgbm", "xgboost"]
    # catboost wheels are often unavailable on newest Python releases (e.g. 3.14),
    # so avoid noisy install suggestions when unsupported.
    if sys.version_info < (3, 14):
        optional.append("catboost")

    for r in required:
        if not _has_module(r):
            missing.append(r)
    for r in recommended:
        if not _has_module(r):
            warns.append(r)
    # Optional boosters should not trigger WARN for overall health
    missing_optional = [r for r in optional if not _has_module(r)]

    if missing:
        res.fail(f"Missing required: {', '.join(missing)}")
    elif warns:
        res.warn(f"Missing recommended: {', '.join(warns)}")

    install_cmds = []
    if missing:
        install_cmds.append("pip install " + " ".join(missing))
    if warns:
        install_cmds.append("pip install " + " ".join(warns))
    if missing_optional:
        install_cmds.append("pip install " + " ".join(missing_optional))
    return res, install_cmds


def check_storage(config: TournamentConfig) -> CheckResult:
    res = CheckResult("Storage")
    try:
        storage = Storage(config.db_path, config.ohlcv_table)
        storage.init_db()
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
                "source": ["doctor"],
            },
            index=[pd.Timestamp("2015-01-01 00:00:00", tz="UTC")],
        )
        storage.upsert(df)
        loaded = storage.load()
        if loaded.empty:
            res.fail("Storage write/read failed")
    except Exception as exc:
        res.fail(f"Storage error: {exc}")
    return res


def _missing_intervals(series: pd.DatetimeIndex, candle_minutes: int) -> int:
    if series.empty:
        return 0
    freq = f"{candle_minutes}min"
    full = pd.date_range(start=series.min(), end=series.max(), freq=freq, tz="UTC")
    return len(full.difference(series))


def _is_nse_mode(config: TournamentConfig) -> bool:
    sym = (config.yfinance_symbol or "").strip().upper()
    return sym.endswith(".NS") or sym.endswith(".BO")


def check_data(config: TournamentConfig) -> Tuple[CheckResult, Dict[str, str]]:
    res = CheckResult("Data")
    summary = {}
    storage = Storage(config.db_path, config.ohlcv_table)
    df = storage.load()
    if df.empty:
        res.warn("No cached data available")
        return res, summary

    df = df.sort_index()
    earliest = df.index.min()
    latest = df.index.max()
    nse_mode = _is_nse_mode(config)
    if nse_mode:
        dq_full = validate_ohlcv_quality(
            df,
            config.candle_minutes,
            nse_mode=True,
            holidays=load_nse_holidays(config.data_dir),
            max_missing_ratio=float(os.getenv("MAX_MISSING_RATIO", "0.15")),
        )
        missing = int((dq_full.stats or {}).get("missing_intervals", 0))
    else:
        missing = _missing_intervals(df.index, config.candle_minutes)

    summary["rows"] = str(len(df))
    summary["earliest"] = str(earliest)
    summary["latest"] = str(latest)
    summary["missing_intervals"] = str(missing)

    start = datetime.fromisoformat(config.start_date_utc).replace(tzinfo=timezone.utc)
    if earliest > start:
        res.warn("Earliest data is after 2015-01-01")

    now_utc = _now_utc()
    if latest < now_utc - timedelta(hours=2):
        stale = True
        if nse_mode:
            holidays = load_nse_holidays(config.data_dir)
            now_ist = now_utc.astimezone(IST)
            latest_ist = latest.tz_convert(IST)
            if not is_nse_market_open(now_utc, holidays):
                # Outside market hours, compare against the most recent trading-day expected close slot.
                recent_day = now_ist.date()
                probe = now_ist
                if not is_nse_trading_day(probe, holidays):
                    while not is_nse_trading_day(probe, holidays):
                        probe = (probe - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
                    recent_day = probe.date()
                elif (now_ist.hour * 60 + now_ist.minute) < (9 * 60 + 15):
                    probe = (now_ist - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
                    while not is_nse_trading_day(probe, holidays):
                        probe = (probe - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
                    recent_day = probe.date()

                close_min = NSE_CLOSE_MIN
                if config.candle_minutes >= 1440:
                    exp_min = close_min
                else:
                    open_min = 9 * 60 + 15
                    span = close_min - open_min
                    exp_min = open_min + (span // max(1, config.candle_minutes)) * max(1, config.candle_minutes)
                exp_h, exp_m = divmod(exp_min, 60)
                expected_slot = datetime(
                    recent_day.year,
                    recent_day.month,
                    recent_day.day,
                    exp_h,
                    exp_m,
                    tzinfo=IST,
                )
                # Allow one interval grace for delayed inserts.
                grace = timedelta(minutes=max(60, int(config.candle_minutes)))
                if latest_ist >= (expected_slot - grace):
                    stale = False
        if stale:
            res.warn("Latest data is stale (>2h)")

    if df.index.duplicated().any():
        res.fail("Duplicate timestamps found")

    if not df.index.is_monotonic_increasing:
        res.fail("Timestamps not monotonic")

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        res.fail("NaNs in OHLCV fields")

    last_30 = df.loc[df.index >= _now_utc() - timedelta(days=30)]
    if not last_30.empty:
        if nse_mode:
            dq_30 = validate_ohlcv_quality(
                last_30,
                config.candle_minutes,
                nse_mode=True,
                holidays=load_nse_holidays(config.data_dir),
                max_missing_ratio=float(os.getenv("MAX_MISSING_RATIO", "0.15")),
            )
            miss_ratio = float((dq_30.stats or {}).get("missing_ratio", 1.0))
            completeness = max(0.0, min(1.0, 1.0 - miss_ratio))
        else:
            expected = int((24 * 60 / max(1, config.candle_minutes)) * 30)
            completeness = 1 - (_missing_intervals(last_30.index, config.candle_minutes) / expected)
        summary["last_30d_completeness"] = f"{completeness:.2%}"
    return res, summary


def check_features(config: TournamentConfig) -> CheckResult:
    res = CheckResult("Features")
    try:
        prev_exo_req = os.getenv("EXOGENOUS_REQUIRED")
        os.environ["EXOGENOUS_REQUIRED"] = "0"
        freq = f"{config.candle_minutes}min"
        idx = pd.date_range("2024-01-01", periods=300, freq=freq, tz="UTC")
        rng = np.random.default_rng(1)
        data = pd.DataFrame(
            {
                "open": rng.normal(100, 1, size=len(idx)),
                "high": rng.normal(101, 1, size=len(idx)),
                "low": rng.normal(99, 1, size=len(idx)),
                "close": rng.normal(100, 1, size=len(idx)),
                "volume": rng.uniform(100, 200, size=len(idx)),
            },
            index=idx,
        )
        sup = make_supervised(data, candle_minutes=config.candle_minutes, feature_windows_hours=config.feature_windows)
        required = {"y_ret_raw", "y_ret_model", "y_ret", "y_dir", "target_scale"}
        if not required.issubset(set(sup.columns)):
            missing = sorted(list(required.difference(set(sup.columns))))
            res.fail(f"Missing supervised target columns: {', '.join(missing)}")
            return res

        if not np.all((sup["y_dir"].values == (sup["y_ret_raw"].values > 0).astype(int))):
            res.fail("y_dir target misaligned with y_ret_raw")
            return res

        if not np.allclose(sup["y_ret"].values, sup["y_ret_raw"].values, atol=1e-12, rtol=1e-9):
            res.fail("y_ret must mirror y_ret_raw")
            return res

        mode = os.getenv("RETURN_TARGET_MODE", "volnorm_logret").strip().lower()
        if mode in {"volnorm", "volnorm_logret", "normalized"}:
            rec = sup["y_ret_model"].to_numpy() * sup["target_scale"].to_numpy()
            if not np.allclose(rec, sup["y_ret_raw"].to_numpy(), atol=1e-6, rtol=1e-3):
                res.fail("y_ret_model * target_scale mismatch with y_ret_raw")
                return res
        else:
            if not np.allclose(sup["y_ret_model"].values, sup["y_ret_raw"].values, atol=1e-12, rtol=1e-9):
                res.fail("y_ret_model should match y_ret_raw in raw target mode")
                return res

        feats = compute_features(
            data,
            candle_minutes=config.candle_minutes,
            feature_windows_hours=config.feature_windows,
        ).dropna()
        close_ret = np.log(data["close"]).diff()
        future = close_ret.shift(-1)
        warnings = []
        for col in feats.columns:
            if col in {"open", "high", "low", "close", "volume"}:
                continue
            aligned = pd.DataFrame(
                {
                    "feat": feats[col],
                    "future": future.reindex(feats.index),
                    "lag": close_ret.reindex(feats.index),
                }
            ).dropna()
            if len(aligned) < 80:
                continue
            if float(aligned["feat"].std()) <= 1e-9:
                continue
            if float(aligned["future"].std()) <= 1e-12:
                continue
            if float(aligned["lag"].std()) <= 1e-12:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                corr_future = float(aligned["feat"].corr(aligned["future"]))
                corr_lag = float(aligned["feat"].corr(aligned["lag"]))
            if np.isnan(corr_future) or np.isnan(corr_lag):
                continue
            lag_scale = max(0.05, abs(corr_lag))
            # Treat as suspicious only when future-link is unusually strong.
            suspicious = abs(corr_future) > 0.70 and abs(corr_future) > (lag_scale * 3.0)
            if suspicious:
                warnings.append(col)
        if warnings:
            res.warn(f"Potential leakage signals in: {', '.join(warnings[:5])}")
        if prev_exo_req is None:
            os.environ.pop("EXOGENOUS_REQUIRED", None)
        else:
            os.environ["EXOGENOUS_REQUIRED"] = prev_exo_req
    except Exception as exc:
        res.fail(f"Feature check error: {exc}")
        try:
            if prev_exo_req is None:
                os.environ.pop("EXOGENOUS_REQUIRED", None)
            else:
                os.environ["EXOGENOUS_REQUIRED"] = prev_exo_req
        except Exception:
            pass
    return res


def check_tests() -> CheckResult:
    res = CheckResult("Tests")
    if not _has_module("pytest"):
        res.warn("pytest not installed")
        return res
    try:
        import subprocess

        prev_exo_req = os.getenv("EXOGENOUS_REQUIRED")
        os.environ["EXOGENOUS_REQUIRED"] = "0"
        cmd = [sys.executable, "-m", "pytest", "tests/test_leakage.py", "tests/test_splits.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if prev_exo_req is None:
            os.environ.pop("EXOGENOUS_REQUIRED", None)
        else:
            os.environ["EXOGENOUS_REQUIRED"] = prev_exo_req
        if result.returncode != 0:
            res.fail("pytest failed")
    except Exception as exc:
        res.warn(f"pytest error: {exc}")
    return res


def check_model_zoo(config: TournamentConfig) -> Tuple[CheckResult, Dict[str, str]]:
    res = CheckResult("Model Zoo")
    summary = {}
    min_candidates = 50
    storage = Storage(config.db_path, config.ohlcv_table)
    df = storage.load()
    if df.empty:
        fs_map = {"minimal": ["ret_1c", "ret_1h", "ret_4h", "ret_24h"]}
    else:
        sup = make_supervised(df, candle_minutes=config.candle_minutes, feature_windows_hours=config.feature_windows)
        fs_map = feature_sets(sup)
        if not fs_map:
            fs_map = {"minimal": ["ret_1c", "ret_1h", "ret_4h", "ret_24h"]}

    total = 0
    for task in ["direction", "return", "range"]:
        specs = get_candidates(
            task,
            config.max_candidates_per_target,
            config.enable_dl,
            candle_minutes=config.candle_minutes,
            strict_horizon_pool=True,
        )
        count = len(specs) * max(1, len(fs_map))
        summary[f"{task}_candidates"] = str(count)
        total += count
    summary["total_candidates"] = str(total)
    if total < min_candidates:
        res.fail(f"Too few candidates: {total}")
    return res, summary


def check_dry_run(config: TournamentConfig) -> CheckResult:
    res = CheckResult("Dry Run")
    try:
        ok = run_dry_tournament(config)
        if not ok:
            res.fail("Dry run did not complete")
    except Exception as exc:
        res.fail(f"Dry run error: {exc}")
    return res


def check_registry_predictor(config: TournamentConfig) -> CheckResult:
    res = CheckResult("Registry + Predictor")
    try:
        registry = load_registry(config.registry_path)
        champs = registry.get("champions", {})
        if not champs:
            # Fresh setup may legitimately have no champions yet.
            res.message = "No champions yet (fresh setup)"
            return res
        outputs = predict_latest(config)
        if not outputs:
            res.fail("Predictor returned no output")
    except Exception as exc:
        res.fail(f"Predictor error: {exc}")
    return res


def run_doctor(base: Path, debug: bool = False) -> int:
    results = []
    prev_exo_enable = os.getenv("EXOGENOUS_ENABLE")
    prev_exo_required = os.getenv("EXOGENOUS_REQUIRED")
    os.environ["EXOGENOUS_ENABLE"] = "0"
    os.environ["EXOGENOUS_REQUIRED"] = "0"

    try:
        results.append(check_structure(base))

        dep_res, install_cmds = check_dependencies()
        results.append(dep_res)

        config = TournamentConfig()

        results.append(check_storage(config))

        data_res, data_summary = check_data(config)
        results.append(data_res)

        results.append(check_features(config))
        results.append(check_tests())

        zoo_res, zoo_summary = check_model_zoo(config)
        results.append(zoo_res)

        results.append(check_dry_run(config))
        results.append(check_registry_predictor(config))
    finally:
        if prev_exo_enable is None:
            os.environ.pop("EXOGENOUS_ENABLE", None)
        else:
            os.environ["EXOGENOUS_ENABLE"] = prev_exo_enable
        if prev_exo_required is None:
            os.environ.pop("EXOGENOUS_REQUIRED", None)
        else:
            os.environ["EXOGENOUS_REQUIRED"] = prev_exo_required

    overall = "PASS"
    for r in results:
        if r.status == "FAIL":
            overall = "FAIL"
            break
        if r.status == "WARN" and overall != "FAIL":
            overall = "WARN"

    for r in results:
        if r.status == "PASS":
            print(f"[PASS] {r.name}")
        elif r.status == "WARN":
            print(f"[WARN] {r.name} - {r.message}")
        else:
            print(f"[FAIL] {r.name} - {r.message}")

    if data_summary:
        print("Data summary:")
        for k, v in data_summary.items():
            print(f"  {k}: {v}")

    if zoo_summary:
        print("Model zoo:")
        for k, v in zoo_summary.items():
            print(f"  {k}: {v}")

    if install_cmds:
        print("Install suggestions:")
        for cmd in install_cmds:
            print(f"  {cmd}")

    print(f"Overall: {overall}")
    if overall == "FAIL":
        print("Next actions: fix the FAIL sections above and rerun doctor.")
    elif overall == "WARN":
        print("Next actions: consider fixing WARN items for best results.")
    else:
        print("Next actions: none.")

    return 0 if overall == "PASS" else 1

