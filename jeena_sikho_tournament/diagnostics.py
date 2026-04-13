import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TournamentConfig
from .features import compute_features, make_supervised
from .models_zoo import get_candidates
from .features import feature_sets
from .storage import Storage
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
        self.details: Dict[str, Any] = {}

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


def _capability_rank(name: str) -> int:
    order = {"baseline-only": 0, "core-ml": 1, "full-ensemble": 2}
    return order.get((name or "").strip().lower(), 0)


def _normalize_capability_name(name: Optional[str]) -> str:
    normalized = (name or "").strip().lower()
    if normalized in {"", "auto"}:
        return "baseline-only"
    if normalized not in {"baseline-only", "core-ml", "full-ensemble"}:
        return "baseline-only"
    return normalized


def _catboost_applicable() -> bool:
    return sys.version_info < (3, 14)


def inspect_runtime_capabilities(required_capability: Optional[str] = None) -> Dict[str, Any]:
    dependencies = {
        "sklearn": {"group": "core-ml", "required": True, "applicable": True, "installed": _has_module("sklearn")},
        "joblib": {"group": "core-ml", "required": True, "applicable": True, "installed": _has_module("joblib")},
        "xgboost": {"group": "ensemble", "required": False, "applicable": True, "installed": _has_module("xgboost")},
        "lightgbm": {"group": "ensemble", "required": False, "applicable": True, "installed": _has_module("lightgbm")},
        "catboost": {
            "group": "ensemble",
            "required": False,
            "applicable": _catboost_applicable(),
            "installed": _catboost_applicable() and _has_module("catboost"),
        },
        "ccxt": {"group": "data-provider", "required": False, "applicable": True, "installed": _has_module("ccxt")},
        "yfinance": {"group": "data-provider", "required": False, "applicable": True, "installed": _has_module("yfinance")},
    }

    missing_core_ml = [
        name
        for name, meta in dependencies.items()
        if meta["group"] == "core-ml" and meta["applicable"] and not meta["installed"]
    ]
    missing_ensemble = [
        name
        for name, meta in dependencies.items()
        if meta["group"] == "ensemble" and meta["applicable"] and not meta["installed"]
    ]
    missing_data_providers = [
        name
        for name, meta in dependencies.items()
        if meta["group"] == "data-provider" and meta["applicable"] and not meta["installed"]
    ]
    unavailable_unsupported = [
        name for name, meta in dependencies.items() if not meta["applicable"]
    ]
    available_ensemble = [
        name
        for name, meta in dependencies.items()
        if meta["group"] == "ensemble" and meta["applicable"] and meta["installed"]
    ]
    available_data_providers = [
        name
        for name, meta in dependencies.items()
        if meta["group"] == "data-provider" and meta["applicable"] and meta["installed"]
    ]

    if missing_core_ml:
        capability_status = "baseline-only"
        capability_reason = "Missing core ML runtime dependencies"
    elif missing_ensemble:
        capability_status = "core-ml"
        capability_reason = "Core ML is available, but ensemble boosters are incomplete"
    else:
        capability_status = "full-ensemble"
        capability_reason = "Core ML and ensemble boosters are available"

    if not available_data_providers:
        data_provider_status = "blocked"
    elif missing_data_providers:
        data_provider_status = "degraded"
    else:
        data_provider_status = "ready"

    if missing_core_ml:
        dependency_health = "blocked"
    elif missing_ensemble or data_provider_status != "ready":
        dependency_health = "degraded"
    else:
        dependency_health = "healthy"

    required_capability = _normalize_capability_name(
        required_capability or os.getenv("JSLL_REQUIRED_CAPABILITY") or os.getenv("REQUIRED_CAPABILITY_STATUS")
    )
    promotion_ready = (
        not missing_core_ml and _capability_rank(capability_status) >= _capability_rank(required_capability)
    )
    if promotion_ready:
        promotion_block_reason = ""
    else:
        promotion_block_reason = (
            f"Runtime capability {capability_status} does not satisfy required {required_capability}"
        )

    install_suggestions = []
    all_missing = missing_core_ml + missing_ensemble + missing_data_providers
    if all_missing:
        install_suggestions.append("pip install " + " ".join(all_missing))

    return {
        "capability_status": capability_status,
        "capability_reason": capability_reason,
        "dependency_health": dependency_health,
        "required_capability": required_capability,
        "promotion_ready": promotion_ready,
        "promotion_block_reason": promotion_block_reason,
        "data_provider_status": data_provider_status,
        "dependencies": dependencies,
        "missing_core_ml": missing_core_ml,
        "missing_ensemble": missing_ensemble,
        "missing_data_providers": missing_data_providers,
        "unsupported_dependencies": unavailable_unsupported,
        "available_ensemble": available_ensemble,
        "available_data_providers": available_data_providers,
        "install_suggestions": install_suggestions,
    }


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


def check_dependencies(required_capability: Optional[str] = None) -> Tuple[CheckResult, List[str], Dict[str, Any]]:
    res = CheckResult("Dependencies")
    missing_python = [name for name in ["pandas", "numpy"] if not _has_module(name)]
    capability = inspect_runtime_capabilities(required_capability=required_capability)
    res.details = capability

    if missing_python:
        res.fail(f"Missing required: {', '.join(missing_python)}")
    elif not capability["promotion_ready"]:
        res.fail(capability["promotion_block_reason"] or "Runtime capability requirement not satisfied")
    elif capability["dependency_health"] == "blocked":
        missing = capability["missing_core_ml"] + capability["missing_data_providers"]
        res.fail(f"Runtime blocked by missing dependencies: {', '.join(missing)}")
    elif capability["dependency_health"] == "degraded":
        missing = capability["missing_ensemble"] + capability["missing_data_providers"]
        res.warn(
            f"Capability {capability['capability_status']} with degraded dependencies: {', '.join(missing)}"
        )

    install_cmds = []
    if missing_python:
        install_cmds.append("pip install " + " ".join(missing_python))
    install_cmds.extend(capability["install_suggestions"])
    return res, install_cmds, capability


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
    missing = _missing_intervals(df.index, config.candle_minutes)

    summary["rows"] = str(len(df))
    summary["earliest"] = str(earliest)
    summary["latest"] = str(latest)
    summary["missing_intervals"] = str(missing)

    start = datetime.fromisoformat(config.start_date_utc).replace(tzinfo=timezone.utc)
    if earliest > start:
        res.warn("Earliest data is after 2015-01-01")

    if latest < _now_utc() - timedelta(hours=2):
        res.warn("Latest data is stale (>2h)")

    if df.index.duplicated().any():
        res.fail("Duplicate timestamps found")

    if not df.index.is_monotonic_increasing:
        res.fail("Timestamps not monotonic")

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        res.fail("NaNs in OHLCV fields")

    last_30 = df.loc[df.index >= _now_utc() - timedelta(days=30)]
    if not last_30.empty:
        expected = int((24 * 60 / max(1, config.candle_minutes)) * 30)
        completeness = 1 - (_missing_intervals(last_30.index, config.candle_minutes) / expected)
        summary["last_30d_completeness"] = f"{completeness:.2%}"
    return res, summary


def check_features(config: TournamentConfig) -> CheckResult:
    res = CheckResult("Features")
    try:
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
        orig = np.log(data["close"]).diff().shift(-1)
        aligned = orig.loc[sup.index]
        if not np.allclose(sup["y_ret"].values, aligned.values, equal_nan=False):
            res.fail("y_ret target misaligned")
        if not np.all((sup["y_dir"].values == (sup["y_ret"].values > 0).astype(int))):
            res.fail("y_dir target misaligned")

        feats = compute_features(
            data,
            candle_minutes=config.candle_minutes,
            feature_windows_hours=config.feature_windows,
        ).dropna()
        future = np.log(data["close"]).diff().shift(-1)
        warnings = []
        for col in feats.columns:
            if col in {"open", "high", "low", "close", "volume"}:
                continue
            series = feats[col]
            corr_future = series.corr(future.loc[series.index])
            corr_lag = series.corr(np.log(data["close"]).diff().loc[series.index])
            if abs(corr_future) > 0.2 and abs(corr_future) > abs(corr_lag) * 5.0:
                warnings.append(col)
        if warnings:
            res.warn(f"Potential leakage signals in: {', '.join(warnings[:5])}")
    except Exception as exc:
        res.fail(f"Feature check error: {exc}")
    return res


def check_tests() -> CheckResult:
    res = CheckResult("Tests")
    if not _has_module("pytest"):
        res.warn("pytest not installed")
        return res
    try:
        import subprocess

        cmd = [sys.executable, "-m", "pytest", "tests/test_leakage.py", "tests/test_splits.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
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


def build_doctor_report(base: Path, debug: bool = False, required_capability: Optional[str] = None) -> Dict[str, Any]:
    results = []

    results.append(check_structure(base))

    dep_res, install_cmds, capability_report = check_dependencies(required_capability=required_capability)
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

    overall = "PASS"
    for r in results:
        if r.status == "FAIL":
            overall = "FAIL"
            break
        if r.status == "WARN" and overall != "FAIL":
            overall = "WARN"

    next_actions = "none."
    if overall == "FAIL":
        next_actions = "fix the FAIL sections above and rerun doctor."
    elif overall == "WARN":
        next_actions = "consider fixing WARN items for best results."

    return {
        "overall": overall,
        "exit_code": 0 if overall == "PASS" else 1,
        "results": [
            {
                "name": r.name,
                "status": r.status,
                "message": r.message,
                "details": r.details,
            }
            for r in results
        ],
        "capability_status": capability_report["capability_status"],
        "capability_report": capability_report,
        "data_summary": data_summary,
        "model_zoo_summary": zoo_summary,
        "install_suggestions": install_cmds,
        "next_actions": next_actions,
        "debug": debug,
    }


def print_doctor_report(report: Dict[str, Any], json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return

    for r in report["results"]:
        if r["status"] == "PASS":
            print(f"[PASS] {r['name']}")
        elif r["status"] == "WARN":
            print(f"[WARN] {r['name']} - {r['message']}")
        else:
            print(f"[FAIL] {r['name']} - {r['message']}")

    capability = report["capability_report"]
    print("Runtime capability:")
    print(f"  capability_status: {capability['capability_status']}")
    print(f"  dependency_health: {capability['dependency_health']}")
    print(f"  required_capability: {capability['required_capability']}")
    print(f"  promotion_ready: {capability['promotion_ready']}")
    print(f"  data_provider_status: {capability['data_provider_status']}")
    if capability["missing_core_ml"]:
        print(f"  missing_core_ml: {', '.join(capability['missing_core_ml'])}")
    if capability["missing_ensemble"]:
        print(f"  missing_ensemble: {', '.join(capability['missing_ensemble'])}")
    if capability["missing_data_providers"]:
        print(f"  missing_data_providers: {', '.join(capability['missing_data_providers'])}")
    if capability["unsupported_dependencies"]:
        print(f"  unsupported_dependencies: {', '.join(capability['unsupported_dependencies'])}")

    if report["data_summary"]:
        print("Data summary:")
        for k, v in report["data_summary"].items():
            print(f"  {k}: {v}")

    if report["model_zoo_summary"]:
        print("Model zoo:")
        for k, v in report["model_zoo_summary"].items():
            print(f"  {k}: {v}")

    if report["install_suggestions"]:
        print("Install suggestions:")
        for cmd in report["install_suggestions"]:
            print(f"  {cmd}")

    print(f"Overall: {report['overall']}")
    print(f"Next actions: {report['next_actions']}")

def run_doctor(
    base: Path,
    debug: bool = False,
    required_capability: Optional[str] = None,
    json_output: bool = False,
) -> int:
    report = build_doctor_report(base, debug=debug, required_capability=required_capability)
    print_doctor_report(report, json_output=json_output)
    return report["exit_code"]

