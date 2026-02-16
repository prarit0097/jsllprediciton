from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from .config import TournamentConfig
from .diagnostics import run_doctor
from .multi_timeframe import run_multi_timeframe_tournament


def run_weekly_reoptimization(config: TournamentConfig) -> dict:
    prev_force = os.getenv("FORCE_RUN")
    prev_mode = os.getenv("RUN_MODE")
    prev_total = os.getenv("MAX_CANDIDATES_TOTAL")
    prev_per = os.getenv("MAX_CANDIDATES_PER_TARGET")
    try:
        os.environ["FORCE_RUN"] = "1"
        os.environ["RUN_MODE"] = os.getenv("WEEKLY_REOPT_RUN_MODE", "all")
        os.environ["MAX_CANDIDATES_TOTAL"] = os.getenv("WEEKLY_REOPT_MAX_CANDIDATES_TOTAL", "320")
        os.environ["MAX_CANDIDATES_PER_TARGET"] = os.getenv("WEEKLY_REOPT_MAX_CANDIDATES_PER_TARGET", "140")
        diag_code = run_doctor(Path("."))
        results = run_multi_timeframe_tournament(config)
        return {
            "weekly_reopt_at": datetime.now(timezone.utc).isoformat(),
            "diagnostics_ok": bool(diag_code == 0),
            "diag_code": int(diag_code),
            "results": results,
        }
    finally:
        if prev_force is None:
            os.environ.pop("FORCE_RUN", None)
        else:
            os.environ["FORCE_RUN"] = prev_force
        if prev_mode is None:
            os.environ.pop("RUN_MODE", None)
        else:
            os.environ["RUN_MODE"] = prev_mode
        if prev_total is None:
            os.environ.pop("MAX_CANDIDATES_TOTAL", None)
        else:
            os.environ["MAX_CANDIDATES_TOTAL"] = prev_total
        if prev_per is None:
            os.environ.pop("MAX_CANDIDATES_PER_TARGET", None)
        else:
            os.environ["MAX_CANDIDATES_PER_TARGET"] = prev_per


def main() -> None:
    config = TournamentConfig()
    report = run_weekly_reoptimization(config)
    print(report)


if __name__ == "__main__":
    main()
