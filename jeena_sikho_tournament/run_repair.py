from pathlib import Path
import json

from .config import TournamentConfig
from .env import load_env
from .repair import run_nightly_repair


def main() -> None:
    load_env()
    cfg = TournamentConfig()
    cfg.base_dir = Path(".")
    report = run_nightly_repair(cfg)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
