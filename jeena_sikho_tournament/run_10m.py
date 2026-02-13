from pathlib import Path
from .config import TournamentConfig
from .tournament import run_tournament
from .env import load_env


def main():
    load_env()
    config = TournamentConfig()
    config.base_dir = Path(".")
    run_tournament(config)


if __name__ == "__main__":
    main()
