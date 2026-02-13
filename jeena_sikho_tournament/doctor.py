import argparse
import sys
from pathlib import Path

from .diagnostics import run_doctor
from .env import load_env


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Jeena Sikho Tournament Doctor")
    parser.add_argument("--debug", action="store_true", help="Print tracebacks on failures")
    args = parser.parse_args()

    base = Path(".")
    try:
        code = run_doctor(base, debug=args.debug)
    except Exception:
        if args.debug:
            raise
        print("[FAIL] Doctor - unexpected error. Re-run with --debug for traceback.")
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
