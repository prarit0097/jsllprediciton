import argparse
import sys
from pathlib import Path

from .diagnostics import run_doctor
from .env import load_env


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Jeena Sikho Tournament Doctor")
    parser.add_argument("--debug", action="store_true", help="Print tracebacks on failures")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON report")
    parser.add_argument(
        "--require-capability",
        choices=["baseline-only", "core-ml", "full-ensemble"],
        help="Fail doctor if runtime capability is below this threshold",
    )
    args = parser.parse_args()

    base = Path(".")
    try:
        code = run_doctor(
            base,
            debug=args.debug,
            required_capability=args.require_capability,
            json_output=args.json,
        )
    except Exception:
        if args.debug:
            raise
        print("[FAIL] Doctor - unexpected error. Re-run with --debug for traceback.")
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
