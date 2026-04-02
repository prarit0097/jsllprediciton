# JSLL Prediction Platform

Django-based prediction and monitoring app for Jeena Sikho Lifecare Limited (`JSLL.NS`), built around NSE session-aware data handling, tournament-driven model selection, and a live dashboard.

## What This Repo Contains

- live market price APIs and dashboard views
- tournament pipeline for horizon-wise model selection
- prediction generation with confidence, uncertainty bands, and match tracking
- drift checks, data-quality checks, and repair flows
- SQLite-backed storage for runtime predictions, scoreboards, and OHLCV history

## Main Modules

- `btcsite/`: Django settings, WSGI, URL routing
- `jeena_sikho_dashboard/`: dashboard page, JSON APIs, prediction/tournament service layer, runtime DB helpers
- `jeena_sikho_tournament/`: feature engineering, training, validation, drift logic, repair jobs, market calendar logic
- `price/`: older simple price page/API kept alongside the richer dashboard
- `data/`: runtime artifacts in deployed environments

## Routes

- `/`: main JSLL dashboard
- `/price/`: legacy/simple price page
- `/api/price`: simple price API
- `/api/jeena-sikho/price`: dashboard price API
- `/api/jeena-sikho/price_at?ts=...`: aligned historic price lookup
- `/api/jeena-sikho/tournament/summary`: latest run summary, champions, drift, completeness
- `/api/jeena-sikho/tournament/scoreboard`: scoreboard rows
- `/api/jeena-sikho/tournament/run`: manual tournament trigger (`POST`)
- `/api/jeena-sikho/tournament/run/status`: async run status
- `/api/jeena-sikho/prediction/latest`: latest ready predictions
- `/api/jeena-sikho/prediction/refresh`: on-demand prediction refresh (`POST`)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Open `http://127.0.0.1:8000/`.

## Runtime Commands

Tournament scheduler entry:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho.run_hourly
```

Repair job:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho.run_repair
```

Optional force-run:

```powershell
$env:FORCE_RUN='1'
.\.venv\Scripts\python.exe -m jeena_sikho.run_hourly
```

Doctor / smoke checks:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho_tournament.doctor
.\.venv\Scripts\python.exe manage.py check
```

## Testing

The repo has standalone tests under `tests/`. They are not discovered by `manage.py test`.

Typical run:

```powershell
python -m pytest tests -q
```

If `pytest` is not installed in the environment, install it first:

```powershell
pip install pytest
```

## Important Config

See `.env.example` for the full surface. The most important variables are:

- `APP_API_PREFIX=/api/jeena-sikho`
- `APP_BRAND_NAME`, `APP_MARKET_LABEL`
- `MARKET_SYMBOL=JSLL/INR`
- `MARKET_YFINANCE_SYMBOL=JSLL.NS`
- `PRICE_SOURCE=yfinance`
- `MARKET_TIMEFRAMES=1h,2h,1d`
- `RUN_MODE=daily`
- `AUTO_RETRAIN_ON_DRIFT=1`
- `STRICT_DATA_QUALITY=1`
- `LOW_CONFIDENCE_PCT`, `LOW_CONFIDENCE_SKIP_PCT`

## Current Production Shape

As currently deployed by the maintainer, this app is served behind a path prefix:

- public URL: `https://api.seestox.com/jsll/`
- static URL: `https://api.seestox.com/jsll/static/`

The live VPS deployment uses nginx path rewriting in front of Gunicorn, so backend routes still run as root Django paths while users access them under `/jsll/`.

## Deep Project Notes

For the detailed project analysis, deployment notes, file-by-file explanation, and command reference, see [`jsll.md`](./jsll.md).
