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
- `BUSINESS_PRIMARY_TIMEFRAME=1d`
- `RUN_MODE=daily`
- `AUTO_RETRAIN_ON_DRIFT=1`
- `STRICT_DATA_QUALITY=1`
- `LOW_CONFIDENCE_PCT`, `LOW_CONFIDENCE_SKIP_PCT`
- `EXOGENOUS_FEEDS_ENABLE=1`, `EXOGENOUS_NIFTY_SYMBOL`, `EXOGENOUS_VIX_SYMBOL`, `EXOGENOUS_USDINR_SYMBOL`
- `EVENT_FEATURES_ENABLE=1` with `EVENT_CALENDAR_FILE=...` when you have a JSLL event calendar

## Current Production Shape

As currently deployed by the maintainer, this app is served behind a path prefix:

- public URL: `https://api.seestox.com/jsll/`
- static URL: `https://api.seestox.com/jsll/static/`

The live VPS deployment uses nginx path rewriting in front of Gunicorn, so backend routes still run as root Django paths while users access them under `/jsll/`.

## VPS Runbook

Live production facts:

- code path: `/opt/jsll/app`
- venv path: `/opt/jsll/venv`
- public URL: `https://api.seestox.com/jsll/`
- static URL: `https://api.seestox.com/jsll/static/`
- Gunicorn service: `jsll-gunicorn.service`
- scheduler service: `jsll-scheduler.service`
- socket: `/run/jsll/jsll.sock`

Safe deploy / update:

```bash
cd /opt/jsll/app
source /opt/jsll/venv/bin/activate
git fetch origin
git reset --hard origin/main
python manage.py collectstatic --noinput
python manage.py check
sudo systemctl restart jsll-gunicorn
sudo systemctl restart jsll-scheduler
```

Full data refresh:

```bash
cd /opt/jsll/app
source /opt/jsll/venv/bin/activate

python -m jeena_sikho_tournament.run_repair

curl -X POST https://api.seestox.com/jsll/api/jeena-sikho/tournament/run \
  -H "Content-Type: application/json" \
  -d '{}'

sleep 10

curl https://api.seestox.com/jsll/api/jeena-sikho/tournament/run/status
curl -X POST https://api.seestox.com/jsll/api/jeena-sikho/prediction/refresh
curl https://api.seestox.com/jsll/api/jeena-sikho/prediction/latest
curl https://api.seestox.com/jsll/api/jeena-sikho/tournament/summary
```

Health checks:

```bash
sudo systemctl status jsll-gunicorn --no-pager
sudo systemctl status jsll-scheduler --no-pager
curl https://api.seestox.com/jsll/api/jeena-sikho/price
curl https://api.seestox.com/jsll/api/jeena-sikho/prediction/latest
curl https://api.seestox.com/jsll/api/jeena-sikho/tournament/summary
```

Logs:

```bash
sudo journalctl -u jsll-gunicorn -n 50 --no-pager -l
sudo journalctl -u jsll-scheduler -n 80 --no-pager -l
sudo journalctl -u jsll-scheduler --since "10 minutes ago" --no-pager -l
```

Important `.env` notes:

```env
APP_BASE_PREFIX=/jsll
APP_API_PREFIX=/jsll/api/jeena-sikho
APP_ADMIN_TOKEN=replace-with-strong-admin-token
BUSINESS_PRIMARY_TIMEFRAME=1d
DJANGO_DEBUG=0
DJANGO_ALLOWED_HOSTS=api.seestox.com,seestox.com,www.seestox.com
DJANGO_CSRF_TRUSTED_ORIGINS=https://api.seestox.com,https://seestox.com,https://www.seestox.com
DJANGO_STATIC_URL=/jsll/static/
DJANGO_STATIC_ROOT=/opt/jsll/app/staticfiles
EXOGENOUS_FEEDS_ENABLE=1
EXOGENOUS_NIFTY_SYMBOL=^NSEI
EXOGENOUS_VIX_SYMBOL=^INDIAVIX
EXOGENOUS_USDINR_SYMBOL=INR=X
EVENT_FEATURES_ENABLE=0
EVENT_CALENDAR_FILE=/opt/jsll/app/data/jsll_events.json
MAX_MISSING_RATIO=0.08
COMPLETENESS_MIN_PCT=90
AUTO_REPAIR_ON_DQ_FAIL=1
```

nginx gotcha:

- `/etc/nginx/proxy_params` already sets `proxy_set_header Host $http_host;`
- do not duplicate `proxy_set_header Host $host;` in the JSLL location block
- otherwise Django can receive a duplicated host header and throw `DisallowedHost`

Long-horizon note:

- `2d` to `7d` predictions can appear before true matched actuals exist
- in that case the dashboard can still show pending/insufficient-sample states
- that is a data maturity issue, not always a deployment bug

## Admin Mutation Endpoints

The write endpoints are now admin-token gated outside debug mode:

- `POST /api/jeena-sikho/tournament/run`
- `POST /api/jeena-sikho/prediction/refresh`

Use either:

```bash
curl -X POST https://api.seestox.com/jsll/api/jeena-sikho/tournament/run \
  -H "X-App-Admin-Token: $APP_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

or:

```bash
curl -X POST https://api.seestox.com/jsll/api/jeena-sikho/prediction/refresh \
  -H "Authorization: Bearer $APP_ADMIN_TOKEN"
```

The dashboard no longer triggers `prediction/refresh` from public browser polling.

## Deep Project Notes

For the detailed project analysis, deployment notes, file-by-file explanation, and command reference, see [`jsll.md`](./jsll.md).
