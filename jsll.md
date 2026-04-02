# JSLL Project Deep Notes

## 1. What This Project Is

This repository is the JSLL prediction web app for Jeena Sikho Lifecare Limited. It combines:

- a Django web layer
- a dashboard-facing service layer
- an NSE-aware tournament/prediction engine
- local SQLite/runtime artifact storage

Its job is not just to show price. It tries to continuously answer:

- current market price kya hai
- different horizons par model kya predict kar raha hai
- kaunsa model champion hai
- prediction kitni confidence ke saath aayi hai
- recent production performance stable hai ya drift ho raha hai
- data complete aur healthy hai ya nahi

## 2. Why It Exists

Plain price dashboards sirf latest quote dikhate hain. Yeh system usse aage jaata hai:

- market data collect karta hai
- features engineer karta hai
- horizon-wise tournaments chalata hai
- best model family choose karta hai
- predictions ko confidence/band ke saath serve karta hai
- prediction vs actual matching maintain karta hai
- drift aur data-quality failure detect karta hai

In short, this is a monitoring + prediction ops app, not a simple quote widget.

## 3. High-Level Architecture

Request path:

1. Browser `/` hit karta hai
2. `btcsite.urls` request ko dashboard view tak route karta hai
3. `jeena_sikho_dashboard.views.dashboard()` template render karta hai
4. Frontend JS periodic API calls bhejta hai
5. `jeena_sikho_dashboard.services` live price, summary, scoreboard, predictions aur match data assemble karta hai
6. Tournament code `jeena_sikho_tournament` se model/state/load logic use hota hai
7. Runtime state SQLite/files/JSON registry se read hota hai

Background flow:

1. Scheduler `jeena_sikho.run_hourly` ya service-level runner invoke karta hai
2. Tournament/data refresh pipeline execute hoti hai
3. Registry, scores, predictions, drift metrics update hote hain
4. Dashboard next poll par naya state dikha deta hai

## 4. Top-Level Folder Map

- `btcsite/`: Django project shell
- `jeena_sikho/`: thin entrypoint wrappers for hourly and repair jobs
- `jeena_sikho_dashboard/`: user-facing app, APIs, runtime DB access, dashboard static/template
- `jeena_sikho_tournament/`: model training, feature engineering, validation, repair, calendar, diagnostics
- `price/`: older/simple price app
- `templates/`: shared root templates
- `static/`: shared static files
- `tests/`: standalone project tests
- `scripts/`: helper launch commands

## 5. Web Routes

Defined via [`btcsite/urls.py`](./btcsite/urls.py) and [`jeena_sikho_dashboard/urls.py`](./jeena_sikho_dashboard/urls.py).

Main routes:

- `/`: main dashboard
- `/price/`: simple legacy page
- `/api/price`: basic price JSON
- `/api/jeena-sikho/price`
- `/api/jeena-sikho/price_at`
- `/api/jeena-sikho/tournament/summary`
- `/api/jeena-sikho/tournament/scoreboard`
- `/api/jeena-sikho/tournament/run`
- `/api/jeena-sikho/tournament/run/status`
- `/api/jeena-sikho/prediction/latest`
- `/api/jeena-sikho/prediction/refresh`

Legacy redirects:

- `/btc/` -> `/`
- `/jeena-sikho/` -> `/`

## 6. Dashboard Layer

Main template:

- [`jeena_sikho_dashboard/templates/jeena_sikho_dashboard.html`](./jeena_sikho_dashboard/templates/jeena_sikho_dashboard.html)

Main frontend JS:

- [`jeena_sikho_dashboard/static/jeena_sikho_dashboard.js`](./jeena_sikho_dashboard/static/jeena_sikho_dashboard.js)

Dashboard shows:

- current price
- update time / stale state
- manual historical price lookup
- latest prediction row or multi-horizon list
- prediction confidence and band
- last matched prediction vs actual
- horizon metrics like MAE, MAPE, hit-rate, utility, calibration RMSE
- tournament summary
- drift state
- horizon champion summary
- completeness summary
- scoreboard with filters

Frontend behavior:

- page load par APIs fetch hoti hain
- polling loop data refresh karta hai
- prediction refresh endpoint ko time-based manner mein trigger kiya ja sakta hai
- UI multi-horizon aur single-primary-prediction dono modes handle karti hai

## 7. Service Layer

Main backend logic file:

- [`jeena_sikho_dashboard/services.py`](./jeena_sikho_dashboard/services.py)

This file handles:

- live quote sourcing
- FX conversion for USD-priced feeds
- NSE market-state interpretation
- tournament summary assembly
- prediction retrieval and refresh
- pending prediction settlement
- scoreboard access
- confidence calculation
- backtest report shaping
- drift/completeness summaries

Notable behaviors:

- price source fallback exists across yfinance/binance/coinbase/kraken
- `.NS` / Indian equity symbols trigger NSE-aware market handling
- live quote response can include `price_inr`, `fx_rate`, `fx_stale`, `price_mode`
- cached stale result fallback exists if fresh fetch fails
- run-state JSON file helps async tournament status
- prediction target timestamps respect NSE interval alignment

## 8. Tournament / ML Layer

Main package:

- [`jeena_sikho_tournament/`](./jeena_sikho_tournament)

This package is responsible for:

- OHLCV acquisition/stitching
- feature generation
- supervised dataset creation
- candidate/model evaluation
- champion selection
- prediction generation
- diagnostics
- drift-based retrain decisions
- repair/backfill flows

Important modules:

- `features.py`: engineered features and supervised targets
- `predict.py`: prediction helpers
- `tournament.py`: main tournament logic
- `data_sources.py`: source stitching and freshness preference
- `market_calendar.py`: NSE time/holiday/session logic
- `validator.py`: data validation checks
- `repair.py`: missing/gap repair
- `run_hourly.py`: scheduled orchestrator
- `run_repair.py`: repair entry flow
- `doctor.py`: self-check/smoke diagnostics

## 9. Feature Engineering Shape

The tournament package already contains a richer feature pipeline than a basic momentum-only model.

Observed feature families include:

- returns across windows
- rolling means/std/z-scores
- RSI
- ATR-style volatility
- MACD / signal / histogram
- Bollinger width
- VWAP distance
- volume regime features
- trend/range/high-vol/low-vol flags
- gap from previous close
- NSE session-aware time features

Targets are also controlled, clipped, and normalized for more stable training.

## 10. Prediction Behavior

Predictions are not just one raw number. The app tracks:

- predicted price
- low/high band
- confidence %
- horizon/timeframe
- regime
- later actual matched price
- match %
- error metrics

The service layer also stores and updates pending predictions once target time passes.

## 11. Runtime Storage

There are two storage styles here:

1. Django default DB:
- `db.sqlite3`

2. App/runtime artifacts under configured data dir:
- OHLCV SQLite files
- registry JSON
- log files
- run-state JSON

Dashboard-specific runtime tables are managed by [`jeena_sikho_dashboard/db.py`](./jeena_sikho_dashboard/db.py).

Stored data includes:

- runs
- scores
- predictions
- match metadata
- confidence flags
- regime info

## 12. Environment and Configuration

Primary reference:

- [`.env.example`](./.env.example)

Important groups:

Branding / routing:

- `APP_BRAND_NAME`
- `APP_MARKET_LABEL`
- `APP_API_PREFIX`
- `APP_DATA_DIR`

Market selection:

- `MARKET_SYMBOL`
- `MARKET_YFINANCE_SYMBOL`
- `PRICE_SOURCE`

Tournament sizing:

- `RUN_MODE`
- `MAX_CANDIDATES_TOTAL`
- `MAX_CANDIDATES_PER_TARGET`
- `MAX_WORKERS`
- `TOURNAMENT_CV_FOLDS`

Prediction quality / production gates:

- `LOW_CONFIDENCE_PCT`
- `LOW_CONFIDENCE_SKIP_PCT`
- `LOW_CONF_DOWNWEIGHT`
- `PRED_BAND_Z`
- `PROD_MAX_MAPE`
- `PROD_MIN_HIT_RATE`

Data quality / repair:

- `STRICT_DATA_QUALITY`
- `MAX_MISSING_RATIO`
- `COMPLETENESS_MIN_PCT`
- `AUTO_REPAIR_ON_DQ_FAIL`
- `BACKFILL_GAP_REPAIR`
- `NIGHTLY_REPAIR_AFTER_CLOSE`
- `PREOPEN_REFILL_ENABLE`
- `SCHEDULED_AUTO_REPAIR_ENABLE`

NSE calendar:

- `NSE_HOLIDAYS`
- `NSE_ADHOC_HOLIDAYS`
- `NSE_HOLIDAY_FILE`

## 13. Commands

Local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Hourly/tournament run:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho.run_hourly
```

Force run:

```powershell
$env:FORCE_RUN='1'
.\.venv\Scripts\python.exe -m jeena_sikho.run_hourly
```

Repair:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho.run_repair
```

Doctor:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho_tournament.doctor
```

Django checks:

```powershell
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py migrate
```

Tests:

```powershell
python -m pytest tests -q
```

Important note:

- `manage.py test` does not discover the standalone `tests/` suite in this repo
- `pytest` is not listed in `requirements.txt`, so some environments will need `pip install pytest`

## 14. Current VPS Deployment Notes

As provided by the maintainer, the live app currently runs like this:

- code path: `/opt/jsll/app`
- venv path: `/opt/jsll/venv`
- public URL: `https://api.seestox.com/jsll/`
- static URL: `https://api.seestox.com/jsll/static/`
- Gunicorn service: `jsll-gunicorn.service`
- scheduler service: `jsll-scheduler.service`
- socket: `/run/jsll/jsll.sock`

nginx mounts the app under `/jsll/` and strips that prefix before proxying to Gunicorn. That means Django itself still serves root-style paths, but users access the app through `/jsll/`.

Typical deploy/update sequence on VPS:

```bash
cd /opt/jsll/app
source /opt/jsll/venv/bin/activate
git fetch origin
git reset --hard origin/main
pip install -r requirements.txt
python manage.py migrate
python manage.py check
sudo systemctl restart jsll-gunicorn
sudo systemctl restart jsll-scheduler
```

Only use `git reset --hard` after taking a backup if the VPS folder contains local-only edits.

## 15. File-by-File Quick Notes

- [`manage.py`](./manage.py): loads env and boots Django using `btcsite.settings`
- [`btcsite/settings.py`](./btcsite/settings.py): Django config, installed apps, sqlite DB, static dirs
- [`btcsite/urls.py`](./btcsite/urls.py): project URL entrypoint
- [`price/views.py`](./price/views.py): simple/legacy quote page + JSON response
- [`jeena_sikho_dashboard/views.py`](./jeena_sikho_dashboard/views.py): dashboard page and API view functions
- [`jeena_sikho_dashboard/services.py`](./jeena_sikho_dashboard/services.py): core aggregation/business logic
- [`jeena_sikho_dashboard/db.py`](./jeena_sikho_dashboard/db.py): runtime SQLite schema and accessors
- [`jeena_sikho_dashboard/templates/jeena_sikho_dashboard.html`](./jeena_sikho_dashboard/templates/jeena_sikho_dashboard.html): main dashboard HTML
- [`jeena_sikho_dashboard/static/jeena_sikho_dashboard.js`](./jeena_sikho_dashboard/static/jeena_sikho_dashboard.js): dashboard client logic
- [`jeena_sikho_tournament/run_hourly.py`](./jeena_sikho_tournament/run_hourly.py): periodic orchestration
- [`jeena_sikho_tournament/run_repair.py`](./jeena_sikho_tournament/run_repair.py): repair orchestration
- [`jeena_sikho_tournament/doctor.py`](./jeena_sikho_tournament/doctor.py): health/smoke checks
- [`scripts/run_hourly_task.cmd`](./scripts/run_hourly_task.cmd): helper launcher
- [`scripts/run_nightly_repair.cmd`](./scripts/run_nightly_repair.cmd): helper launcher

## 16. Gaps / Risks

Current repo risks visible from inspection:

- root `btcsite/settings.py` is still dev-oriented (`DEBUG=True`, permissive/basic local settings)
- `pytest` test runner is implied by repo layout but not pinned in `requirements.txt`
- some docs were stale and BTC-oriented before this update
- deployment is path-prefixed in production, so any future hardcoded root-relative frontend links must be checked carefully

## 17. Bottom Line

This repository is a JSLL-specific market intelligence application with:

- live dashboard UX
- NSE-aware tournament ML pipeline
- prediction confidence/banding
- drift and data-quality controls
- deployment-ready service structure

It is materially more than a basic Django CRUD or price app. The core value is operational prediction monitoring for one market instrument with scheduled retraining and production-facing summaries.

## 18. Production Incident Notes

During live VPS deployment/debugging, the following production issues were identified and fixed:

1. `DisallowedHost` because settings did not correctly read the VPS env shape.
2. Duplicate `Host` header from nginx because `proxy_params` already set `Host` and the site config added another `proxy_set_header Host $host;`.
3. `DJANGO_*` env aliases were present in `.env`, but settings originally only read plain `DEBUG` / `ALLOWED_HOSTS` style keys.
4. `collectstatic` failed because `STATIC_ROOT` env support was missing.
5. Dashboard JS asset path was hardcoded root-relative and did not respect the `/jsll/` path prefix.
6. Scheduler failed because `timedelta` import was missing in tournament code.
7. Historical `last_ready` dashboard rendering was too strict and could hide legacy matched rows.

## 19. Recommended VPS Runbook

### Safe Deploy / Update

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

### Full Data Refresh

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

### Health / Logs

```bash
sudo systemctl status jsll-gunicorn --no-pager
sudo systemctl status jsll-scheduler --no-pager
sudo journalctl -u jsll-gunicorn -n 50 --no-pager -l
sudo journalctl -u jsll-scheduler -n 80 --no-pager -l
sudo journalctl -u jsll-scheduler --since "10 minutes ago" --no-pager -l
```

### Browser Refresh

After deploy or static changes:

- use `Ctrl+Shift+R`
- or open a new incognito tab

### Important `.env` Values

```env
APP_BASE_PREFIX=/jsll
APP_API_PREFIX=/jsll/api/jeena-sikho
DJANGO_DEBUG=0
DJANGO_ALLOWED_HOSTS=api.seestox.com,seestox.com,www.seestox.com
DJANGO_CSRF_TRUSTED_ORIGINS=https://api.seestox.com,https://seestox.com,https://www.seestox.com
DJANGO_STATIC_URL=/jsll/static/
DJANGO_STATIC_ROOT=/opt/jsll/app/staticfiles
MAX_MISSING_RATIO=0.08
COMPLETENESS_MIN_PCT=90
AUTO_REPAIR_ON_DQ_FAIL=1
```

### Current Long-Horizon Reality

- `2d` to `7d` rows can still show pending/insufficient-sample states until enough true matched actuals exist
- this is not always a deploy failure
- it can simply mean those horizons do not yet have enough completed outcomes in the runtime DB
