import json
import logging
import os
import secrets

from django.http import JsonResponse
from django.shortcuts import render

from jeena_sikho_tournament.config import TournamentConfig

from .services import (
    get_live_price,
    get_price_at_timestamp,
    get_scoreboard,
    get_tournament_summary,
    latest_prediction,
    refresh_prediction,
    run_tournament_async,
    run_status,
)

LOGGER = logging.getLogger(__name__)


def _config() -> TournamentConfig:
    return TournamentConfig()


def _admin_token() -> str:
    return (
        os.getenv("APP_ADMIN_TOKEN", "").strip()
        or os.getenv("JSLL_ADMIN_TOKEN", "").strip()
        or os.getenv("ADMIN_API_TOKEN", "").strip()
    )


def _debug_enabled() -> bool:
    return os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"} or os.getenv("DJANGO_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _provided_admin_token(request) -> str:
    auth_header = request.META.get("HTTP_AUTHORIZATION", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.META.get("HTTP_X_APP_ADMIN_TOKEN", "").strip()


def _admin_required(request):
    expected = _admin_token()
    if not expected:
        if _debug_enabled():
            return None
        return JsonResponse({"error": "admin token required"}, status=403)
    provided = _provided_admin_token(request)
    if provided and secrets.compare_digest(provided, expected):
        return None
    return JsonResponse({"error": "admin token required"}, status=403)


def dashboard(request):
    brand_name = os.getenv("APP_BRAND_NAME", "Jeena Sikho")
    market_label = os.getenv("APP_MARKET_LABEL", "Jeena Sikho")
    api_prefix = os.getenv("APP_API_PREFIX", "/api/jeena-sikho").strip() or "/api/jeena-sikho"
    base_prefix = os.getenv("APP_BASE_PREFIX", "").strip()
    if base_prefix and not base_prefix.startswith("/"):
        base_prefix = f"/{base_prefix}"
    base_prefix = base_prefix.rstrip("/")
    if not api_prefix.startswith("/"):
        api_prefix = f"/{api_prefix}"
    return render(
        request,
        "jeena_sikho_dashboard.html",
        {
            "brand_name": brand_name,
            "market_label": market_label,
            "base_prefix": base_prefix,
            "api_prefix": api_prefix.rstrip("/"),
        },
    )


def api_price(request):
    try:
        data = get_live_price()
        return JsonResponse(data)
    except Exception as exc:
        LOGGER.exception("price error")
        return JsonResponse({"error": str(exc)}, status=502)


def api_price_at(request):
    try:
        value = request.GET.get("ts")
        if not value:
            return JsonResponse({"error": "ts required"}, status=400)
        data = get_price_at_timestamp(_config(), value)
        return JsonResponse(data)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except LookupError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        LOGGER.exception("price at error")
        return JsonResponse({"error": str(exc)}, status=500)


def api_tournament_summary(request):
    try:
        data = get_tournament_summary(_config())
        return JsonResponse(data)
    except Exception as exc:
        LOGGER.exception("summary error")
        return JsonResponse({"error": str(exc)}, status=500)


def api_scoreboard(request):
    try:
        limit = int(request.GET.get("limit", 500))
        data = get_scoreboard(_config(), limit)
        return JsonResponse(data, safe=False)
    except Exception as exc:
        LOGGER.exception("scoreboard error")
        return JsonResponse({"error": str(exc)}, status=500)


def api_tournament_run(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    denied = _admin_required(request)
    if denied is not None:
        return denied
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        body = {}
    mode = body.get("run_mode")
    status = run_tournament_async(_config(), mode)
    return JsonResponse(status)


def api_tournament_run_status(request):
    try:
        return JsonResponse(run_status())
    except Exception as exc:
        LOGGER.exception("run status error")
        return JsonResponse({"error": str(exc)}, status=500)


def api_prediction_latest(request):
    try:
        data = latest_prediction(_config())
        if not data:
            return JsonResponse({"error": "no prediction"}, status=404)
        return JsonResponse(data)
    except Exception as exc:
        LOGGER.exception("prediction latest error")
        return JsonResponse({"error": str(exc)}, status=500)


def api_prediction_refresh(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    denied = _admin_required(request)
    if denied is not None:
        return denied
    try:
        data = refresh_prediction(_config())
        return JsonResponse(data)
    except Exception as exc:
        LOGGER.exception("prediction refresh error")
        return JsonResponse({"error": str(exc)}, status=500)


