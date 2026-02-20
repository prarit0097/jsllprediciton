import json
import logging
import os

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt

from jeena_sikho_tournament.config import TournamentConfig
from jeena_sikho_tournament.kite_client import exchange_request_token, kite_login_url

from .services import (
    get_live_price,
    get_kite_auth_status,
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


def dashboard(request):
    brand_name = os.getenv("APP_BRAND_NAME", "Jeena Sikho")
    market_label = os.getenv("APP_MARKET_LABEL", "Jeena Sikho")
    api_prefix = os.getenv("APP_API_PREFIX", "/api/jeena-sikho").strip() or "/api/jeena-sikho"
    if not api_prefix.startswith("/"):
        api_prefix = f"/{api_prefix}"
    return render(
        request,
        "jeena_sikho_dashboard.html",
        {
            "brand_name": brand_name,
            "market_label": market_label,
            "api_prefix": api_prefix.rstrip("/"),
        },
    )


def kite_login(request):
    try:
        return redirect(kite_login_url())
    except Exception as exc:
        LOGGER.exception("kite login url error")
        return JsonResponse({"error": str(exc)}, status=500)


def kite_callback(request):
    request_token = (request.GET.get("request_token") or "").strip()
    if not request_token:
        return JsonResponse({"error": "request_token missing"}, status=400)
    try:
        data = exchange_request_token(request_token)
        access_token = (data.get("access_token") or "").strip()
        return JsonResponse(
            {
                "ok": True,
                "message": "Kite access token updated",
                "user_id": data.get("user_id"),
                "user_name": data.get("user_name"),
                "login_time": data.get("login_time"),
                "access_token_tail": (access_token[-6:] if access_token else None),
            }
        )
    except Exception as exc:
        LOGGER.exception("kite callback error")
        return JsonResponse({"error": str(exc)}, status=502)


def api_price(request):
    try:
        data = get_live_price()
        return JsonResponse(data)
    except Exception as exc:
        LOGGER.exception("price error")
        return JsonResponse({"error": str(exc)}, status=502)


def api_kite_auth_status(request):
    try:
        return JsonResponse(get_kite_auth_status())
    except Exception as exc:
        LOGGER.exception("kite auth status error")
        return JsonResponse({"error": str(exc)}, status=500)


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
        data = get_scoreboard(limit)
        return JsonResponse(data, safe=False)
    except Exception as exc:
        LOGGER.exception("scoreboard error")
        return JsonResponse({"error": str(exc)}, status=500)


@csrf_exempt
def api_tournament_run(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
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


@csrf_exempt
def api_prediction_refresh(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        data = refresh_prediction(_config())
        return JsonResponse(data)
    except Exception as exc:
        LOGGER.exception("prediction refresh error")
        return JsonResponse({"error": str(exc)}, status=500)


