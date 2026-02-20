from django.urls import path
from django.views.generic.base import RedirectView

from . import views

urlpatterns = [
    path('jeena-sikho/', RedirectView.as_view(url='/', permanent=False), name='jeena_sikho_dashboard'),
    path('kite/login', views.kite_login, name='kite_login'),
    path('kite/callback', views.kite_callback, name='kite_callback'),

    path('api/jeena-sikho/price', views.api_price, name='js_price'),
    path('api/jeena-sikho/kite/auth/status', views.api_kite_auth_status, name='js_kite_auth_status'),
    path('api/jeena-sikho/price_at', views.api_price_at, name='js_price_at'),
    path('api/jeena-sikho/tournament/summary', views.api_tournament_summary, name='js_tournament_summary'),
    path('api/jeena-sikho/tournament/scoreboard', views.api_scoreboard, name='js_tournament_scoreboard'),
    path('api/jeena-sikho/tournament/run', views.api_tournament_run, name='js_tournament_run'),
    path('api/jeena-sikho/tournament/run/status', views.api_tournament_run_status, name='js_tournament_run_status'),
    path('api/jeena-sikho/prediction/latest', views.api_prediction_latest, name='js_prediction_latest'),
    path('api/jeena-sikho/prediction/refresh', views.api_prediction_refresh, name='js_prediction_refresh'),
]
