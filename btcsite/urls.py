from django.contrib import admin
from django.urls import path, include
from jeena_sikho_dashboard import views as js_views
from price.views import index, api_price

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', js_views.dashboard, name='index'),
    path('price/', index, name='price_index'),
    path('api/price', api_price, name='api_price'),
    path('', include('jeena_sikho_dashboard.urls')),
]

