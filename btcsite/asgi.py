import os

from jeena_sikho.env import load_env
from django.core.asgi import get_asgi_application

load_env()
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'btcsite.settings')

application = get_asgi_application()

