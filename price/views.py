import requests
import os
from django.http import JsonResponse
from django.shortcuts import render

MARKET_SYMBOL = os.getenv('MARKET_SYMBOL', 'BTC/USDT')
MARKET_YFINANCE_SYMBOL = os.getenv('MARKET_YFINANCE_SYMBOL', 'BTC-USD')
PRICE_SOURCE = os.getenv('PRICE_SOURCE', 'auto').strip().lower()
_default_binance_symbol = MARKET_SYMBOL.replace('/', '').replace('-', '').upper()
BINANCE_TICKER_SYMBOL = os.getenv('BINANCE_TICKER_SYMBOL', _default_binance_symbol)
BINANCE_TICKER_URL = f'https://api.binance.com/api/v3/ticker/price?symbol={BINANCE_TICKER_SYMBOL}'


def index(request):
    return render(
        request,
        'index.html',
        {
            'brand_name': os.getenv('APP_BRAND_NAME', 'Jeena Sikho'),
            'market_label': os.getenv('APP_MARKET_LABEL', 'Jeena Sikho'),
        },
    )


def api_price(request):
    try:
        if PRICE_SOURCE == 'yfinance':
            try:
                import yfinance as yf  # type: ignore
            except Exception as exc:
                raise RuntimeError('yfinance not installed') from exc
            ticker = yf.Ticker(MARKET_YFINANCE_SYMBOL)
            hist = ticker.history(period='1d', interval='1m')
            if hist is not None and not hist.empty:
                amount = float(hist['Close'].dropna().iloc[-1])
            else:
                fast_info = getattr(ticker, 'fast_info', None) or {}
                amount = (
                    fast_info.get('lastPrice')
                    or fast_info.get('regularMarketPrice')
                    or fast_info.get('previousClose')
                )
                if amount is None:
                    raise RuntimeError(f'yfinance price unavailable for {MARKET_YFINANCE_SYMBOL}')
        else:
            resp = requests.get(
                BINANCE_TICKER_URL,
                timeout=4,
                headers={'User-Agent': 'jeena-sikho-price/1.0'},
            )
            resp.raise_for_status()
            data = resp.json()
            amount = data.get('price')
        return JsonResponse({'ok': True, 'amount': amount})
    except Exception as exc:
        return JsonResponse({'ok': False, 'error': str(exc)}, status=502)
