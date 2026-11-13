"""Data feeds: Binance spot prices (public) and predict.fun markets/orderbooks.

Binance needs no key. predict.fun's REST API (https://api.predict.fun) needs an
API key sent as the `x-api-key` header; put it in the PREDICT_API_KEY env var.
Without a key the bot still runs, but fills are simulated at an assumed price
instead of the real orderbook.

NOTE: predict.fun's market JSON fields were taken from its public docs
(dev.predict.fun) and could not be tested live from the build machine, so the
parsing below is deliberately defensive. Run `python -m updown probe` once with
your key and check the output.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

from .model import Bars

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT"}
BINANCE_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
PREDICT_API = os.environ.get("PREDICT_API_BASE", "https://api.predict.fun")
WINDOW = 300  # seconds in a 5-minute market


def _get(url: str, headers: dict | None = None, timeout: float = 10.0):
    req = urllib.request.Request(url, headers={"User-Agent": "updown-bot/1.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------- Binance
def binance_klines(asset: str, interval: str = "1s", limit: int = 1000,
                   start_ms: int | None = None, end_ms: int | None = None) -> list[list]:
    q = {"symbol": SYMBOLS[asset], "interval": interval, "limit": limit}
    if start_ms is not None:
        q["startTime"] = start_ms
    if end_ms is not None:
        q["endTime"] = end_ms
    err = None
    for host in BINANCE_HOSTS:
        try:
            return _get(f"{host}/api/v3/klines?{urllib.parse.urlencode(q)}")
        except Exception as e:  # try the mirror
            err = e
    raise RuntimeError(f"Binance unreachable: {err}")


def klines_to_bars(rows: list[list], bar_seconds: float = 1.0) -> Bars:
    return Bars(
        closes=[float(r[4]) for r in rows],
        volumes=[float(r[5]) for r in rows],
        taker_buy=[float(r[9]) for r in rows],
        bar_seconds=bar_seconds,
    )


def binance_price_at(asset: str, ts: int) -> float | None:
    """Open price of the 1-second bar starting at unix second `ts`."""
    rows = binance_klines(asset, "1s", 1, start_ms=ts * 1000)
    if rows and int(rows[0][0]) // 1000 == ts:
        return float(rows[0][1])
    return None


# ---------------------------------------------------------------- predict.fun
class PredictFun:
    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.environ.get("PREDICT_API_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def _api(self, path: str, **params):
        q = f"?{urllib.parse.urlencode(params)}" if params else ""
        return _get(f"{PREDICT_API}{path}{q}", {"x-api-key": self.key or ""})

    def category(self, slug: str):
        return self._api(f"/v1/categories/{slug}")

    def markets(self, **params):
        return self._api("/v1/markets", **params)

    def orderbook(self, market_id) -> dict:
        return self._api(f"/v1/markets/{market_id}/orderbook")

    def find_current_market(self, asset: str, window_start: int) -> dict | None:
        """Best effort: locate the open 5-minute market for this asset/window."""
        candidates = []
        for slug in (f"{asset.lower()}-updown-5m-{window_start}", f"{asset.lower()}-updown-5m"):
            try:
                data = self.category(slug).get("data") or {}
                candidates += data.get("markets") or []
            except Exception:
                pass
        if not candidates:
            try:
                candidates = self.markets(status="OPEN", first=100).get("data") or []
            except Exception:
                return None
        names = {"BTC": ("btc", "bitcoin"), "ETH": ("eth", "ethereum"), "BNB": ("bnb",)}[asset]
        best = None
        for m in candidates:
            text = f"{m.get('title', '')} {m.get('question', '')}".lower()
            if not any(n in text for n in names) or "up" not in text:
                continue
            if str(m.get("status", "")).upper() in ("RESOLVED", "CLOSED"):
                continue
            best = m if best is None else best
            if str(window_start) in json.dumps(m):
                return m
        return best


def top_of_book(book: dict) -> tuple[float | None, float | None]:
    """Returns (up_ask, down_ask) from a Yes(=Up) side orderbook."""
    d = book.get("data", book)
    asks, bids = d.get("asks") or [], d.get("bids") or []
    up_ask = float(asks[0][0]) if asks else None
    down_ask = round(1 - float(bids[0][0]), 3) if bids else None
    return up_ask, down_ask


def window_bounds(now: float | None = None) -> tuple[int, int]:
    now = time.time() if now is None else now
    start = int(now) // WINDOW * WINDOW
    return start, start + WINDOW
