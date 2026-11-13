"""Technical indicators used by the 5-minute up/down model.

All functions take plain Python lists (oldest first) so the bot has no
third-party dependencies.
"""
from __future__ import annotations

import math
from typing import Sequence


def ema(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(closes: Sequence[float], period: int = 14) -> float:
    """Wilder RSI of the latest bar (50 when there is not enough data)."""
    if len(closes) <= period:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_g, avg_l = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0.0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0.0)) / period
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def log_returns(closes: Sequence[float]) -> list[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]


def realized_vol_per_sec(closes: Sequence[float], bar_seconds: float) -> float:
    """Std-dev of log returns, rescaled to a per-second volatility."""
    r = log_returns(closes)
    if len(r) < 2:
        return 0.0
    mean = sum(r) / len(r)
    var = sum((x - mean) ** 2 for x in r) / (len(r) - 1)
    return math.sqrt(var / bar_seconds)


def ewma_vol_per_sec(closes: Sequence[float], bar_seconds: float, lam: float = 0.94) -> float:
    """RiskMetrics-style EWMA volatility: reacts faster to volatility bursts."""
    r = log_returns(closes)
    if len(r) < 2:
        return 0.0
    var = r[0] ** 2
    for x in r[1:]:
        var = lam * var + (1 - lam) * x * x
    return math.sqrt(var / bar_seconds)


def vwap(closes: Sequence[float], volumes: Sequence[float]) -> float:
    tot = sum(volumes)
    if tot <= 0:
        return closes[-1] if closes else 0.0
    return sum(c * v for c, v in zip(closes, volumes)) / tot


def taker_imbalance(volumes: Sequence[float], taker_buy_volumes: Sequence[float]) -> float:
    """Order-flow imbalance in [-1, 1]: +1 = all aggressive buying."""
    tot = sum(volumes)
    if tot <= 0:
        return 0.0
    buy = sum(taker_buy_volumes)
    return (2 * buy - tot) / tot


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
