"""Probability model for "will the price close at/above the price to beat?".

Core idea (this is where any real edge in 5-minute markets comes from):

    P(Up) = Phi( (ln(S / K) + mu * t) / (sigma * sqrt(t)) )

    S     = current spot price
    K     = price to beat (price at the start of the 5-minute window)
    t     = seconds left in the window
    sigma = short-term volatility per sqrt(second)
    mu    = small drift tilt from momentum / order-flow indicators

Early in the window S ~= K, so P(Up) ~= 50% and the indicators only nudge it
a little: nobody predicts 5-minute noise reliably. Late in the window, when
price has moved away from K and little time is left, P(Up) becomes very
confident. The trade is only worth taking when the market's ask price is
*cheaper* than this probability (minus fees).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import indicators as ind


@dataclass
class Bars:
    """1-second (or 1-minute) bars, oldest first."""
    closes: list[float]
    volumes: list[float]
    taker_buy: list[float]
    bar_seconds: float = 1.0


@dataclass
class Signal:
    p_up: float
    spot: float
    price_to_beat: float
    seconds_left: float
    sigma: float
    drift: float
    features: dict = field(default_factory=dict)

    @property
    def side(self) -> str:
        return "UP" if self.p_up >= 0.5 else "DOWN"

    @property
    def confidence(self) -> float:
        return max(self.p_up, 1 - self.p_up)


@dataclass
class ModelParams:
    vol_lookback_s: int = 900          # seconds of bars used for volatility
    vol_floor: float = 2e-5            # min per-sqrt-second vol (avoids overconfidence)
    vol_mult: float = 1.15             # >1 = deliberately humble (fat tails)
    momentum_weight: float = 0.15      # how much indicators tilt the drift
    max_prob: float = 0.985            # never claim certainty


def features(bars: Bars) -> dict:
    c = bars.closes
    if len(c) < 30:
        return {"rsi": 50.0, "ema_gap": 0.0, "imbalance": 0.0, "vwap_gap": 0.0, "mom": 0.0}
    fast = ind.ema(c[-120:], 12)[-1]
    slow = ind.ema(c[-120:], 48)[-1]
    w = min(len(c), 300)
    vw = ind.vwap(c[-w:], bars.volumes[-w:])
    look = min(len(c) - 1, 60)
    return {
        "rsi": ind.rsi(c[-200:], 14),
        "ema_gap": (fast - slow) / slow,
        "imbalance": ind.taker_imbalance(bars.volumes[-60:], bars.taker_buy[-60:]),
        "vwap_gap": (c[-1] - vw) / vw,
        "mom": math.log(c[-1] / c[-1 - look]),
    }


def momentum_score(f: dict) -> float:
    """Combine indicators into a score in roughly [-1, 1] (+ = bullish)."""
    s = 0.0
    s += max(-1.0, min(1.0, f["ema_gap"] / 0.0008)) * 0.35
    s += max(-1.0, min(1.0, f["imbalance"] / 0.4)) * 0.30
    s += max(-1.0, min(1.0, f["mom"] / 0.0015)) * 0.20
    # RSI extremes mean-revert on very short horizons
    if f["rsi"] > 75:
        s -= 0.15
    elif f["rsi"] < 25:
        s += 0.15
    return max(-1.0, min(1.0, s))


def predict(bars: Bars, price_to_beat: float, seconds_left: float,
            params: ModelParams | None = None) -> Signal:
    p = params or ModelParams()
    spot = bars.closes[-1]
    n = max(2, int(p.vol_lookback_s / bars.bar_seconds))
    window = bars.closes[-n:]
    sigma = max(
        ind.realized_vol_per_sec(window, bars.bar_seconds),
        ind.ewma_vol_per_sec(window, bars.bar_seconds),
        p.vol_floor,
    ) * p.vol_mult
    f = features(bars)
    score = momentum_score(f)
    # drift expressed as a fraction of one sigma per sqrt(second)
    drift = score * p.momentum_weight * sigma / math.sqrt(60.0)
    t = max(seconds_left, 0.5)
    z = (math.log(spot / price_to_beat) + drift * t) / (sigma * math.sqrt(t))
    pu = ind.norm_cdf(z)
    pu = min(p.max_prob, max(1 - p.max_prob, pu))
    f["score"] = score
    return Signal(pu, spot, price_to_beat, seconds_left, sigma, drift, f)
