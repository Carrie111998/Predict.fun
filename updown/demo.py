"""Offline demo feed: simulated prices and a simulated, slightly mispriced market.

Lets you watch the dashboard and the paper trader without any internet access.
The clock can run faster than real time (--speed). Everything here is fake:
the market noise gives the model something to find, which real markets may not.
"""
from __future__ import annotations

import math
import random
import time

from .feeds import WINDOW
from .model import Bars
from .strategy import simulated_market

START = {"BTC": 100_000.0, "ETH": 3_500.0, "BNB": 650.0}
VOL = {"BTC": 6e-5, "ETH": 8e-5, "BNB": 7e-5}
HISTORY = 1200


class DemoClock:
    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self.t0 = time.time()
        # start 20 s into a window so the first one is visible right away
        self.s0 = int(self.t0) // WINDOW * WINDOW + 20

    def __call__(self) -> float:
        return self.s0 + (time.time() - self.t0) * self.speed


class DemoFeed:
    name = "demo"
    pf = None

    def __init__(self, clock: DemoClock, seed: int = 42, mispricing: float = 0.06):
        self.clock = clock
        self.rng = random.Random(seed)
        self.mispricing = mispricing
        self.t_first = int(clock.s0) - HISTORY
        self.data = {a: {"open": [], "close": [], "vol": [], "tb": [], "r": 0.0, "p": START[a]}
                     for a in START}
        self._extend(int(clock.s0))

    def _extend(self, upto: int):
        for a, d in self.data.items():
            need = upto - self.t_first + 1 - len(d["open"])
            for _ in range(max(0, need)):
                # returns with slight autocorrelation so momentum indicators have signal
                d["r"] = 0.15 * d["r"] + self.rng.gauss(0, VOL[a])
                d["open"].append(d["p"])
                d["p"] *= math.exp(d["r"])
                d["close"].append(d["p"])
                v = self.rng.uniform(0.5, 2.0)
                d["vol"].append(v)
                tb = 0.5 + max(-0.45, min(0.45, d["r"] / VOL[a] * 0.15)) + self.rng.uniform(-0.1, 0.1)
                d["tb"].append(v * max(0.0, min(1.0, tb)))

    def _now_index(self):
        now = int(self.clock())
        self._extend(now)
        return now - self.t_first

    def bars(self, asset):
        i = self._now_index()
        d = self.data[asset]
        lo = max(0, i - 1000)
        return Bars(d["close"][lo:i], d["vol"][lo:i], d["tb"][lo:i], 1.0)

    def price_at(self, asset, ts):
        if ts > self.clock():
            return None
        self._now_index()
        i = int(ts) - self.t_first
        return self.data[asset]["open"][i] if 0 <= i < len(self.data[asset]["open"]) else None

    def market(self, asset, ws, bars, k, left, cfg):
        up, down = simulated_market(bars, k, left)
        noise = random.Random(f"{asset}{ws}{int(left // 15)}").gauss(0, self.mispricing)
        up = round(min(0.99, max(0.01, up + noise)), 2)
        down = round(min(0.99, max(0.01, down - noise)), 2)
        return up, down, "demo", None
