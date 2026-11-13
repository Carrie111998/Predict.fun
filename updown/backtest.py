"""Backtest on historical 1-second Binance bars.

Two outputs:
1. Calibration: when the model says 70%, does that side really win ~70%?
   This is what "make the prediction more accurate" means in practice.
2. Simulated P&L against an efficient simulated market (see strategy.py).
   Real predict.fun prices are not in the history, so treat P&L as a rough
   guide; the calibration table is the trustworthy part.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .feeds import WINDOW, binance_klines
from .model import Bars, ModelParams, predict
from .strategy import StrategyConfig, decide, settle, simulated_market

CHECKPOINTS = (240, 180, 120, 90, 60, 30, 15)
BUCKETS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))


@dataclass
class Series:
    ts: list[int]          # unix second of each bar's open
    opens: list[float]
    bars: Bars


def fetch_history(asset: str, hours: float) -> Series:
    end = int(time.time()) // WINDOW * WINDOW
    start = end - int(hours * 3600) - 900  # extra 15 min warm-up for volatility
    rows, cur = [], start
    while cur < end:
        chunk = binance_klines(asset, "1s", 1000, start_ms=cur * 1000, end_ms=end * 1000 - 1)
        if not chunk:
            break
        rows += chunk
        cur = int(chunk[-1][0]) // 1000 + 1
    return Series(
        ts=[int(r[0]) // 1000 for r in rows],
        opens=[float(r[1]) for r in rows],
        bars=Bars([float(r[4]) for r in rows], [float(r[5]) for r in rows],
                  [float(r[9]) for r in rows], 1.0),
    )


def synthetic_history(hours: float, start_price: float = 100_000.0,
                      vol_per_sec: float = 8e-5, seed: int = 7) -> Series:
    """Random-walk prices for offline testing (no edge exists by design)."""
    rng = random.Random(seed)
    n = int(hours * 3600) + 900
    t0 = (int(time.time()) // WINDOW * WINDOW) - n
    t0 = t0 // WINDOW * WINDOW
    price, opens, closes, vols, tb = start_price, [], [], [], []
    for _ in range(n):
        opens.append(price)
        price *= math.exp(rng.gauss(0, vol_per_sec))
        closes.append(price)
        v = rng.uniform(0.5, 2.0)
        vols.append(v)
        tb.append(v * rng.uniform(0.3, 0.7))
    return Series(list(range(t0, t0 + n)), opens, Bars(closes, vols, tb, 1.0))


@dataclass
class Report:
    windows: int = 0
    calib: dict = field(default_factory=dict)     # (checkpoint, bucket) -> [n, wins]
    trades: list = field(default_factory=list)
    bankroll_start: float = 1000.0
    bankroll: float = 1000.0

    def text(self, asset: str) -> str:
        lines = [f"== {asset}: {self.windows} five-minute windows =="]
        lines.append("Calibration (model's favoured side: predicted vs actual win rate)")
        lines.append("  secs-left  " + "  ".join(f"{lo:.0%}-{min(hi,1):.0%}".rjust(11) for lo, hi in BUCKETS))
        for cp in CHECKPOINTS:
            cells = []
            for b in BUCKETS:
                n, w = self.calib.get((cp, b), [0, 0])
                cells.append(f"{w/n:5.0%} n={n:<4}" if n else "     -     ")
            lines.append(f"  {cp:>9}  " + "  ".join(c.rjust(11) for c in cells))
        n = len(self.trades)
        if n:
            wins = sum(1 for t in self.trades if t["won"])
            pnl = sum(t["pnl"] for t in self.trades)
            avg_px = sum(t["price"] for t in self.trades) / n
            lines.append(f"Paper trades: {n}  win rate {wins/n:.1%}  avg entry price {avg_px:.2f} "
                         f"(break-even win rate ~{avg_px:.0%})")
            lines.append(f"P&L {pnl:+.2f}  bankroll {self.bankroll_start:.0f} -> {self.bankroll:.2f}")
        else:
            lines.append("Paper trades: none (no edge found under current settings)")
        return "\n".join(lines)


def run(series: Series, cfg: StrategyConfig | None = None, params: ModelParams | None = None,
        bankroll: float = 1000.0, step: int = 5) -> Report:
    cfg = cfg or StrategyConfig()
    rep = Report(bankroll_start=bankroll, bankroll=bankroll)
    idx = {t: i for i, t in enumerate(series.ts)}
    first = (series.ts[0] + 900 + WINDOW - 1) // WINDOW * WINDOW
    for ws in range(first, series.ts[-1] - WINDOW + 1, WINDOW):
        i0, i1 = idx.get(ws), idx.get(ws + WINDOW)
        if i0 is None or i1 is None:
            continue
        k, final = series.opens[i0], series.opens[i1]
        up_won = final >= k
        rep.windows += 1
        traded = False
        for left in range(WINDOW - step, 0, -step):
            i = idx.get(ws + WINDOW - left)
            if i is None:
                continue
            view = Bars(series.bars.closes[max(0, i - 1000):i], series.bars.volumes[max(0, i - 1000):i],
                        series.bars.taker_buy[max(0, i - 1000):i], 1.0)
            if len(view.closes) < 60:
                continue
            sig = predict(view, k, left, params)
            if left in CHECKPOINTS:
                fav_up = sig.p_up >= 0.5
                for b in BUCKETS:
                    if b[0] <= sig.confidence < b[1]:
                        cell = rep.calib.setdefault((left, b), [0, 0])
                        cell[0] += 1
                        cell[1] += int(fav_up == up_won)
            if not traded:
                up_ask, down_ask = simulated_market(view, k, left)
                d = decide(sig, up_ask, down_ask, rep.bankroll, cfg)
                if d:
                    won, pnl = settle(d, k, final, cfg.fee_rate)
                    rep.bankroll += pnl
                    rep.trades.append({"window": ws, "left": left, "side": d.side, "price": d.price,
                                       "prob": round(d.prob, 3), "stake": d.stake, "won": won, "pnl": pnl})
                    traded = True
    return rep
