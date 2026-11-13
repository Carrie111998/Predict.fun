"""Turn a model probability + market prices into a trade decision."""
from __future__ import annotations

from dataclasses import dataclass

from .model import ModelParams, Signal, predict, Bars


@dataclass
class StrategyConfig:
    max_seconds_left: float = 150   # don't trade earlier than this (too close to a coin flip)
    min_seconds_left: float = 8     # too late: oracle/latency risk
    min_edge: float = 0.04          # model prob - price - fee must beat this
    min_prob: float = 0.60          # only bet sides the model favours
    max_price: float = 0.93         # above this, one loss wipes out ~13 wins
    fee_rate: float = 0.01          # fee as a fraction of stake (set to the real fee)
    kelly_fraction: float = 0.25    # quarter-Kelly sizing
    max_stake_pct: float = 0.05     # never risk more than 5% of bankroll on one bet
    min_stake: float = 1.0


@dataclass
class Decision:
    side: str
    price: float
    prob: float
    edge: float
    stake: float

    @property
    def shares(self) -> float:
        return self.stake / self.price


def simulated_market(bars: Bars, price_to_beat: float, seconds_left: float,
                     spread: float = 0.02) -> tuple[float, float]:
    """Stand-in for the real orderbook when no API key is set.

    Models an *efficient* market: it prices the same distance-to-strike maths
    with plain volatility and no momentum tilt, plus a spread. Any edge you see
    in this mode comes only from the indicators - which is the honest test.
    """
    s = predict(bars, price_to_beat, seconds_left,
                ModelParams(momentum_weight=0.0, vol_mult=1.0))
    up = min(0.99, round(s.p_up + spread / 2, 2))
    down = min(0.99, round(1 - s.p_up + spread / 2, 2))
    return max(0.01, up), max(0.01, down)


def decide(sig: Signal, up_ask: float | None, down_ask: float | None,
           bankroll: float, cfg: StrategyConfig) -> Decision | None:
    if not (cfg.min_seconds_left <= sig.seconds_left <= cfg.max_seconds_left):
        return None
    best = None
    for side, prob, ask in (("UP", sig.p_up, up_ask), ("DOWN", 1 - sig.p_up, down_ask)):
        if ask is None or not (0 < ask < 1) or ask > cfg.max_price or prob < cfg.min_prob:
            continue
        edge = prob - ask - cfg.fee_rate
        if edge < cfg.min_edge:
            continue
        if best is None or edge > best[3]:
            best = (side, ask, prob, edge)
    if best is None:
        return None
    side, ask, prob, edge = best
    kelly = (prob - ask) / (1 - ask)
    stake = bankroll * min(cfg.max_stake_pct, kelly * cfg.kelly_fraction)
    if stake < cfg.min_stake:
        return None
    return Decision(side, ask, prob, edge, round(stake, 2))


def settle(decision: Decision, price_to_beat: float, final_price: float,
           fee_rate: float) -> tuple[bool, float]:
    """Returns (won, pnl). Up wins on final >= price to beat."""
    up_won = final_price >= price_to_beat
    won = (decision.side == "UP") == up_won
    fee = decision.stake * fee_rate
    pnl = (decision.shares - decision.stake if won else -decision.stake) - fee
    return won, round(pnl, 4)
