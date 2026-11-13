"""Live auto paper trader for BTC / ETH / BNB 5-minute up/down markets.

Every few seconds, for each asset:
  1. work out the current 5-minute window and its price to beat
  2. pull the last ~16 minutes of 1-second price bars
  3. compute P(Up) with the model + indicators
  4. read predict.fun's orderbook (or a simulated market without an API key)
  5. place a *paper* bet when the edge clears the thresholds
  6. settle bets when their window closes, and log everything

Nothing here sends real orders.
"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
from dataclasses import asdict

from . import feeds
from .model import ModelParams, predict
from .strategy import Decision, StrategyConfig, decide, settle, simulated_market

STATE_FILE = "paper_state.json"
TRADES_FILE = "paper_trades.csv"
FIELDS = ["opened_at", "asset", "window_start", "seconds_left", "side", "price", "prob", "edge",
          "stake", "price_to_beat", "spot_at_entry", "market_source", "final_price", "won", "pnl",
          "bankroll_after"]


class LiveFeed:
    """Binance prices + predict.fun orderbook (simulated market without an API key)."""
    name = "live"

    def __init__(self):
        self.pf = feeds.PredictFun()

    def bars(self, asset):
        return feeds.klines_to_bars(feeds.binance_klines(asset, "1s", 1000))

    def price_at(self, asset, ts):
        return feeds.binance_price_at(asset, ts)

    def market(self, asset, ws, bars, k, left, cfg):
        """Returns (up_ask, down_ask, source, market_json)."""
        if self.pf.enabled and left <= cfg.max_seconds_left:
            try:
                m = self.pf.find_current_market(asset, ws)
                if m:
                    up, down = feeds.top_of_book(self.pf.orderbook(m["id"]))
                    return up, down, "predict.fun", m
            except Exception as e:
                print(f"  [{asset}] predict.fun error: {e}; using simulated market")
        up, down = simulated_market(bars, k, left)
        return up, down, "simulated", None


class PaperTrader:
    def __init__(self, assets, cfg: StrategyConfig, params: ModelParams, bankroll: float,
                 out_dir: str = ".", feed=None, clock=time.time):
        self.assets, self.cfg, self.params = assets, cfg, params
        self.feed = feed or LiveFeed()
        self.clock = clock
        self.out = out_dir
        self.lock = threading.Lock()
        self.signals: dict[str, dict] = {}
        self.equity: list[list[float]] = []
        self.start_bankroll = bankroll
        os.makedirs(out_dir, exist_ok=True)
        self.state = self._load() or {"bankroll": bankroll, "open": [], "strikes": {}, "stats": {}}
        self.closed: list[dict] = self._load_trades()

    # ------------------------------------------------------------ persistence
    def _path(self, name):
        return os.path.join(self.out, name)

    def _load(self):
        try:
            with open(self._path(STATE_FILE)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _load_trades(self):
        try:
            with open(self._path(TRADES_FILE)) as f:
                return list(csv.DictReader(f))[-500:]
        except OSError:
            return []

    def _save(self):
        with open(self._path(STATE_FILE), "w") as f:
            json.dump(self.state, f, indent=1)

    def _log_trade(self, row: dict):
        path = self._path(TRADES_FILE)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in FIELDS})
        self.closed.append({k: row.get(k, "") for k in FIELDS})
        self.closed = self.closed[-500:]

    # ------------------------------------------------------------ helpers
    def _strike(self, asset: str, ws: int, market: dict | None) -> float | None:
        key = f"{asset}:{ws}"
        if key in self.state["strikes"]:
            return self.state["strikes"][key]
        k = None
        if market:  # prefer the market's own price to beat if it publishes one
            for f in ("priceToBeat", "strikePrice", "startPrice", "referencePrice"):
                if market.get(f):
                    k = float(market[f])
                    break
        if k is None:
            k = self.feed.price_at(asset, ws)
        if k is not None:
            self.state["strikes"][key] = k
            for old in [s for s in self.state["strikes"] if int(s.split(":")[1]) < ws - 3600]:
                del self.state["strikes"][old]
        return k

    def equity_value(self) -> float:
        return self.state["bankroll"] + sum(p["stake"] for p in self.state["open"])

    # ------------------------------------------------------------ main loop
    def settle_due(self, now: float):
        still = []
        for p in self.state["open"]:
            end = p["window_start"] + feeds.WINDOW
            if now < end + 3:
                still.append(p)
                continue
            final = self.feed.price_at(p["asset"], end)
            if final is None:
                still.append(p)
                continue
            d = Decision(p["side"], p["price"], p["prob"], p["edge"], p["stake"])
            won, pnl = settle(d, p["price_to_beat"], final, self.cfg.fee_rate)
            self.state["bankroll"] = round(self.state["bankroll"] + p["stake"] + pnl, 4)
            st = self.state["stats"].setdefault(p["asset"], {"n": 0, "wins": 0, "pnl": 0.0})
            st["n"] += 1
            st["wins"] += int(won)
            st["pnl"] = round(st["pnl"] + pnl, 4)
            self._log_trade({**p, "final_price": final, "won": won, "pnl": pnl,
                             "bankroll_after": self.state["bankroll"]})
            print(f"  SETTLED {p['asset']} {p['side']} @ {p['price']:.2f}  "
                  f"{'WIN ' if won else 'LOSS'} {pnl:+.2f}  bankroll {self.state['bankroll']:.2f}")
        self.state["open"] = still

    def tick(self, verbose: bool = True):
        with self.lock:
            self._tick(verbose)

    def _tick(self, verbose):
        now = self.clock()
        self.settle_due(now)
        ws, we = feeds.window_bounds(now)
        left = we - now
        for asset in self.assets:
            try:
                bars = self.feed.bars(asset)
                holding = any(p["asset"] == asset and p["window_start"] == ws for p in self.state["open"])
                k0 = self.state["strikes"].get(f"{asset}:{ws}") or bars.closes[-1]
                up, down, src, market = self.feed.market(asset, ws, bars, k0, left, self.cfg)
                k = self._strike(asset, ws, market)
                if k is None:
                    continue
                if src == "simulated" and k != k0:
                    up, down = simulated_market(bars, k, left)
                sig = predict(bars, k, left, self.params)
                d = None if holding else decide(sig, up, down, self.equity_value(), self.cfg)
                f = sig.features
                edge_up = None if up is None else sig.p_up - up - self.cfg.fee_rate
                edge_dn = None if down is None else (1 - sig.p_up) - down - self.cfg.fee_rate
                self.signals[asset] = {
                    "asset": asset, "ts": now, "window_start": ws, "seconds_left": left,
                    "spot": sig.spot, "price_to_beat": k, "p_up": sig.p_up, "up_ask": up,
                    "down_ask": down, "edge_up": edge_up, "edge_down": edge_dn, "source": src,
                    "features": f, "sigma": sig.sigma,
                    "action": f"BET {d.side} ${d.stake:.2f} @ {d.price:.2f}" if d else
                              ("HOLDING" if holding else "WATCHING"),
                    "history": (self.signals.get(asset, {}).get("history", []) + [[now, sig.p_up]])[-150:],
                }
                if verbose:
                    print(f"{time.strftime('%H:%M:%S', time.localtime(now))} {asset} left {left:5.1f}s "
                          f"spot {sig.spot:.2f} beat {k:.2f} P(up) {sig.p_up:.3f} ask U/D {up}/{down} "
                          f"[{src}] RSI {f['rsi']:.0f} flow {f['imbalance']:+.2f} score {f['score']:+.2f}"
                          + (f"  -> BET {d.side} ${d.stake} @ {d.price:.2f}" if d else ""))
                if d:
                    self.state["bankroll"] = round(self.state["bankroll"] - d.stake, 4)
                    self.state["open"].append({
                        "opened_at": int(now), "asset": asset, "window_start": ws,
                        "seconds_left": round(left, 1), **asdict(d), "edge": round(d.edge, 4),
                        "prob": round(d.prob, 4), "price_to_beat": k, "spot_at_entry": sig.spot,
                        "market_source": src})
            except Exception as e:
                print(f"  [{asset}] error: {e}")
        if not self.equity or now - self.equity[-1][0] >= 5:
            self.equity.append([now, round(self.equity_value(), 2)])
            self.equity = self.equity[-2000:]
        self._save()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "now": self.clock(), "mode": getattr(self.feed, "name", "live"),
                "bankroll": self.state["bankroll"], "equity": self.equity_value(),
                "start_bankroll": self.start_bankroll, "stats": self.state["stats"],
                "open": self.state["open"], "signals": self.signals,
                "trades": self.closed[-100:][::-1], "equity_curve": self.equity,
                "config": asdict(self.cfg),
            }

    def summary(self) -> str:
        parts = [f"bankroll {self.state['bankroll']:.2f}"]
        for a, s in self.state["stats"].items():
            if s["n"]:
                parts.append(f"{a}: {s['n']} bets {s['wins']/s['n']:.0%} win pnl {s['pnl']:+.2f}")
        return " | ".join(parts)

    def run(self, poll: float = 3.0, verbose: bool = True, stop: threading.Event | None = None):
        pf = getattr(self.feed, "pf", None)
        print(f"Paper trading {', '.join(self.assets)} | feed: {getattr(self.feed, 'name', 'live')} | "
              f"predict.fun orderbook: {'ON' if pf and pf.enabled else 'OFF - simulated market'}")
        last = 0.0
        while not (stop and stop.is_set()):
            self.tick(verbose)
            if time.time() - last > 60:
                print("  >> " + self.summary())
                last = time.time()
            time.sleep(poll)
