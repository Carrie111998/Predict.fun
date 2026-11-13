"""Live auto paper trader for BTC / ETH / BNB 5-minute up/down markets.

Every few seconds, for each asset:
  1. work out the current 5-minute window and its price to beat
  2. pull the last ~16 minutes of 1-second Binance bars
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


class PaperTrader:
    def __init__(self, assets, cfg: StrategyConfig, params: ModelParams, bankroll: float,
                 out_dir: str = "."):
        self.assets, self.cfg, self.params = assets, cfg, params
        self.pf = feeds.PredictFun()
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.state = self._load() or {"bankroll": bankroll, "open": [], "strikes": {}, "stats": {}}

    # ------------------------------------------------------------ persistence
    def _path(self, name):
        return os.path.join(self.out, name)

    def _load(self):
        try:
            with open(self._path(STATE_FILE)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

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
            k = feeds.binance_price_at(asset, ws)
        if k is not None:
            self.state["strikes"][key] = k
            # keep the map small
            for old in [s for s in self.state["strikes"] if int(s.split(":")[1]) < ws - 3600]:
                del self.state["strikes"][old]
        return k

    def _market_prices(self, asset, ws, bars, k, left):
        if self.pf.enabled:
            try:
                m = self.pf.find_current_market(asset, ws)
                if m:
                    up, down = feeds.top_of_book(self.pf.orderbook(m["id"]))
                    return up, down, "predict.fun", m
            except Exception as e:
                print(f"  [{asset}] predict.fun error: {e}; using simulated market")
        up, down = simulated_market(bars, k, left)
        return up, down, "simulated", None

    # ------------------------------------------------------------ main loop
    def settle_due(self, now: float):
        still = []
        for p in self.state["open"]:
            end = p["window_start"] + feeds.WINDOW
            if now < end + 3:
                still.append(p)
                continue
            final = feeds.binance_price_at(p["asset"], end)
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
            row = {**p, "final_price": final, "won": won, "pnl": pnl,
                   "bankroll_after": self.state["bankroll"]}
            self._log_trade(row)
            print(f"  SETTLED {p['asset']} {p['side']} @ {p['price']:.2f}  "
                  f"{'WIN ' if won else 'LOSS'} {pnl:+.2f}  bankroll {self.state['bankroll']:.2f}")
        self.state["open"] = still

    def tick(self):
        now = time.time()
        self.settle_due(now)
        ws, we = feeds.window_bounds(now)
        left = we - now
        for asset in self.assets:
            if any(p["asset"] == asset and p["window_start"] == ws for p in self.state["open"]):
                continue
            try:
                bars = feeds.klines_to_bars(feeds.binance_klines(asset, "1s", 1000))
                market = None
                up = down = None
                src = "simulated"
                if self.pf.enabled and left <= self.cfg.max_seconds_left:
                    up, down, src, market = self._market_prices(asset, ws, bars, 0, left)
                k = self._strike(asset, ws, market)
                if k is None:
                    continue
                sig = predict(bars, k, left, self.params)
                if src == "simulated":
                    up, down = simulated_market(bars, k, left)
                d = decide(sig, up, down, self.state["bankroll"], self.cfg)
                f = sig.features
                print(f"{time.strftime('%H:%M:%S')} {asset} left {left:5.1f}s  spot {sig.spot:.2f}  "
                      f"beat {k:.2f}  P(up) {sig.p_up:.3f}  ask U/D {up}/{down} [{src}]  "
                      f"RSI {f['rsi']:.0f} flow {f['imbalance']:+.2f} score {f['score']:+.2f}"
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
        self._save()

    def summary(self) -> str:
        parts = [f"bankroll {self.state['bankroll']:.2f}"]
        for a, s in self.state["stats"].items():
            if s["n"]:
                parts.append(f"{a}: {s['n']} bets {s['wins']/s['n']:.0%} win pnl {s['pnl']:+.2f}")
        return " | ".join(parts)

    def run(self, poll: float = 3.0):
        print(f"Paper trading {', '.join(self.assets)} | predict.fun orderbook: "
              f"{'ON' if self.pf.enabled else 'OFF (set PREDICT_API_KEY) - simulated market'}")
        last = 0.0
        while True:
            self.tick()
            if time.time() - last > 60:
                print("  >> " + self.summary())
                last = time.time()
            time.sleep(poll)
