"""CLI:  python -m updown {paper,backtest,signal,probe} [options]"""
from __future__ import annotations

import argparse
import json
import time

from . import backtest, feeds
from .model import ModelParams, predict
from .paper import PaperTrader
from .strategy import StrategyConfig


def main(argv=None):
    ap = argparse.ArgumentParser(prog="updown", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--assets", default="BTC,ETH,BNB")
        p.add_argument("--min-edge", type=float, default=0.04)
        p.add_argument("--min-prob", type=float, default=0.60)
        p.add_argument("--max-left", type=float, default=150, help="earliest entry (seconds left)")
        p.add_argument("--min-left", type=float, default=8, help="latest entry (seconds left)")
        p.add_argument("--max-price", type=float, default=0.93)
        p.add_argument("--fee", type=float, default=0.01, help="fee as fraction of stake")
        p.add_argument("--kelly", type=float, default=0.25)
        p.add_argument("--bankroll", type=float, default=1000.0)
        p.add_argument("--momentum", type=float, default=0.15, help="indicator weight (0 = pure maths)")

    p = sub.add_parser("paper", help="run the live auto paper trader")
    common(p)
    p.add_argument("--poll", type=float, default=3.0)
    p.add_argument("--out", default="paper_data")

    d = sub.add_parser("dashboard", help="run the paper trader with a live web dashboard")
    common(d)
    d.add_argument("--port", type=int, default=8050)
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--poll", type=float, default=2.0)
    d.add_argument("--out", default=None, help="data folder (default paper_data, or demo_data with --demo)")
    d.add_argument("--demo", action="store_true", help="offline simulated prices and market")
    d.add_argument("--speed", type=float, default=1.0, help="demo clock speed (e.g. 10 = 10x)")

    b = sub.add_parser("backtest", help="calibration + simulated P&L on Binance history")
    common(b)
    b.add_argument("--hours", type=float, default=6)
    b.add_argument("--synthetic", action="store_true", help="offline random-walk data")

    s = sub.add_parser("signal", help="print the current prediction once")
    s.add_argument("--assets", default="BTC,ETH,BNB")

    sub.add_parser("probe", help="check predict.fun API access and dump the current markets")

    a = ap.parse_args(argv)

    if a.cmd == "probe":
        pf = feeds.PredictFun()
        if not pf.enabled:
            print("PREDICT_API_KEY is not set")
            return
        ws, _ = feeds.window_bounds()
        for asset in ("BTC", "ETH", "BNB"):
            m = pf.find_current_market(asset, ws)
            print(f"--- {asset}:", json.dumps(m, indent=1)[:2000] if m else "not found")
            if m:
                print("book:", feeds.top_of_book(pf.orderbook(m["id"])))
        return

    assets = [x.strip().upper() for x in a.assets.split(",") if x.strip()]

    if a.cmd == "signal":
        ws, we = feeds.window_bounds()
        for asset in assets:
            bars = feeds.klines_to_bars(feeds.binance_klines(asset, "1s", 1000))
            k = feeds.binance_price_at(asset, ws)
            sig = predict(bars, k, we - time.time())
            print(f"{asset}: spot {sig.spot:.2f} beat {k:.2f} left {sig.seconds_left:.0f}s "
                  f"P(up)={sig.p_up:.3f} -> {sig.side}  features "
                  + " ".join(f"{n}={v:.4g}" for n, v in sig.features.items()))
        return

    cfg = StrategyConfig(max_seconds_left=a.max_left, min_seconds_left=a.min_left,
                         min_edge=a.min_edge, min_prob=a.min_prob, max_price=a.max_price,
                         fee_rate=a.fee, kelly_fraction=a.kelly)
    params = ModelParams(momentum_weight=a.momentum)

    if a.cmd == "backtest":
        for asset in assets:
            series = (backtest.synthetic_history(a.hours) if a.synthetic
                      else backtest.fetch_history(asset, a.hours))
            rep = backtest.run(series, cfg, params, a.bankroll)
            print(rep.text(asset + (" (synthetic)" if a.synthetic else "")))
            print()
        return

    if a.cmd == "dashboard":
        from .dashboard import serve
        feed = clock = None
        if a.demo:
            from .demo import DemoClock, DemoFeed
            clock = DemoClock(a.speed)
            feed = DemoFeed(clock)
        out = a.out or ("demo_data" if a.demo else "paper_data")
        if a.demo:  # a demo always starts fresh
            import shutil
            shutil.rmtree(out, ignore_errors=True)
        trader = PaperTrader(assets, cfg, params, a.bankroll, out, feed=feed,
                             **({"clock": clock} if clock else {}))
        serve(trader, a.host, a.port, a.poll)
        return

    PaperTrader(assets, cfg, params, a.bankroll, a.out).run(a.poll)


if __name__ == "__main__":
    main()
