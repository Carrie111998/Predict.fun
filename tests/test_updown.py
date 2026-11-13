import math
import tempfile
import unittest
from unittest import mock

from updown import backtest, feeds, indicators as ind
from updown.model import Bars, predict
from updown.paper import PaperTrader
from updown.model import ModelParams
from updown.strategy import Decision, StrategyConfig, decide, settle


def flat_bars(price=100.0, n=1000, drift=0.0):
    closes = [price * math.exp(drift * i + 1e-4 * math.sin(i)) for i in range(n)]
    return Bars(closes, [1.0] * n, [0.5] * n, 1.0)


class Indicators(unittest.TestCase):
    def test_rsi_extremes(self):
        self.assertGreater(ind.rsi(list(range(1, 50))), 99)
        self.assertLess(ind.rsi(list(range(50, 1, -1))), 1)

    def test_imbalance(self):
        self.assertAlmostEqual(ind.taker_imbalance([1, 1], [1, 1]), 1.0)
        self.assertAlmostEqual(ind.taker_imbalance([1, 1], [0.5, 0.5]), 0.0)


class Model(unittest.TestCase):
    def test_probability_moves_with_distance_and_time(self):
        b = flat_bars()
        spot = b.closes[-1]
        at_strike = predict(b, spot, 120).p_up
        above_late = predict(b, spot * 0.999, 10).p_up
        above_early = predict(b, spot * 0.999, 290).p_up
        self.assertAlmostEqual(at_strike, 0.5, delta=0.05)
        self.assertGreater(above_late, above_early)
        self.assertGreater(above_early, 0.5)
        self.assertLess(predict(b, spot * 1.001, 10).p_up, 0.1)


class Strategy(unittest.TestCase):
    def test_decide_and_settle(self):
        b = flat_bars()
        sig = predict(b, b.closes[-1] * 0.999, 20)
        cfg = StrategyConfig()
        self.assertIsNone(decide(sig, 0.99, 0.99, 1000, cfg))          # too expensive
        d = decide(sig, 0.70, 0.40, 1000, cfg)
        self.assertEqual(d.side, "UP")
        self.assertLessEqual(d.stake, 50)
        won, pnl = settle(d, 100, 101, 0.0)
        self.assertTrue(won)
        self.assertAlmostEqual(pnl, d.shares - d.stake, places=3)
        won, pnl = settle(Decision("DOWN", 0.5, 0.6, 0.1, 10), 100, 100, 0.0)
        self.assertFalse(won)  # tie resolves Up
        self.assertEqual(pnl, -10)

    def test_backtest_runs(self):
        rep = backtest.run(backtest.synthetic_history(2))
        self.assertGreater(rep.windows, 15)
        self.assertIn("Calibration", rep.text("BTC"))


class Paper(unittest.TestCase):
    def test_tick_bets_and_settles(self):
        rows = [[i * 1000, "100", "100", "100", "100.2", "1", 0, 0, 0, "0.8"] for i in range(1000)]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(feeds, "binance_klines", return_value=rows), \
                mock.patch.object(feeds, "binance_price_at", side_effect=[100.0, 100.3]), \
                mock.patch.object(feeds.time, "time", return_value=999_999_900 + 280), \
                mock.patch("updown.paper.time.time", return_value=999_999_900 + 280), \
                mock.patch("updown.paper.simulated_market", return_value=(0.80, 0.25)):
            pt = PaperTrader(["BTC"], StrategyConfig(), ModelParams(), 1000, d)
            pt.pf.key = None
            pt.tick()
            self.assertEqual(len(pt.state["open"]), 1)
            self.assertEqual(pt.state["open"][0]["side"], "UP")
            pt.settle_due(999_999_900 + 400)
            self.assertEqual(pt.state["open"], [])
            self.assertGreater(pt.state["bankroll"], 1000)
            self.assertIn("BTC", pt.summary())


if __name__ == "__main__":
    unittest.main()
