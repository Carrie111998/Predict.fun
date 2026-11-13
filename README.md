# updown: 5-minute BTC / ETH / BNB up-or-down predictor and paper trader

A bot for predict.fun's `btc-updown-5m`, `eth-updown-5m` and `bnb-updown-5m`
markets. It predicts whether price will close **at or above the price to beat**
(the price at the start of the 5-minute window), compares that with the
market's price, and **paper-trades** automatically when the odds are in your
favour. It never sends real orders.

Pure Python 3.10+ standard library, so there is nothing to install.

## Is there a trick to win every bet?

No. Nobody can predict 5-minute crypto moves reliably: early in a window the
true odds are close to 50/50. If a "win every time" trick existed, market makers
would already have priced it in. Any tool that promises it is selling you something.

There are real edges, but they are small:

1. **Timing beats direction.** With 30–90 seconds left and price already away
   from the price to beat, the outcome is mostly decided. The maths below
   gives the true probability. You profit only when the market's price is
   **below** that probability (after fees). Example: model says 92% Up and the
   Up ask is 0.85, so you have about 7% edge. With an ask of 0.95 there is none, even though Up is likely.
2. **Price matters more than winning.** If you buy at 0.90 you must win more
   than 90% of the time just to break even. One loss wipes out about nine wins.
   That's why the bot caps entry at 0.93 and requires a minimum edge.
3. **Indicators give a small nudge, not a crystal ball.** EMA trend,
   order-flow imbalance (aggressive buys vs. sells), momentum and RSI
   extremes tilt the probability slightly. In practice they're worth a few
   percent at most.
4. **Size small.** The bot uses quarter-Kelly sizing capped at 5% of bankroll,
   so a losing streak doesn't wipe you out.
5. **Latency and oracle risk.** Markets resolve on an oracle feed (similar
   markets elsewhere use Chainlink, sometimes a 60-second average). This bot
   uses Binance spot as a proxy, so in the last few seconds the two can
   disagree. The bot won't trade with fewer than 8 seconds left.
   **Check the resolution rules on the market page** and adjust if they use a TWAP.

## The model

```
P(Up) = Φ( (ln(S/K) + μ·t) / (σ·√t) )
```
`S` = spot, `K` = price to beat, `t` = seconds left, `σ` = recent 1-second
volatility (max of realized and EWMA, inflated 15% for fat tails), `μ` = a
small drift from the indicator score.

The bot bets when `P(side) − ask − fee ≥ min_edge` and `P(side) ≥ min_prob`,
and only inside the entry window.

## Usage

```bash
# 1. Current prediction for each asset
python -m updown signal

# 2. Backtest on the last 24h of real 1-second Binance data:
#    calibration table + simulated P&L
python -m updown backtest --hours 24

# 3. Live auto paper trading (logs to paper_data/paper_trades.csv)
export PREDICT_API_KEY=...        # optional: real predict.fun orderbook
python -m updown paper

# Check predict.fun API access / market lookup
python -m updown probe
```

Useful options (all commands): `--min-edge 0.04 --min-prob 0.60 --max-left 150
--min-left 8 --max-price 0.93 --fee 0.01 --kelly 0.25 --momentum 0.15 --bankroll 1000`.

### Without a predict.fun API key
The bot simulates an efficient market: same maths, no indicator tilt, plus a
2¢ spread. It's an honest test, because any profit then comes only from your
indicators. Expect roughly break-even or small losses. For real paper trading,
get an API key at [dev.predict.fun](https://dev.predict.fun/) so it reads real
ask prices.

### Reading the backtest calibration table
Each cell shows: when the model favoured a side with that confidence, how often
that side actually won. If the 80–90% column shows about 85%, the model is
calibrated. Improve the model by changing settings until the columns line up,
then trade only where the market's price is below the actual win rate.

## Data sources
- **Binance public REST** (`api.binance.com`, mirror `data-api.binance.vision`):
  1-second klines with taker-buy volume, for price, volatility and indicators. No key needed.
- **predict.fun REST** (`https://api.predict.fun`, `x-api-key` header):
  `/v1/markets`, `/v1/categories/{slug}`, `/v1/markets/{id}/orderbook`
  (bids/asks are the Yes/Up side; Down ask = 1 − best bid). There's also a
  WebSocket at `wss://ws.predict.fun/ws` (topic `predictOrderbook/<marketId>`)
  for lower latency later.

> The predict.fun JSON parsing follows the public docs but could not be tested
> against the live API when this was built. Run `python -m updown probe` first.
> If the market lookup or the price-to-beat field differs, adjust
> `updown/feeds.py` (`find_current_market`) and `updown/paper.py` (`_strike`).

## Tests
```bash
python -m unittest discover -s tests -t .
```

*Educational tool. Not financial advice. Prediction markets can lose you money.*
