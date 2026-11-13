# predict.fun 5-minute predictor (BTC, ETH, BNB)

Prediction only: it never places orders. Open **http://localhost:8790** after starting it.

## Start
Double-click `start.cmd` (keeps running and restarts itself if it stops). Leave the window open.

## What it shows, per coin
- **Current round:** chance of Up (price vs the round's open, scaled by recent volatility and time left),
  the Predict.fun price for Up and Down, the most the side is worth, and a plain suggestion:
  `BUY <side> at X¢ or less – stake $S to win about $0.25`, or why to skip.
- **Next round:** market price and a momentum lean, with its measured past accuracy (about 48–51% –
  essentially a coin flip, so the tool says so).
- **Scoreboard:** every prediction at 1:00, 2:00, 3:00 and 4:00 into a round is logged to
  `data/predictions.jsonl` and checked against the result, showing accuracy and profit per $1 at each checkpoint.

## Target profit
Set "Target profit per win" (default $0.25). Stake for a target: `stake = target × price ÷ (1 − price)`.
Example: at 80¢, a $1.00 stake wins $0.25; at 90¢ you would need $2.25 to win $0.25.

## What the research says (and what we measured on your own bets)
- High win rates do not mean profit: at 85¢ you must be right about 85% of the time just to break even.
  A public test of 176 bot trades on Polymarket's 15-min markets had a 92% win rate and made about $2 in total,
  because wins were ~$0.15 and losses ~$12.
- Obvious moves are already priced in; the edge is buying only when the market price is below the
  model's value. The tool applies a 3-point safety margin for fees, slippage and model error.
- "Buy both sides for under $1" arbitrage, the one consistently profitable bot strategy reported, does not
  exist on Predict.fun: the Down price mirrors the Up price, so both sides always cost more than $1 together.
- Predict.fun settles on Chainlink prices; this tool uses Binance. They are very close, but near-flat rounds can differ.
