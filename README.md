# Momentum Factor Optimization

Finominal Quant Challenge, Junior Quantitative Analyst.

A monthly-rebalanced, market-cap-weighted momentum strategy on 30 Indian stocks. The lookback
period is chosen from 21, 63, 126, 189 and 252 trading days to maximise the annualised Sharpe ratio.

**Contents:** [Results](#results) | [Running it](#running-it) | [Approach](#approach) | [Catches](#catches-and-how-they-are-handled) | [The strategy](#the-strategy) | [The code](#the-code) | [Assumptions](#assumptions) | [Data quality](#data-quality) | [All lookbacks](#all-lookbacks-best-sharpe-first) | [Critical assessment](#critical-assessment) | [Troubleshooting](#troubleshooting)

## Results

The optimal lookback is **189 days**, on both the data as given and a cleaned version of it.

| Metric | Raw data (as given) | Cleaned data |
|---|---|---|
| **Optimal lookback (days)** | **189** | **189** |
| **Maximum annualised Sharpe** | **1.4532** | **0.9829** |
| **Cumulative return (%)** | **187,313,822.26** | **8,396.77** |
| **CAGR (%)** | **102.49** | **24.23** |

The raw numbers are not believable. The price file contains bad ticks that inflate returns (see
[Data quality](#data-quality)), so the cleaned column is the realistic answer and the raw column is
what the spec produces on the file exactly as supplied. Both are reported.

Backtest window: first trading day of 2006 to 2026-06-24, giving 5,083 daily returns and
246 monthly rebalances.

## Running it

```bash
pip install pandas numpy

python final_strategy.py --data-dir ./data               # runs BOTH raw and cleaned data (default)
python final_strategy.py --data-dir ./data --mode raw    # only the data as given
python final_strategy.py --data-dir ./data --mode clean  # only the cleaned data
```

`./data` must contain `Test_price_data.csv` and `Test_mcap_data.csv`. A run takes a few seconds.

For each dataset the script prints a table of all five lookbacks (best Sharpe on top) and the four
final numbers, then ends with a raw vs clean summary. It writes these files to the current folder:

| File | Content |
|---|---|
| `results_raw.csv`, `results_clean.csv` | Sharpe, cumulative return and CAGR for each lookback, sorted by Sharpe |
| `rebalance_log_raw.csv`, `rebalance_log_clean.csv` | For the winning lookback: each rebalance date, its calculation date, and the 10 holdings with weights |

## Approach

1. Read both files and reviewed the data before writing any strategy code.
2. Implemented the rules from the spec exactly, as a day-by-day loop from the first trading day of 2006.
3. Ran all five lookbacks and ranked them by Sharpe ratio.
4. The raw results looked impossible (102% CAGR for 20 years), so I investigated and found bad ticks
   in the price file. I now run the strategy on both the raw and the cleaned data.
5. Checked the output: every rebalance holds 10 stocks, the maximum weight is 20.0%, weights sum to 100%.
6. Tried wider searches (finer lookbacks, other settings, walk-forward selection) to see whether the
   answer is robust. It is not much better than 189 days, so I kept the spec's setup.

## Catches and how they are handled

These are the places where the task is easy to get wrong.

| # | Catch | Why it matters | How it is handled |
|---|---|---|---|
| 1 | **Bad price data.** About 5,000 one-day spikes that reverse the next day (e.g. 21.98, 109.75, 22.54). | Momentum rewards big rises, so the strategy "buys" fake spikes. Raw Sharpe is 1.45 and CAGR 102%, against 0.98 and 24% once cleaned. | Run both raw (the spec on the file as given) and cleaned (`remove_spikes`). The best lookback is 189 either way, and the difference is documented. |
| 2 | **Look-ahead bias.** The signal must not use the rebalance day's own price. | Using information you would not yet have inflates results. | Momentum and market caps are read from the calculation date (the previous trading day, `calc = i - 1`). Trades happen at the rebalance day's close, and returns start the next day. |
| 3 | **The 20% cap needs iteration.** | Cutting one stock to 20% and redistributing the excess can push other stocks over 20%. A single pass leaves illegal weights. | `cap_weights` repeats capping and redistribution (pro-rata to market cap) until no weight exceeds 20% and the total is 100%. |
| 4 | **Order of the buffer rule.** | The spec says evaluate holdings, exit if rank > 12, keep if rank <= 12, then fill with the best non-held stocks. Doing the steps in a different order gives different portfolios. | `choose_stocks` applies the steps in exactly that order and always ends with exactly 10 stocks. On the first rebalance nothing is held, so the top 10 are bought. |
| 5 | **Two different date formats.** Prices are dd-mm-yyyy, market caps are yyyy-mm-dd. | A default parser silently mixes up days and months. | `load_data` parses each file with an explicit format and asserts that the dates and stocks of both files match. |
| 6 | **Details the spec leaves open.** | Small choices move the numbers. The CAGR year count alone changes raw CAGR from 102.49% to 104.63%. | Each choice is stated in [Assumptions](#assumptions), and both CAGR conventions are printed. |
| 7 | **The "optimum" is weak.** | On raw data 189 days has a Sharpe of 1.4532 and 252 days has 1.4523, practically a tie. Picking the best of five on one history is mild data mining. | Reported honestly in the critical assessment. Finer grids and walk-forward selection did not reliably beat 189 days, so the spec's setup is kept. |

## The strategy

On the **first trading day of each month**, trading at that day's close:

1. **Dates.** The calculation date is the previous trading day. All inputs (prices and market caps)
   come from that day, so there is no look-ahead.
2. **Momentum.** `M = price(calc date) / price(calc date - L trading days) - 1`. Rank all stocks
   from highest M (rank 1) to lowest.
3. **Selection.** Hold exactly 10 stocks. A current holding is kept if its new rank is 12 or
   better and sold if worse. Empty slots are filled with the best-ranked stocks not held. The
   buffer avoids selling a stock that has only slipped slightly.
4. **Weights.** Market-cap weights, with a 20% cap. Any stock above 20% is set to 20% and the excess
   is shared among the uncapped stocks in proportion to market cap. This repeats until no stock is
   above 20%, because redistribution can push others over the limit.
5. **Holding.** Weights drift with prices until the next rebalance (buy and hold).

Cap example (market caps 50, 20, 15, 5, 4, 2, 1, 1, 1, 1): the 50% stock is cut to 20%, which lifts
two others to 32% and 24%, so they are capped too. The final weights are 20%, 20%, 20%, 13.3%, 10.7%,
5.3% and 2.7% x 4.

**Metrics** (daily returns, risk-free rate 0%):
- Sharpe = mean daily return / sample standard deviation of daily returns x sqrt(252)
- Cumulative return = product of (1 + daily return) - 1
- CAGR = (1 + cumulative return)^(1 / years) - 1, with years = calendar days / 365.25

## The code

Everything is in `final_strategy.py`, with a comment on each line.

| Function | What it does |
|---|---|
| `load_data` | Reads both CSVs, parses their different date formats, checks that dates and stocks match |
| `remove_spikes` | Replaces one-day bad ticks (more than 25% from the 5-day median) with the last good value |
| `cap_weights` | Market-cap weights with the iterative 20% cap |
| `choose_stocks` | Ranking, exit rule (rank > 12) and filling to exactly 10 stocks |
| `run_backtest` | Day-by-day simulation for one lookback |
| `performance` | Sharpe, cumulative return and CAGR |
| `run_mode`, `main` | Test every lookback, sort by Sharpe, print and save results |

## Assumptions

The spec leaves some details open:

- **First return.** The portfolio is built at the close of the first trading day of 2006, so the
  first return is earned the next day. Including a zero-return first day changes nothing meaningful.
- **CAGR years.** Calendar days / 365.25. The alternative (trading days / 252) gives 104.63% on raw
  data and 24.64% on cleaned data, and the script prints both.
- **Drift and costs.** Weights drift between rebalances. There are no transaction costs or slippage.
- **Ties** in momentum are broken by column order.
- **Missing data** (none in the supplied files): a stock without a price or market cap on the
  calculation date cannot be selected, and missing daily returns count as zero.

## Data quality

The price file has about 5,000 one-day spikes that reverse the next day. For example, `GICH.NS`
goes 21.98, then 109.75, then 22.54 around 31 December 2008. Average daily volatility per stock is
about 44%, and a momentum strategy buys these fake jumps, which produces the 100%+ CAGR.

The clean run replaces any value more than 25% from its centred 5-day median with the previous good
value, in both prices and market caps. Daily volatility drops to about 3% and CAGR to 24%. This is a
simple filter, not a proper data pipeline, and it could remove a genuine large move. The best
lookback is 189 either way.

## All lookbacks, best Sharpe first

Raw data:

| Lookback | Sharpe | Cumulative return (%) | CAGR (%) |
|---|---|---|---|
| **189** | **1.4532** | **187,313,822.26** | **102.49** |
| 252 | 1.4523 | 134,489,606.03 | 99.24 |
| 126 | 1.3972 | 5,424,022.83 | 70.32 |
| 63 | 1.3946 | 4,109,999.03 | 68.03 |
| 21 | 1.3530 | 1,874,179.92 | 61.71 |

Cleaned data:

| Lookback | Sharpe | Cumulative return (%) | CAGR (%) |
|---|---|---|---|
| **189** | **0.9829** | **8,396.77** | **24.23** |
| 126 | 0.9353 | 6,131.16 | 22.37 |
| 252 | 0.9015 | 5,282.79 | 21.50 |
| 63 | 0.8716 | 4,411.71 | 20.45 |
| 21 | 0.8317 | 3,481.61 | 19.10 |

Checks on the rebalance logs: all 246 rebalances hold exactly 10 stocks, the maximum weight is 20.0%
and weights sum to 100%. About 1.4 names are replaced per month on cleaned data (1.9 on raw).

## Critical assessment

- **The optimum is weak.** The Sharpe gaps between lookbacks are small, and 189 vs 252 days is almost
  a tie on raw data. Picking the best of five on one 20-year history is mild data mining. Finer
  lookback grids, blended signals and walk-forward selection did not reliably beat plain 189 days.
- **Survivorship bias.** The 30 stocks all exist in 2026, which flatters the long-run result.
- **No trading costs.** Replacing one to two of ten names every month, in small and mid-cap Indian
  stocks, would reduce real returns.
- **Concentration.** Ten stocks with a 20% cap means high single-stock risk, and momentum is prone
  to sharp crashes when markets reverse.
- **Data sensitivity.** Sharpe falls from 1.45 to 0.98 just from removing bad ticks, so data
  validation matters more than parameter tuning.

With more time: a point-in-time universe including delisted stocks, realistic costs and a liquidity
filter, out-of-sample validation of the lookback, and a proper data-cleaning pipeline.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` | `--data-dir` must point to the folder holding both CSV files |
| Date parsing error | Prices must be dd-mm-yyyy and market caps yyyy-mm-dd, or edit the two formats in `load_data` |
| `AssertionError` on dates or stocks | The two files cover different dates or tickers |
| Cumulative return in the millions of % | Expected on raw data because of bad ticks; use the clean results |
