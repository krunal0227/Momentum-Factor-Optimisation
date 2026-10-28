"""
Finominal Quant Challenge - Momentum Factor Optimization
=========================================================
Every month (first trading day) the strategy:
  1. ranks the 30 stocks by past return over a lookback window ("momentum"),
  2. holds 10 stocks: current holdings stay if still ranked 12 or better, empty slots are
     filled with the best-ranked stocks not held,
  3. weights them by market cap, with no stock above 20%.
It tests lookbacks of 21/63/126/189/252 days and reports the one with the highest Sharpe ratio.

Run:  python final_strategy.py --data-dir DATA_FOLDER
      Tests BOTH the data as given ("raw") and the data with bad price spikes removed ("clean").
      Use --mode raw or --mode clean to run only one.
"""
import argparse
import numpy as np
import pandas as pd

LOOKBACKS = [21, 63, 126, 189, 252]   # lookback windows to test, in trading days
N_STOCKS = 10                         # stocks held at all times
KEEP_RANK = 12                        # keep a held stock only if its new rank is 12 or better
MAX_WEIGHT = 0.20                     # no stock may exceed a 20% weight
START_YEAR = 2006                     # backtest starts on the first trading day of 2006


def load_data(folder):
    """Read prices and market caps (rows = dates, columns = stocks)."""
    prices = pd.read_csv(f"{folder}/Test_price_data.csv")
    mcaps = pd.read_csv(f"{folder}/Test_mcap_data.csv")
    prices["date"] = pd.to_datetime(prices["date"], format="%d-%m-%Y")   # prices: 31-12-2005
    mcaps["date"] = pd.to_datetime(mcaps["date"], format="%Y-%m-%d")     # mcaps: 2005-12-31
    prices = prices.set_index("date").sort_index()
    mcaps = mcaps.set_index("date").sort_index()
    assert prices.index.equals(mcaps.index), "price and mcap dates do not match"
    assert list(prices.columns) == list(mcaps.columns), "price and mcap stocks do not match"
    return prices, mcaps


def remove_spikes(df, threshold=0.25):
    """Optional cleaning: a value more than 25% away from its 5-day median is a bad tick,
    so replace it with the previous good value."""
    df = df.copy()
    for _ in range(5):                                                   # repeat to catch neighbouring spikes
        median = df.rolling(5, center=True, min_periods=3).median()      # typical value around each day
        df = df.mask((df / median - 1).abs() > threshold).ffill().bfill()  # blank out spikes, refill
    return df


def cap_weights(mcap):
    """Market-cap weights capped at 20%. Excess from capped stocks goes to the uncapped ones
    in proportion to market cap; repeat until nobody is above 20%."""
    weights = mcap / mcap.sum()                                  # start with plain market-cap weights
    capped = np.zeros(len(weights), dtype=bool)                  # which stocks are already capped
    while True:
        newly_capped = (weights > MAX_WEIGHT + 1e-12) & ~capped  # stocks still above 20%
        if not newly_capped.any():                               # nobody above 20% -> finished
            return weights
        capped |= newly_capped
        weights[capped] = MAX_WEIGHT                             # capped stocks get exactly 20%
        room = 1 - weights[capped].sum()                         # weight left for the others
        weights[~capped] = room * mcap[~capped] / mcap[~capped].sum()   # share it by market cap


def choose_stocks(momentum, held):
    """Pick exactly 10 stocks: keep holdings ranked <= 12, fill the rest with the best-ranked others."""
    order = np.argsort(-momentum, kind="stable")                 # stock indices, best momentum first
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)                   # rank[s] = 1 for the best stock
    usable = ~np.isnan(momentum)                                 # ignore stocks without data
    chosen = [s for s in held if usable[s] and rank[s] <= KEEP_RANK]   # exit rule: drop if rank > 12
    for s in order:                                              # fill vacancies, best first
        if len(chosen) == N_STOCKS:
            break
        if usable[s] and s not in chosen:
            chosen.append(s)
    return chosen


def run_backtest(prices, mcaps, lookback):
    """Simulate one lookback. Returns daily portfolio returns and a rebalance log."""
    dates, names = prices.index, list(prices.columns)
    P, M = prices.values, mcaps.values                           # numpy arrays (faster than pandas)
    stock_ret = np.vstack([np.zeros(P.shape[1]), P[1:] / P[:-1] - 1])   # each stock's daily return
    stock_ret = np.nan_to_num(stock_ret)                         # missing data -> 0 return

    in_backtest = dates.year >= START_YEAR
    first_of_month = ~dates.to_period("M").duplicated()          # True on the first date of each month
    rebalance_rows = set(np.where(first_of_month & in_backtest)[0])    # rows where we rebalance
    start = int(np.where(in_backtest)[0][0])                     # first trading day of 2006

    held, weights = [], None                                     # no portfolio yet
    returns, return_dates, log = [], [], []
    for i in range(start, len(dates)):
        if weights is not None:                                  # we own a portfolio: earn today's return
            r = float(stock_ret[i] @ weights)                    # sum of weight x stock return
            returns.append(r)
            return_dates.append(dates[i])
            weights = weights * (1 + stock_ret[i]) / (1 + r)     # weights drift with prices (buy and hold)

        if i in rebalance_rows:                                  # first trading day of the month: trade at close
            calc = i - 1                                         # T_calc = previous trading day
            momentum = P[calc] / P[calc - lookback] - 1          # return over the lookback window
            momentum[np.isnan(M[calc])] = np.nan                 # no market cap -> not usable
            held = choose_stocks(momentum, held)                 # selection with the rank-12 buffer
            weights = np.zeros(P.shape[1])                       # 0% everywhere...
            weights[held] = cap_weights(M[calc, held])           # ...except the 10 held (capped mcap weights)
            log.append({"rebalance_date": dates[i].date(), "calc_date": dates[calc].date(),
                        "holdings": ", ".join(f"{names[s]} {weights[s]:.1%}" for s in held)})

    return pd.Series(returns, index=return_dates), pd.DataFrame(log)


def performance(returns):
    """Sharpe, cumulative return and CAGR from daily returns (risk-free rate = 0)."""
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)       # annualised Sharpe, sample std
    cumulative = (1 + returns).prod() - 1                              # total compounded return
    years = (returns.index[-1] - returns.index[0]).days / 365.25       # backtest length in calendar years
    cagr = (1 + cumulative) ** (1 / years) - 1                         # compound annual growth rate
    cagr_252 = (1 + cumulative) ** (252 / len(returns)) - 1            # alternative: 252 trading days per year
    return sharpe, cumulative, cagr, cagr_252


def run_mode(prices, mcaps, mode):
    """Test every lookback on this data, print and save the table, return the best row."""
    print(f"===== {mode.upper()} DATA: {prices.index[0].date()} to {prices.index[-1].date()}, "
          f"{prices.shape[1]} stocks =====")
    rows, logs = [], {}
    for lookback in LOOKBACKS:                                   # each lookback is tested independently
        returns, logs[lookback] = run_backtest(prices, mcaps, lookback)
        sharpe, cumulative, cagr, cagr_252 = performance(returns)
        rows.append({"Lookback": lookback, "Sharpe": sharpe, "CumReturn_%": cumulative * 100,
                     "CAGR_%": cagr * 100, "CAGR_252day_%": cagr_252 * 100})

    table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)  # best on top
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    table.to_csv(f"results_{mode}.csv", index=False)

    best = table.iloc[0]                                         # highest Sharpe
    logs[int(best["Lookback"])].to_csv(f"rebalance_log_{mode}.csv", index=False)
    print(f"\n--- {mode.upper()} RESULT ---")
    print(f"Optimal lookback (days) : {int(best['Lookback'])}")
    print(f"Max annualised Sharpe   : {best['Sharpe']:.4f}")
    print(f"Cumulative return (%)   : {best['CumReturn_%']:,.2f}")
    print(f"CAGR (%)                : {best['CAGR_%']:.2f}   (252-day-year convention: {best['CAGR_252day_%']:.2f})\n")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="folder containing the two CSV files")
    parser.add_argument("--mode", choices=["both", "raw", "clean"], default="both",
                        help="raw = data as given, clean = spikes removed, both = run each (default)")
    args = parser.parse_args()

    prices, mcaps = load_data(args.data_dir)
    datasets = {"raw": (prices, mcaps), "clean": (remove_spikes(prices), remove_spikes(mcaps))}
    modes = ["raw", "clean"] if args.mode == "both" else [args.mode]
    best = {m: run_mode(*datasets[m], m) for m in modes}

    if len(best) == 2:                                           # side-by-side summary
        print("===== SUMMARY (raw vs clean) =====")
        print(f"{'':26}{'RAW':>22}{'CLEAN':>16}")
        for label, key, fmt in [("Optimal lookback (days)", "Lookback", "{:.0f}"),
                                ("Max annualised Sharpe", "Sharpe", "{:.4f}"),
                                ("Cumulative return (%)", "CumReturn_%", "{:,.2f}"),
                                ("CAGR (%)", "CAGR_%", "{:.2f}")]:
            print(f"{label:26}{fmt.format(best['raw'][key]):>22}{fmt.format(best['clean'][key]):>16}")


if __name__ == "__main__":
    main()
