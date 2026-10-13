# US v5 and India v8 on US data, 2000 to October 2026

Backtest study, 13 October 2026. The live system is unchanged; everything here is in
`usbt/` on branch `claude/usa-india-backtest-optimization-fr5gs8`.

## Answer

**No combination beats SPY, QQQ and SMH in every year, from every start date.** No variant came close
once tested on years it was not tuned on. The years that break every version are the ones where
semiconductors ran far ahead: SMH returned +64% in 2019, +73% in 2023 and +39% in 2024. A
diversified stock book, even a concentrated momentum one, does not keep up with that.

Three things did improve the US system, and they hold in the 2000-2011 test as well:

| | 2012-18 (train) | 2019-26 (test) | 2000-11 (test, biased) |
|---|---|---|---|
| **US v5 as live** | 13.4% a year, worst fall -24% | 29.7%, -39% | 1.6%, -60% |
| **Candidate 3** (chosen on 2012-18 only) | 14.3%, -18% | 27.8%, -30% | 12.9%, -27% |
| **Candidate 3 + v5 adds** (seen after the test) | 20.1%, -20% | 34.8%, -35% | 11.3%, -30% |
| **v5 + decide twice a month** (seen after the test) | 14.4%, -25% | 34.8%, -33% | 5.0%, -50% |
| SPY / QQQ / SMH worst fall | -19% / -23% / -27% | -34% / -35% / -45% | -55% / -83% / -84% |

Median of five start dates per window, $10,000, whole shares, every cost. "Worst fall" is the
deepest drawdown across the five paths.

**Candidate 3** is the only one picked strictly by the rules you set: chosen on 2012-2018 and then
tested. It is the US v5 stock engine with four India features:
- the 500 most-traded US stocks as the universe, instead of everything above $50m a day;
- India's buy pause (breadth, large-cap trend, HYG) instead of the regime overlay, so stocks are
  never sold to match a regime and the -12% sleeve exit and daily re-entry go away;
- decisions twice a month (the 11th and the last session) instead of daily;
- also selling a holding when it ranks outside the top 45, as well as on its stop.
It also takes no v5 adds and a stricter Recovery cutoff (0.95).

Against live v5 it gives up about 2 points a year in 2019-2026, cuts the worst fall from -39% to
-30% (smaller than every index's), and in 2000-2011 makes 12.9% a year instead of 1.6% with a worst
fall of -27% instead of -60%.

The other two rows were found by taking Candidate 3 apart after the test years were already seen,
so their test numbers are optimistic. Treat them as hypotheses for a forward test, not results.

## How it was tested

- **Data.** FMP dividend-adjusted daily prices, 1999 to 12 Oct 2026, for 1,505 stocks: every
  name the live store has ever used as a universe, plus every ticker in the S&P 500 since 1996
  (fja05680/sp500), with tickers now used by a different company removed. Splits from FMP so share
  counts and per-share commission use the price actually traded.
- **Survivorship.** FMP on this plan has almost no stocks delisted before about 2012 (Enron,
  Lehman, Bear Stearns, WaMu, Wyeth and others are missing). Share of each year's S&P 500 we
  can price: 52% in 2000, 75% in 2008, 84% in 2012, 95% in 2018, 97% in 2019, 99-100% from 2020
  (`results/coverage.csv`). So 2019-2026 is close to survivorship-free, 2012-2018 mostly, and
  2000-2011 flatters every system. Read 2000-2011 for direction only.
- **Account.** $10,000, whole shares for stocks and ETFs, orders at the next open, the framework's
  cost model on every fill ($0.005 a share + 10 bps, $1 minimum), no taxes. At this size the $1
  minimum is the biggest cost: about 4% a year for v5, which buys a little most days.
- **Windows.** Choose on 2012-2018, test on 2019-Oct 2026 and on 2000-2011. Each window is run
  from five start dates a quarter apart; full calendar years only for the year counts.
- **Engine check.** The study engine (`code/engine.py`) was checked against the live code itself
  (`code/live_harness.py` imports `v5_engine`, `v5_sleeve` and `v5_alloc` unchanged): 13.6% vs
  13.4% a year in 2012-18, 31.4% vs 30.2% in 2019-26, -2.3% vs -3.3% in 2000-11
  (`results/ref_live_vs_engine.csv`).

## The 15 differences, one at a time (2012-2018)

Each row changes one thing in live v5 (`results/exp1_switches.txt`).

| Change to v5 | CAGR | Worst fall | Beat all 3 |
|---|---|---|---|
| none (live v5) | 13.4% | -24% | 13% of years |
| hold 10 in equal slots | 15.3% | -31% | 16% |
| fully invested, no overlay | 9.7% | -42% | 19% |
| India buy pause instead of regimes | 14.5% | -30% | 29% |
| drop the -12% exit and daily re-entry | 13.5% | -24% | 16% |
| sell on rank instead of stops | 15.3% | -30% | 22% |
| no adds | 12.0% | -22% | 22% |
| India ranking | 7.9% | -30% | 3% |
| no pullback-band entry | 13.8% | -25% | 13% |
| decide twice a month | 14.4% | -25% | 16% |
| idle cash in GLD | 12.7% | -25% | 13% |
| top-500 universe | 15.9% | -24% | 16% |
| no ramp-in | 13.3% | -24% | 16% |
| 1% of daily volume order cap (never binds at $10,000) | 13.4% | -24% | 13% |
| tax-aware sell delay | not tested: no US tax modelled | | |

India v8 run as-is on US data: 16.2% (2012-18), 20.0% (2019-26), 13.6% (2000-11), with
worst falls of -22%, -49% and -61%. It does well in the 2000s and badly in 2021 (-10%) and in the
2020 and 2022 falls, because its pause stops buying but never sells.

## What the search found, and what it means

About 600 random combinations of both systems, then a local search, all scored on 2012-2018
only (`code/exp2_search.py`, `code/exp2b_local.py`). The best of them, "Hybrid A" (12 equal slots,
India ranking, pause, gold), beat all three indexes in 84% of training years with a -18% worst
fall. On 2019-2026 it beat them in 32% of years with a -51% worst fall, worse than live v5. That
is overfitting, and it is why the recommendation is the plainer Candidate 3, which ranked third in
training and held up out of sample.

Which parts of Candidate 3 matter (`results/exp4_ablation.csv`):
- deciding twice a month helps in every window; removing it costs 3 to 8 points a year;
- the buy pause matters most in 2000-2011 (without it the worst fall goes from -27% to -46%);
- the top-500 universe helps 2000-2011 but costs about 3 points in 2019-2026;
- putting v5 adds back raises returns in 2012-2026 but deepens the falls a little.

## Year by year (one run from Jan 2012 and one from Jan 2000)

| Year | v5 live | v5 twice-monthly | Cand. 3 | Cand. 3 + adds | SPY | QQQ | SMH |
|---|---|---|---|---|---|---|---|
| 2000 | -36.7 | -31.1 | -15.1 | -10.9 | -8.9 | -38.4 | -47.1 |
| 2001 | -11.1 | -7.3 | 4.9 | -0.4 | -11.7 | -33.3 | -14.6 |
| 2002 | -15.6 | -15.4 | -2.1 | -2.4 | -21.6 | -37.4 | -47.0 |
| 2003 | 35.0 | 29.7 | 60.7 | 58.5 | 28.2 | 49.7 | 87.3 |
| 2004 | -0.5 | 6.2 | 29.0 | 31.8 | 10.7 | 10.5 | -19.3 |
| 2005 | -0.0 | 4.2 | 9.9 | 7.6 | 4.8 | 1.6 | 10.8 |
| 2006 | -6.0 | 3.1 | 16.3 | 11.0 | 15.9 | 7.2 | -6.7 |
| 2007 | 14.8 | 38.1 | 43.5 | 33.2 | 5.1 | 19.0 | 6.0 |
| 2008 | -9.0 | -2.7 | -11.4 | -9.1 | -36.8 | -41.7 | -44.7 |
| 2009 | -1.1 | 16.1 | 26.8 | 22.4 | 26.4 | 54.7 | 60.5 |
| 2010 | 18.4 | 9.0 | 4.3 | -1.3 | 15.1 | 19.9 | 19.0 |
| 2011 | -8.4 | -1.6 | -15.0 | -9.9 | 1.9 | 3.4 | 6.8 |
| 2012 | 5.4 | 8.3 | 4.9 | 14.3 | 14.2 | 15.9 | 7.0 |
| 2013 | 42.5 | 50.1 | 39.5 | 64.9 | 32.3 | 36.6 | 33.4 |
| 2014 | 8.7 | 9.1 | 11.4 | 10.8 | 13.5 | 19.2 | 30.2 |
| 2015 | 2.2 | 4.7 | 4.4 | 4.8 | 1.2 | 9.4 | -0.3 |
| 2016 | 16.7 | 25.3 | 1.4 | 6.2 | 12.0 | 7.1 | 35.6 |
| 2017 | 33.2 | 15.8 | 44.8 | 38.9 | 21.7 | 32.7 | 38.5 |
| 2018 | -7.1 | -9.5 | -0.3 | 15.8 | -4.6 | -0.1 | -9.1 |
| 2019 | 29.0 | 21.0 | 21.9 | 18.5 | 31.2 | 39.0 | 64.4 |
| 2020 | 89.3 | 71.0 | 54.3 | 50.3 | 18.3 | 48.6 | 55.5 |
| 2021 | 56.2 | 30.5 | 33.4 | 21.5 | 28.7 | 27.4 | 42.1 |
| 2022 | -15.1 | -5.7 | -2.5 | -4.9 | -18.2 | -32.6 | -33.5 |
| 2023 | 34.1 | 25.7 | 23.5 | 30.2 | 26.2 | 54.9 | 73.4 |
| 2024 | 26.4 | 44.6 | 22.4 | 102.0 | 24.9 | 25.6 | 39.1 |
| 2025 | 8.1 | 12.9 | 54.6 | 46.1 | 17.7 | 20.8 | 49.2 |
| 2026 to 12 Oct | 62.2 | 63.9 | 45.3 | 54.9 | 15.0 | 21.9 | 64.1 |

Per cent. These are one path each; the five-start medians are in `results/exp3_periods.csv`.
The 2024 +102% for Candidate 3 + adds comes from real winners (Vertiv, XPO, CAVA, Coinbase).

## Start-date test (2019-2025)

A fresh $10,000 account every quarter from Jan 2019 to Oct 2025, held to 12 Oct 2026
(`results/exp3_fresh_starts.csv`):

| | Beat all 3 to today | Beat all 3 in first 12 months | Worst fall smaller than all 3 | Median CAGR |
|---|---|---|---|---|
| US v5 live | 0% | 32% | 7% | 29.6% |
| Candidate 3 | 4% | 25% | 18% | 30.4% |
| Candidate 3 + v5 adds | 29% | 21% | 0% | 37.6% |
| v5 + twice-monthly | 7% | 29% | 21% | 33.8% |

No version beats all three indexes from most start dates. Over these holding periods the median
return was 44% a year for SMH, 23% for QQQ and 18% for SPY.

## Next steps (nothing is changed in the live repos)

1. If you want one change to try live, Candidate 3 is the one chosen without seeing the test
   years. A paper forward test beside v5 is the honest next step; its config is in
   `results/exp2_top.csv` row 3.
2. A survivorship-free dataset (MarketParquet $79, Norgate, Sharadar) would make 2000-2018 a real
   test instead of a biased one. Everything here would rerun with `code/panel.py` pointed at it.
3. Beating SMH every year needs semiconductor exposure, not better stock selection. If that is
   the real goal, the question is how much of the book should simply follow the leading index ETF.

## Files

- `code/fetch.py`, `fetch_splits.py`, `panel.py`: data; `coverage.py`: survivorship audit
- `code/engine.py`: simulator with each difference as a switch; `live_harness.py`: the live code
- `code/runner.py`, `metrics.py`: windows, start dates, scorecard
- `code/exp1_switches.py` ... `exp4_ablation.py`: the experiments, results in `results/`
- Raw prices are not committed (about 300 MB); `fetch.py` rebuilds them in about an hour.
