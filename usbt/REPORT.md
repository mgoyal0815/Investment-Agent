# US v5 and India v8 on US data: train 2012-2019, test 2020-2026 and 2000-2011

Backtest study, 13 October 2026. The live system is unchanged; everything is in `usbt/` on
branch `claude/usa-india-backtest-optimization-fr5gs8`. This report uses the owner's windows:
choose on 2012-2019, test on January 2020 to 12 October 2026 and on 2000-2011. An earlier round
trained on 2012-2018; its files are in `results_train2012_2018/`.

## Answer

**No combination beats SPY, QQQ and SMH in every year from every start date.** In the 2020-2026
test no train-chosen version beat all three in more than 58% of years, and none did it with a
worst fall smaller than the indexes'. The years that break every version are semiconductor and
mega-cap years: SMH returned +73% in 2023 and +39% in 2024, QQQ +55% in 2023.

**The rule fixed before testing picks a configuration that fails out of sample.** The best
2012-2019 score belongs to "Hybrid A" (12 equal slots from the 300 most-traded stocks, India
ranking, India buy pause, gold slots). In training it made 23.9% a year, beat all three indexes in
72% of years and its worst fall was -18%, smaller than every index's. In 2020-2026 it made 27.7%
with a -54% worst fall, against 30.2% and -35% for live v5. That is overfitting.

**Candidate 3 is the robust result**, third in training both times the search was run:

| | 2012-19 (train) | 2020-26 (test) | 2000-11 (test, biased) |
|---|---|---|---|
| **US v5 as live** | 15.2% a year, worst fall -24% | 30.2%, -35% | 1.6%, -60% |
| **Candidate 3** | 15.2%, -18% | 30.0%, -31% | 12.9%, -27% |
| Hybrid A (best training score) | 23.9%, -18% | 27.7%, -54% | 4.4%, -58% |
| India v8 on US data | 15.6%, -22% | 21.7%, -50% | 13.6%, -61% |
| SPY / QQQ / SMH worst fall | -19% / -23% / -27% | -34% / -35% / -45% | -55% / -83% / -84% |

Median of five start dates a quarter apart, $10,000, whole shares, every cost. Worst fall is the
deepest drawdown across the five paths.

Candidate 3 is the US v5 stock engine with India's pieces where they help:
- the 500 most-traded US stocks as the universe;
- India's buy pause (breadth under 40%, large-cap trend, HYG) instead of the regime overlay, so
  stock is never sold to match a regime, and no -12% sleeve exit or daily re-entry;
- decisions twice a month (the 11th and the last session) instead of daily;
- a holding is also sold when it ranks outside the top 45, as well as on its v5 stop;
- no v5 adds, and a stricter Recovery cutoff (0.95).

Against live v5 it earns the same in 2020-2026 with a smaller worst fall (-31% against -35%),
and in 2000-2011 it makes 12.9% a year instead of 1.6% with a worst fall of -27% instead of -60%.
It beats all three indexes in about the same share of years as v5 (35% against 38% in 2020-26).
It does not meet the every-year goal.

One caution: Candidate 3 is not what the fixed rule picked. I recommend it because it is the
only finalist that matched or beat v5 in every window, and I had already seen 2019-2026 in the
first round. Treat it as a candidate for a paper forward test, not a proven system.

## How it was tested

- **Data.** FMP dividend-adjusted daily prices, 1999 to 12 Oct 2026, for 1,505 stocks: every name
  the live store has ever used as a universe, plus every ticker in the S&P 500 since 1996
  (fja05680/sp500), with tickers now used by a different company removed. FMP split histories so
  share counts and per-share commission use the price actually traded.
- **Survivorship.** FMP on this plan has almost no stocks delisted before about 2012. Share of
  each year's S&P 500 the data can price: 52% in 2000, 75% in 2008, 84% in 2012, 97% in 2019,
  99-100% from 2020 (`results/coverage.csv`). The 2020-2026 test is close to survivorship-free;
  2012-2019 mostly; 2000-2011 flatters every system and is for direction only.
- **Account.** $10,000, whole shares for stocks and ETFs, orders at the next open, the framework's
  cost model on every fill ($0.005 a share + 10 bps, $1 minimum), no taxes.
- **Engine check.** `code/engine.py` reproduces the live code (`code/live_harness.py` runs
  `v5_engine`, `v5_sleeve` and `v5_alloc` unchanged) to about one point a year in every window
  (`results_train2012_2018/ref_live_vs_engine.csv`).
- **Selection.** About 600 random combinations of both systems' rules, then a local search, all
  scored on 2012-2019 only, with the same search space, random seed and objective as the first
  round (`code/pipeline.sh`). The objective rewards beating all three indexes each year and
  penalises a worst fall deeper than any index's.

## The differences one at a time (2012-2019)

Each row changes one thing in live v5 (`results/exp1_switches.txt`).

| Change to v5 | CAGR | Worst fall | Years beating all 3 |
|---|---|---|---|
| none (live v5) | 15.2% | -24% | 11% |
| hold 10 in equal slots | 17.3% | -31% | 14% |
| fully invested, no overlay | 14.0% | -42% | 16% |
| India buy pause instead of regimes | 14.8% | -30% | 25% |
| drop the -12% exit and daily re-entry | 14.6% | -24% | 14% |
| sell on rank instead of stops | 15.1% | -30% | 19% |
| no adds | 13.9% | -22% | 19% |
| India ranking | 9.8% | -30% | 3% |
| no pullback-band entry | 15.4% | -25% | 11% |
| decide twice a month | 15.2% | -25% | 14% |
| idle cash in GLD | 14.6% | -25% | 11% |
| top-500 universe | 15.9% | -24% | 14% |
| no ramp-in | 15.0% | -24% | 14% |
| 1% of daily volume order cap (never binds at $10,000) | 15.2% | -24% | 11% |
| tax-aware sell delay | not tested: no US tax modelled | | |

No single change moves v5 much. The gains come only from combinations, and the large training
gains did not survive the test.

## 2020-2026, one run from January 2020

| Year | v5 live | Candidate 3 | Hybrid A | India v8 | SPY | QQQ | SMH |
|---|---|---|---|---|---|---|---|
| 2020 | 92.1 | 49.6 | 44.6 | 27.5 | 17.2 | 46.2 | 52.0 |
| 2021 | 81.9 | 30.0 | -20.1 | -8.3 | 28.7 | 27.4 | 42.1 |
| 2022 | -16.6 | -2.2 | -8.0 | -7.4 | -18.2 | -32.6 | -33.5 |
| 2023 | 14.4 | 18.9 | 18.4 | 25.8 | 26.2 | 54.9 | 73.4 |
| 2024 | 25.2 | 15.3 | 14.0 | 16.4 | 24.9 | 25.6 | 39.1 |
| 2025 | 17.0 | 52.4 | 80.5 | 51.3 | 17.7 | 20.8 | 49.2 |
| 2026 to 12 Oct | 57.8 | 49.3 | 124.6 | 58.2 | 15.0 | 21.9 | 64.1 |
| **2020-26 a year** | **35.2%** | **30.0%** | **29.7%** | **21.7%** | | | |
| worst fall | -33% | -29% | -51% | -49% | -34% | -35% | -45% |

Per cent. One path, so these differ from the five-start medians above (v5's January 2020 start
caught the 2020-21 run in full; its five-start median is 30.2%). Year tables for the 2012 and
2000 runs are in `results/exp3_years.csv` and on the charts page.

## Start-date test (2020-2025)

A fresh $10,000 account every quarter from January 2020 to October 2025, held to 12 Oct 2026
(`results/exp3_fresh_starts.csv`):

| | Beat all 3 to today | Beat all 3 in first 12 months | Worst fall smaller than all 3 | Median CAGR |
|---|---|---|---|---|
| US v5 live | 0% | 33% | 4% | 29.7% |
| Candidate 3 | 4% | 29% | 4% | 31.3% |
| Hybrid A | 67% | 38% | 0% | 52.8% |
| India v8 on US | 4% | 25% | 0% | 33.3% |

Hybrid A beats the indexes to today from two thirds of start dates because 2025-2026 were
exceptional for it (+80% and +125%), and every start from mid-2021 on includes them. From 2020
starts it earned no more than v5 or SMH, it lost 20% in 2021, and it fell 54% at worst. Its
results lean on a two-year run, which is not a steady edge.

## Next steps (nothing is changed in the live repositories)

1. Paper-test Candidate 3 beside live v5. Its configuration is row 3 of `results/exp2_top.csv`.
2. A survivorship-free dataset (MarketParquet $79, Norgate, Sharadar) would turn 2000-2019 into a
   real test. The code reruns unchanged: `code/fetch.py`, `panel.py`, `pipeline.sh`.
3. Beating SMH every year needs semiconductor exposure, not better stock selection. If that is
   the goal, the question is how much of the book should simply follow the leading index ETF.

## Files

- `code/fetch.py`, `fetch_splits.py`, `panel.py`: data; `coverage.py`: survivorship audit
- `code/engine.py`: simulator with each difference as a switch; `live_harness.py`: the live code
- `code/runner.py` (windows), `metrics.py` (scorecard), `pipeline.sh` (this run, end to end)
- `results/`: this run; `results_train2012_2018/`: the first round (train to 2018), including the
  ablation of Candidate 3 (`exp4_ablation.csv`)
- Raw prices are not committed (about 300 MB); `fetch.py` rebuilds them in about an hour.
