# IMPORTANT: look-ahead bias found (14 October 2026)

Most of the profit in System A, and in live v5's own backtests, comes from stocks that were not in the
S&P 500 when bought (68-101%). Those stocks enter the data through the live universe lists, which only
contain companies that reached $8bn at some point, including in the future. Small and mid caps that
boomed and then failed below $8bn are missing, so every result on that universe is inflated.

On a universe with no look-ahead (stocks that were S&P 500 members on that day; exp21, exp22):

| | 2012-19 (train) | 2020-26 (test) | 2000-11 (test) |
|---|---|---|---|
| US v5 live | 12.6% a year, worst fall -27% | 19.1%, -29% | 6.7%, -36% |
| System A | 10.7%, -25% | 25.0%, -25% | 1.8%, -42% |
| India v8 | 10.7%, -28% | 29.1%, -37% | 7.4%, -40% |
| **Best found on this universe (System B, exp22 pick 1)** | **24.3%, -21%** | **28.6%, -32%** | **6.6%, -34%** |
| SPY / QQQ / SMH | about 13 / 17 / 21% | 15.4 / 20.8 / 37.1% | 0.6 / -4.0 / -6.7% |

System B: 10 equal slots from S&P 500 members, RobustPair rank with the trend route, monthly decisions,
sell outside the top 20, India adds to winners, India buy pause (breadth line 20%), idle cash in the
leading index ETF (Treasuries while paused), gold takes 4 slots while it leads, profit lock 30%/25%.
Configuration in `system_B.json`. Treat every System A number below as an upper bound.

Root cause and luck test for System B (`code/analyze_train.py`, `code/luck_test.py`): one trade, NVDA
2015-2018, made 91% of its 2012-2019 profit. Removing each window's single biggest winner from the
universe, System B still makes 18.3% (2012-19), 17.9% (2020-26) and 6.8% (2000-11) against SPY's 14.6%,
15.4% and 0.6%; removing its top three, 12.8%, 12.3% and 5.9%. Its edge is catching and holding a few
very large winners, which were different stocks in each window. A luck-robust search around it
(scored with and without the biggest winner, `exp23`) found nothing that held up better on 2020-26
and 2000-11, so System B stands.

Decision-day luck (`exp24`): System B decides on the last session of each month. Moving that one day
changes its 2012-19 result from 15% to 24% a year and its 2020-26 result from 26% to 35%. Its training
lead was partly date luck. Splitting the account into two halves that decide on the last and the 11th
session removes most of that luck: 20.4% a year (2012-19, worst fall -18%), 32.1% (2020-26, -33%),
6.1% (2000-11, -41%). That is the honest expectation for this system on clean data.

System C (`system_C.json`, exp25): System B staggered inside ONE account (half the book reviewed on the
last session of the month, half on the 11th) with the exit at the top 15, chosen on 2012-2019 with a
score that ignores both date luck and the single biggest winner. Five-start medians: 23.4% a year with
a -16% worst fall (2012-19), 31.7% / -32% (2020-26), 7.2% / -39% (2000-11). One run from 2000: $10,000
to $725,000, 17.4% a year, worst fall -39%; it beats SPY from every quarterly start held to today, but
beats all three indexes in only 9 of 26 calendar years, and its 2020s gains are lumpy (2023 +9%,
2024 +9%, 2025 +46%, 2026 to date +128%). A higher-scoring 6-slot variant fell 37% in 2020-26 and was
not adopted.

## Leveraged semiconductor/index ETFs with System C (exp26-31, 14 October 2026)

Why System C misses: in most years it loses only to SMH (2003, 2009, 2019, 2020, 2023, 2024: SMH +55
to +87%). A 10-stock S&P 500 portfolio rarely beats that. The other tool that can is an ordinary 2x/3x
ETF (USD/SOXL, QLD/TQQQ, SSO/UPRO). These are not options and need no margin. Real fund prices were
used where they exist (2006-2010 onward). Before that, a daily-reset synthetic version with a 2.5% a
year drag was used; it checks within 1-2 points a year against the real QQQ/SPY funds.

| | Calendar years 2000-2025 beating all three | max fall 2000-26 | CAGR 2000-26 |
|---|---|---|---|
| System C alone | 35% (9/26) | -39% | 17.4% |
| Leveraged ETF trend rules alone, best of 576 (exp26) | 54% | -89% to -99% | |
| Fixed blend System C + leveraged sleeve (exp26b) | 46% | -38% to -41% | about 19% |
| Switch System C <-> 3x SMH in a confirmed SMH boom, real engine (exp31 D) | 58% | -88% | 22.6% |
| Same, 2x on 2/3 of the account (exp31 B) | 50% | -59% | 19.6% |

- The return-curve shortcut (exp27-30, about 3,000 variants) was too optimistic. In the real engine
  (exp31), System C is sold when the switch turns on and rebuilt when it turns off. That costs 2-7
  points a year and adds 10-20 points of drawdown.
- Fresh quarterly starts held to today, real engine: exp31 D beats all three from 98% of starts, B
  from 87%, System C from 19%. Held 1, 3 or 5 years: 50-62%.
- The years that defeat every variant are 2005-07, 2010-11, 2018 and 2022. In those, SMH trend signals
  whipsaw and System C is also weak.

Conclusion so far: nothing found reaches 75% of calendar years. Above 55%, every variant needs 3x
leverage and falls further than SPY (-55%). That breaks the drawdown rule. The owner rejected
leveraged ETFs.

## Wider universe: S&P 500 + Nasdaq-100 (exp32, exp32b, exp33, 14 October 2026)

- Nasdaq-100 membership is point-in-time from 2004 (shardul0701/NQ100-Survivorship-bias-data-2004-2026).
  It is upstream-accurate from 2015; 2004-2014 is reconstructed from public add/remove events. There
  is no Nasdaq-100 data for 2000-2003, so those years stay S&P 500 only.
- FMP has prices for 251 of the 299 Nasdaq-100 names. The 48 missing are mostly delisted before 2012
  (YHOO, RIMM, BEAS, CEPH ...). The panel with them is `panel_ndx.pkl`; `panel.pkl` and System C are
  unchanged.
- Ceiling check: each year, take with hindsight whichever of six rules did best (top 3/5/10 by 6-month
  gain, two versions limited to names moving with the leading index, and System C). On the S&P 500 that
  beats all three indexes in 64% of 2001-2025. With the Nasdaq-100 added, 72%. No real system knows the
  best rule in advance, so the real figure is lower.
- Years even hindsight loses: 2009, 2011, 2014, 2018, 2019, 2023, 2024. In 2019, 2023 and 2024 the
  best S&P/Nasdaq-100 picks that move with SMH made 51%, 59% and 38%; SMH made 64%, 73% and 39%.
  Part of SMH's lead comes from TSMC, which is in neither index.
- System C on the wider universe (exp33, one run from 2000): 38% of years beating all three, against
  42% for the same system on the same panel with S&P 500 members only. Ten nearby variants scored
  27-50%, CAGR 10-21%, worst fall -36% to -50%.

## The owner's yardstick (exp34, exp35, 14 October 2026)

Rules: beat SPY and QQQ every year, beat SMH over every rolling 3 years, worst fall smaller than both
SPY's and QQQ's, from ANY start date. Scored from every monthly start 2000-2025, each a fresh $10,000
(`code/yardstick.py`). Search (exp34): 400 random settings plus fine-tuning around the best 3, chosen on
starts 2012-2016 stopping at 2019; judged on starts 2000-2008 and 2020-2025.

| From every monthly start 2000-25 | System C | best tuned (pick 3) |
|---|---|---|
| Starts that meet every rule | 0% | 0% |
| Years (from the start date) beating both SPY and QQQ | 48% | 46% |
| 3-year windows beating SMH | 32% | 37% |
| Starts with a smaller worst fall than SPY and QQQ | 78% | 78% |

- Tuning lifted the training score (56% to 62-67% of years) but not the unseen starts (45-50%).
  That is curve-fitting, so System C stays.
- Holding period, System C beats both SPY and QQQ: 1 year 47% of starts, 3 years 58%, 5 years 55%.
  Held to today: 100%. Held only to the end of 2025: 31%. The "to today" figure rests on 2026's
  +128% and is not reliable.
- Starts from 2020 on fail the drawdown rule 97% of the time: from those starts, SPY's worst fall
  was about -25%, while System C fell 28-34%.

Verdict: on clean data, no rule found beats SPY and QQQ every year from every start. The best beat
both in about half of years.

---

# System A (frozen 14 October 2026)

`system_A.json` holds the configuration; `code/final_backtest.py` reproduces every number here.
The live system is unchanged.

**What it is:** the US v5 stock engine with
1. the 500 most-traded US stocks as the universe;
2. India v8's buy pause instead of the v5 regime overlay (pause on HYG below its 63-day average,
   the equal-weight large-cap index below its 50-day average, or breadth under 30%); no -12% exit
   or daily re-entry;
3. decisions on the 11th and last session of each month;
4. a sale when a holding ranks outside the top 45, as well as on its v5 stop;
5. idle cash in the leading index ETF (SPY, QQQ or SMH by 6-month return), Treasuries (SHY) while paused;
6. v5 adds to winners;
7. a profit lock: once a position is up 30%, sell it if it closes 25% below its peak.

**Backtest, $10,000, whole shares, every cost:**

| One run from | Ends at | A year | Worst fall | SPY / QQQ / SMH a year | Their worst falls |
|---|---|---|---|---|---|
| Jan 2000 | $2,269,000 | 22.5% | -39% | 8.4 / 8.7 / 12.1% | -55 / -83 / -84% |
| Jan 2012 | $691,000 | 33.3% | -33% | 15.0 / 20.1 / 29.4% | -34 / -35 / -45% |
| Jan 2020 | $132,000 | 46.2% | -32.5% | 15.4 / 20.8 / 37.1% | -34 / -35 / -45% |

Beats SPY, QQQ and SMH together in 15 of 26 calendar years (2000-2025). A fresh $10,000 started in any
quarter and held to today beat all three in 85% of 2000-2011 starts, 97% of 2012-2019 starts and 67%
of 2020-2025 starts; its worst fall was smaller than SMH's from every start before 2020.

**How it was found:** trade-by-trade attribution on 2012-2019 (`code/analyze_train.py`). Positions
held more than 63 sessions made all the profit; winners gave back 28 points (54 in 2020-26) before
their stop. The profit lock and the adds fix that, chosen on 2012-2019 only (`exp13`-`exp15`).

**What it does not do:** beat all three indexes every year. It misses SMH boom years and rebounds
(2003, 2009, 2019, 2023) and 2010-2011. Pausing less, and a rebound mode that holds the leading
index after a fall, both made it worse (`exp16`, `exp17`). The 2010-2011 losses come from crash
bleed (Aug 2011: the pause stops buying but the held stocks fall 25% in 8 sessions while the 88-session
floor blocks their stops). Letting losers exit early (framework 19.8), a hard stop from cost, and
selling losers when the pause fires all made every window worse, because they sell near the lows and
buy back higher (`exp18`). Switching the parking ETF only on decision days helped 2000-11 but hurt
2012-19 and 2020-26, so it is not adopted.

---

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
