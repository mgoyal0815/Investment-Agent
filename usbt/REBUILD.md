# Rebuilding the usbt experiment

How to rebuild the price panel from scratch and reproduce a stored result. Written on 17 Oct 2026 after a rebuild
first gave $1.20M for System R (2000-2025) against the stored $2.87M. Four separate causes were found and fixed;
the scripts now refuse the mistakes below rather than relying on this page.

## The four causes (what went wrong, what now stops it)

| # | Cause | Effect on System R 2000-2025 | Now enforced by |
|---|---|---|---|
| 1 | Nasdaq-100 names were fetched by hand, not by `fetch.py` | 1533 stocks instead of 1556; different picks from 2004 | `fetch.py` reads `../data/ndx_tickers.txt`; `panel.py` preflight refuses a build with unfetched names |
| 2 | Sector ETFs (SOXX IGV XLE XLV XLF XLI XLY XLP XLU XLB XLC) were not in `fetch.py` | the restart signal `ra_cd100` (XLY / XLP) is missing: results differ from 2000 | `fetch.py` fetches them; `panel.py` preflight refuses a build without them |
| 3 | numpy / pandas versions. The engine ranks with `np.argsort`, whose order for **tied** scores is not fixed. Percentile scores tie often (MO and L both 0.978682 on 2000-08-15) | numpy 1.26.4 / pandas 2.2.2: paths split on 2000-08-16, end $2.34M. numpy 2.5.3 / pandas 3.0.5: end $2.87M (the stored result) | `envinfo.check()` in `yardstick.py` and `check_rebuild.py` refuses other versions unless `ALLOW_ANY_ENV=1` |
| 4 | FMP recalculates a stock's whole dividend-adjusted history every time it pays a dividend (28 panel stocks went ex-dividend 13-16 Oct 2026) | cent-level drift; end within 0.05% (0.01% with those dividends undone) | unavoidable with a live data source: `FETCH.json` records the fetch date; archive the raw data (below) |

**Also note:** `final_*.py` rescales curves so the first day's close is $10,000. On that close the account is
already about $9,842 (opening buys, costs, the day's moves), so the stored "$10k -> $2.92M" is the account's
$10k -> $2.87M. Growth rates are unaffected. `final_R.py` now writes this note and the environment into its summary.

## Steps

From `usbt/code`, with `FMP_API_KEY` set and mgoyal0815/v5-forward-test cloned (its `keys/` folder holds the v5
universe lists; point `V5_KEYS` at it if it is not at `/home/user/v5-forward-test/keys`).

1. `python3 ndx_members.py <path to shardul0701/nq100-survivorship-bias-data-2004-2026/src/nasdaq_100_ticker_history>`
   (only if `../data/ndx_tickers.txt` is missing; it is committed).
2. `python3 fetch.py ../data`: about 2,650 symbols, about 45 minutes at FMP's default pace. Resumable. Writes
   `../data/raw/*.parquet` (`*.none` = FMP has no data) and `../data/FETCH.json` (date, counts, versions).
   The last line should read `done ... 0 failed`; if not, run it again (it skips what it already has).
3. `NDX=1 python3 panel.py ../data` must print `1556 stocks kept, 185 dropped`,
   `panel (6986, 1556) 1999-01-04 2026-10-12` and a `stamp` line. It writes `panel_ndx.pkl.BUILD.json` (shape,
   counts, SHA-256, versions, fetch record). Needs several GB of free memory: do not run it beside a scoring job (it was killed for lack of memory on 17 Oct).
4. `python3 volume_ind.py panel_ndx` (UDVR, OBVT, CMF, RVOL; System R needs OBVT). Rerun after every panel build.
5. `python3 check_rebuild.py system_R` must print `REBUILD OK` (end within 1%, no 1% split before 2002).
   17 Oct 2026 rebuild: account end $2,867,613, stored $2,869,015 (-0.05%), first 1% split 2007-06-08.

Scoring (`robust.py`, `accept.py`, `final_*.py`) runs on the container's `python3` (3.13, numpy 2.5.3,
pandas 3.0.5). The v5 venv (`/home/claude/v5/.venv`, numpy 1.26.4) is fine for steps 2-4 only. Each scoring worker
holds about 3.6 GB; on a 15 GB container use 3 workers, or the out-of-memory killer removes workers and
`multiprocessing.Pool.map` waits forever.

## Keep with every result

- `envinfo.stamp(panel)`: versions plus the panel's SHA-256 and fetch date (`yardstick.ENV` after `_init`).
- The panel's `BUILD.json`. A result without one cannot be checked later.
- Archive the raw parquet files (about 1 GB, too large for git; a release asset or cloud storage) whenever a
  system is adopted, so the exact inputs exist after FMP revises its history.

## Tie-break: decided 17 Oct 2026 (System R1)

Cause 3 was a fragility in the engine itself: `np.argsort` left the order of tied scores to the numpy version, and
R's score (a percentile of the average of two percentiles) ties often. The live account runs numpy 1.26.4, the
backtest 2.5.3: on the live panel 2020-06 to 2026-10 the old engine made **different trades** on the two (253 vs 252
trades, $132,603 vs $131,729), so the live account was not running the backtested path.

**Now:** every ranking sorts with `engine.rank_desc(score, key)` (`np.lexsort`, a stable sort): highest score first,
NaN last, ties broken by `Market.tie_key(cfg, i)`. Default key: alphabetical symbol. `cfg["tie_key"]` puts an
information key first (alphabetical after it); `cfg["tie_seed"]` uses a fixed random symbol order instead (only to
measure tie luck). **System R1 = System R + `tie_key: "rs126"`**: a tie goes to the higher 6-month strength.

How it was chosen (rule fixed before the day-15 results were seen): a key must beat the tie-seed range (8 random
tie orders x 5 review calendars) on both day-1 and day-15 starts; among keys that do, the highest growth wins.

| Calendar years beating SPY and QQQ (owner's measure) | Day-1 starts | Day-15 starts | Growth (median start) |
|---|---|---|---|
| Random tie orders, 8 seeds (tie-luck range) | 69.0-70.8% (median 69.9%) | 68.9-70.5% (median 69.9%) | 28.8-28.9% |
| Alphabetical | 71.1% | 71.1% | |
| **6-month strength (rs126) = System R1** | **71.2%** | **71.2%** | **29.9% / 29.8%** |
| Low volatility | 70.1% | 70.0% | 29.5% |
| 52-week high | 70.7% | 70.4% | 28.3% |
| Raw z-sum / 200-day distance / turnover / 12-month / reverse alphabetical | 69.0-69.6% | not run (failed day 1) | 27.8-29.2% |

One account 2000-01-03 to 2025-12-31 (audit a70; account values, not rescaled):

| | End value | Growth/yr | Worst fall |
|---|---|---|---|
| System R, 40 random tie orders: median (range) | $2.55M ($2.06-3.59M) | 23.8% (22.8-25.4%) | -41.4% (-38.3 to -44.9%) |
| System R, alphabetical | $2.64M | 23.9% | -44.0% |
| **System R1** | **$3.15M** | **24.8%** | **-40.2%** |
| (stored System R before the fix, numpy 2.5.3 argsort) | $2.87M | 24.4% | -39.4% |

All seven information keys ended above the random-order median; the reverse-alphabetical control ended below it.
Over about 100 start dates tie luck moves the calendar-year score by under 2 points; a single 26-year path moves by
far more. Full results: `audit/a69_*`, `audit/a70_R_paths.*`, `audit/a71_*` (each with seeds, versions, panel hash).

## Replication and determinism checks

| Check | Command (from `usbt/code`) | Expected |
|---|---|---|
| Tie order is version-proof | `python3 check_ties.py` on two numpy versions | `rank_desc 9264628e7fa85137` on both |
| Same decisions on every environment | `python3 test_determinism.py <panel> ../system_R1.json [start] [end]` on two environments, twice each | identical `decisions_sha`; `curve_sha_exact` identical within an environment; end value within a cent across environments (pandas 2 vs 3 float32 rounding) |
| Rebuild matches the stored result | `python3 check_rebuild.py system_R1` | `REBUILD OK` (account end $3,146,326 on panel_ndx c2d9f773) |
| Tie luck of a system | `python3 ../audit/a69_tie_luck.py R1` | median and seed range per measure, `a69_R1_day1.json` |

17 Oct 2026, live panel (`/home/claude/r/data/panel.pkl`, 771 symbols, 2018-06 to 2026-10-16), System R1 2020-06-01 to
2026-10-16: numpy 2.5.3 / pandas 3.0.5 and numpy 1.26.4 / pandas 2.2.2, twice each: 246 trades, decisions
`4c473d8f559cd271` on all four, $139,778.37 / $139,778.38.

## The data sets

- **Research panel** (`../data/panel_ndx.pkl`): steps 1-4 above. The panel behind every result here was built on 16 Oct
  2026, before the guards; `stamp_existing.py` gave it a `BUILD.json` after the fact: 6,986 sessions x 1,556 stocks,
  185 dropped, SHA-256 `c2d9f773639cf695...`. The S&P-only `panel.pkl` (systems A-K): 1,521 stocks, `88bbe1ca85a2309a...`.
  `stamp_existing.py` refuses to overwrite a `BUILD.json` that panel.py wrote.
- **Live panel** (mgoyal0815/v5-forward-test `claude/r/r_data.py`, run each evening by `bash claude/run_daily.sh prep`):
  FMP dividend-adjusted daily bars from 2018-06-01 for every S&P 500 / Nasdaq-100 member since 2018 (membership: the
  fja05680 S&P 500 history and jmccarrell Nasdaq-100 lists from GitHub, the committed `claude/r/seed/` copies as
  fallback), splits from the seed plus FMP's split calendar, the day's partial bar dropped before 16:45 New York.
  Writes `panel.pkl`, `vol_panel.npz` and `meta.json` (last session, membership dates, missing symbols). It runs on
  the pinned live environment (`claude/v5_env.py`: numpy 1.26.4, pandas 2.2.2).

## Live system tests (v5-forward-test)

`claude/r/test_r.py` (live environment, after `prep`): tie self-test (`216e04a18afecb7e`, identical on numpy 2.5.3 and
1.26.4), same run twice identical, stepping one session at a time = one continuous run (30 sessions: same trades, NAV
gap 0), switch table accepts only its approved row. `r_live.py` runs the tie self-test at every start and stops with
`TIE-BREAK SELFTEST` if this machine would order ties differently.
