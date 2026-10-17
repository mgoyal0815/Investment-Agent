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

## Open decision: fixed tie-break

Cause 3 is a fragility in the engine itself, not just in the environment: the live paper account runs on a
different numpy (the v5 venv, 1.26.4) and a smaller panel (different column set), so its ties can resolve
differently from the backtest whatever the versions. `np.argsort(..., kind="stable")` breaks ties by column
order (alphabetical symbol in both panels) and gives identical results on numpy 1.26.4 and 2.5.3. It also changes
the numbers (System R 2000-2025: $2.64M, 24.0% a year, worst fall -44.1%, against $2.87M / 24.4% / -39.4%), so it
needs re-scoring with `robust.py` and the owner's approval before the engine (and the live fingerprint) changes.
