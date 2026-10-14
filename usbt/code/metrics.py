"""Scorecard against SPY, QQQ and SMH (total return, dividend-adjusted).

Year return = last close of the year / last close of the previous year - 1 (first year of a run:
from the run's first close). A year 'beats all' when it beats SPY, QQQ and SMH in that year.
Max drawdown is compared with each index's max drawdown over the SAME sessions.
Rolling: every month-start inside the run, the next 12 months, system vs each index.
"""
import numpy as np
import pandas as pd

BENCH = ("SPY", "QQQ", "SMH")


def bench_curves(m, idx):
    return {k: pd.Series(m.E[k]["c"], index=m.dates).reindex(idx) for k in BENCH}


def maxdd(s):
    s = s.dropna()
    return float((s / s.cummax() - 1).min())


def cagr(s):
    s = s.dropna()
    days = (s.index[-1] - s.index[0]).days
    return (s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1 if days > 0 else np.nan


def yearly(s):
    s = s.dropna()
    ye = s.groupby(s.index.year).last()
    prev = ye.shift(1)
    prev.iloc[0] = s.iloc[0]
    return ye / prev - 1


def score(m, curve):
    idx = curve.index
    b = bench_curves(m, idx)
    ys = yearly(curve)
    yb = {k: yearly(v) for k, v in b.items()}
    years = pd.DataFrame({"sys": ys, **{k: v for k, v in yb.items()}})
    full = years.index[[(curve[curve.index.year == y].index[0] - pd.Timestamp(f"{y}-01-01")).days < 7
                        and (pd.Timestamp(f"{y}-12-31") - curve[curve.index.year == y].index[-1]).days < 7 for y in years.index]]
    yrs = years.loc[full]
    beat = {k: int((yrs["sys"] > yrs[k]).sum()) for k in BENCH}
    beat_all = int(((yrs["sys"] > yrs["SPY"]) & (yrs["sys"] > yrs["QQQ"]) & (yrs["sys"] > yrs["SMH"])).sum())
    # rolling 12-month windows from each month start
    ms = curve.groupby([curve.index.year, curve.index.month]).head(1).index
    wins, n = 0, 0
    worst_gap = np.inf
    for t0 in ms:
        t1 = t0 + pd.DateOffset(months=12)
        if t1 > idx[-1]:
            break
        e = idx[idx.searchsorted(t1) - 1]
        r = curve[e] / curve[t0] - 1
        rb = [b[k][e] / b[k][t0] - 1 for k in BENCH]
        n += 1
        wins += r > max(rb)
        worst_gap = min(worst_gap, r - max(rb))
    dd = maxdd(curve)
    ddb = {k: maxdd(v) for k, v in b.items()}
    return dict(cagr=cagr(curve), maxdd=dd, **{f"cagr_{k}": cagr(v) for k, v in b.items()},
                **{f"dd_{k}": v for k, v in ddb.items()},
                dd_ok=bool(all(dd > v for v in ddb.values())), dd_ok_SMH=bool(dd > ddb["SMH"]),
                years=len(yrs), beat_all=beat_all, **{f"beat_{k}": v for k, v in beat.items()},
                roll_win=wins / n if n else np.nan, roll_worst_gap=worst_gap,
                min_year_gap=float((yrs["sys"] - yrs[list(BENCH)].max(axis=1)).min()) if len(yrs) else np.nan,
                table=years)


def line(lab, s):
    return (f"{lab:40s} CAGR {s['cagr']:6.1%}  DD {s['maxdd']:6.1%} (idx {min(s['dd_SPY'], s['dd_QQQ'], s['dd_SMH']):.0%}..{max(s['dd_SPY'], s['dd_QQQ'], s['dd_SMH']):.0%})"
            f"  beatAll {s['beat_all']}/{s['years']}  SPY {s['beat_SPY']} QQQ {s['beat_QQQ']} SMH {s['beat_SMH']}"
            f"  roll12 {s['roll_win']:.0%}")
