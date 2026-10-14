"""The owner's yardstick (14 October 2026), scored from EVERY monthly start date, fresh $10,000 each time:

  1a. every year after the start beats BOTH SPY and QQQ  (anniversary years: start -> +1y -> +2y ...;
      also reported for calendar years, the first one partial)
  1b. every rolling 3-year window inside the run beats SMH (windows start each month; SMH from 2000-06)
  2.  the run's worst fall is smaller than BOTH SPY's and QQQ's over the same sessions
  3.  a start PASSES only if 1a, 1b and 2 all hold. The headline is the share of starts that pass.

Total returns (dividend-adjusted) for the indexes; the system pays costs, whole shares only."""
import multiprocessing as mp
import numpy as np
import pandas as pd
import engine as X
from exp24_timing import mat

MM = None


def _init(path):
    global MM
    MM = X.Market(pd.read_pickle(path))


def starts(a="2000-01-01", b="2025-10-31", step=1):
    d = MM.dates if MM is not None else pd.read_pickle("../data/panel.pkl")["dates"]
    s = pd.Series(d, index=d)
    s = s[(s >= a) & (s <= b)]
    first = s.groupby([s.index.year, s.index.month]).first().values
    return [pd.Timestamp(x) for x in first[::step]]


def judge(curve, m):
    """Score one fresh run. curve: NAV indexed by session from the start date."""
    idx = curve.index
    B = {k: pd.Series(m.E[k]["c"], index=m.dates).reindex(idx) for k in ("SPY", "QQQ", "SMH")}
    t0, tN = idx[0], idx[-1]
    at = lambda t: idx[min(idx.searchsorted(t), len(idx) - 1)]
    # 1a anniversary years
    ann, k = [], 1
    while t0 + pd.DateOffset(years=k) <= tN:
        a, b = at(t0 + pd.DateOffset(years=k - 1)), at(t0 + pd.DateOffset(years=k))
        r = curve[b] / curve[a] - 1
        ann.append((r > B["SPY"][b] / B["SPY"][a] - 1) and (r > B["QQQ"][b] / B["QQQ"][a] - 1))
        k += 1
    # calendar years (first partial, last partial year dropped unless it is complete)
    cal = []
    for y in sorted(set(idx.year)):
        seg = idx[idx.year == y]
        if y == tN.year and tN < pd.Timestamp(f"{y}-12-24"):
            continue
        a = idx[max(idx.searchsorted(seg[0]) - 1, 0)] if seg[0] != t0 else t0
        b = seg[-1]
        r = curve[b] / curve[a] - 1
        cal.append((r > B["SPY"][b] / B["SPY"][a] - 1) and (r > B["QQQ"][b] / B["QQQ"][a] - 1))
    # 1b rolling 3-year windows vs SMH
    ms = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first().values
    r3 = []
    for a in ms:
        a = pd.Timestamp(a)
        if a + pd.DateOffset(years=3) > tN or B["SMH"][a] != B["SMH"][a]:
            continue
        b = at(a + pd.DateOffset(years=3))
        r3.append(curve[b] / curve[a] > B["SMH"][b] / B["SMH"][a])
    dd = lambda s: float((s / s.cummax() - 1).min())
    d = dd(curve)
    ddi = {k: dd(B[k].dropna()) for k in B}
    ok_dd = d > ddi["SPY"] and d > ddi["QQQ"]
    ok = (all(ann) if ann else None), (all(r3) if r3 else None)
    passed = ok_dd and ok[0] is not False and ok[1] is not False and (ann or r3)
    yrs = (tN - t0).days / 365.25
    return dict(start=t0, years=round(yrs, 1), cagr=(curve.iloc[-1] / curve.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan,
                dd=d, dd_SPY=ddi["SPY"], dd_QQQ=ddi["QQQ"], dd_ok=ok_dd,
                ann_n=len(ann), ann_win=float(np.mean(ann)) if ann else np.nan, ann_all=ok[0],
                cal_n=len(cal), cal_win=float(np.mean(cal)) if cal else np.nan, cal_all=(all(cal) if cal else None),
                r3_n=len(r3), r3_win=float(np.mean(r3)) if r3 else np.nan, r3_all=ok[1], passed=bool(passed))


def _job(a):
    cfg, s, e = a
    r = X.run(MM, mat(cfg), start=str(s.date()), end=e)
    return judge(r["curve"], MM)


def evaluate(pool, cfg, st, end=None):
    return pd.DataFrame(pool.map(_job, [(cfg, s, end) for s in st], chunksize=4))


def summary(T, lab=""):
    full = T[T.years >= 3]
    return dict(label=lab, starts=len(T), PASS=T.passed.mean(), pass_3y_plus=full.passed.mean(),
                yrs_beat_SPY_QQQ=(T.ann_win * T.ann_n).sum() / max(T.ann_n.sum(), 1),
                starts_all_years=T.ann_all.dropna().astype(float).mean(),
                cal_years_beat=(T.cal_win * T.cal_n).sum() / max(T.cal_n.sum(), 1),
                r3_beat_SMH=(T.r3_win * T.r3_n).sum() / max(T.r3_n.sum(), 1),
                starts_all_r3=T.r3_all.dropna().astype(float).mean(), dd_ok=T.dd_ok.mean(),
                med_cagr=T.cagr.median(), worst_dd=T.dd.min())


if __name__ == "__main__":
    import json, sys
    path = sys.argv[2] if len(sys.argv) > 2 else "../data/panel.pkl"
    cfg = json.load(open(sys.argv[1]))["cfg"]
    pool = mp.get_context("fork").Pool(4, initializer=_init, initargs=(path,))
    _init(path)
    T = evaluate(pool, cfg, starts())
    pd.set_option("display.width", 250)
    print(pd.Series(summary(T, sys.argv[1])).to_string())
    T["era"] = pd.cut(T.start.dt.year, [1999, 2011, 2019, 2026], labels=["2000-11", "2012-19", "2020-25"])
    print(T.groupby("era", observed=True).apply(lambda g: pd.Series(summary(g))).drop(columns="label").round(3).T.to_string())
