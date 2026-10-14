"""Experiment 15: robustness of the exp14 finalists. Fresh $10,000 starts every quarter 2020-2025
held to today, and every quarter 2012-2018 held 3 years; year-by-year table. Writes results/exp15_*."""
import json, multiprocessing as mp
import numpy as np
import pandas as pd
import engine as X, metrics as M
from exp2_search import materialise
from exp8_more import NEW

FIN = [("previous best", NEW),
       ("lock 50/25", dict(NEW, lock_at=0.5, lock_trail=0.25)),
       ("lock 30/25 + adds, rank hold 21", dict(NEW, lock_at=0.3, lock_trail=0.25, adds="v5", rank_min_hold=21)),
       ("lock 30/25 + adds, rank hold 0", dict(NEW, lock_at=0.3, lock_trail=0.25, adds="v5", rank_min_hold=0)),
       ("lock 50/20 + adds, rank hold 0", dict(NEW, lock_at=0.5, lock_trail=0.2, adds="v5", rank_min_hold=0))]
MM = None
def _init():
    global MM
    MM = X.Market(pd.read_pickle("../data/panel.pkl"))
def _fresh(a):
    lab, c, s, e = a
    cv = X.run(MM, materialise(c), start=s, end=e)["curve"]
    b = M.bench_curves(MM, cv.index)
    return dict(label=lab, start=s, cagr=M.cagr(cv), dd=M.maxdd(cv), **{f"cagr_{k}": M.cagr(v) for k, v in b.items()},
                **{f"dd_{k}": M.maxdd(v) for k, v in b.items()})
if __name__ == "__main__":
    pool = mp.get_context("fork").Pool(4, initializer=_init)
    m = X.Market(pd.read_pickle("../data/panel.pkl"))
    def q(a, b):
        return [str(m.dates[m.dates.searchsorted(d)].date()) for d in pd.date_range(a, b, freq="QS")]
    jobs = [(l, c, s, None) for l, c in FIN for s in q("2020-01-01", "2025-10-01")]
    jobs += [(l, c, s, str((pd.Timestamp(s) + pd.DateOffset(years=3)).date())) for l, c in FIN for s in q("2012-01-01", "2018-10-01")]
    fr = pd.DataFrame(pool.map(_fresh, jobs))
    fr.to_csv("../results/exp15_fresh.csv", index=False)
    fr["period"] = np.where(fr.start >= "2020", "test starts 2020-25 (to today)", "train starts 2012-18 (3 years)")
    fr["beat_cagr_all"] = (fr.cagr > fr[["cagr_SPY", "cagr_QQQ", "cagr_SMH"]].max(axis=1))
    fr["dd_smaller_all"] = (fr.dd > fr[["dd_SPY", "dd_QQQ", "dd_SMH"]].max(axis=1))
    print(fr.groupby(["period", "label"], sort=False).agg(starts=("cagr", "size"), median_cagr=("cagr", "median"),
          worst_cagr=("cagr", "min"), worst_dd=("dd", "min"), beat_all_growth=("beat_cagr_all", "mean"),
          dd_below_all=("dd_smaller_all", "mean")).round(3).to_string())
    spans = (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None))
    yrs = {l: pd.concat([M.yearly(x) for x in pool.map(_curve, [(c, s, e) for s, e in spans])]) * 100 for l, c in FIN} if False else {}
    for l, c in FIN:
        yrs[l] = pd.concat([M.yearly(X.run(m, materialise(c), start=s, end=e)["curve"]) for s, e in spans]) * 100
    full = m.dates[m.dates >= "2000-01-03"]
    for k, v in M.bench_curves(m, full).items():
        yrs[k] = pd.concat([M.yearly(v[s:e]) for s, e in spans]) * 100
    t = pd.DataFrame(yrs).round(1)
    t.to_csv("../results/exp15_years.csv")
    b = t[["SPY", "QQQ", "SMH"]].max(axis=1)
    for l, _ in FIN:
        w = (t[l] > b) & (t.index < 2026)
        c_ = lambda a, z: int(w[(t.index >= a) & (t.index <= z)].sum())
        print(f"{l:34s} beat all 3: 2000-11 {c_(2000, 2011)}/12  2012-19 {c_(2012, 2019)}/8  2020-25 {c_(2020, 2025)}/6  total {int(w.sum())}/26")
    print(t.to_string())
