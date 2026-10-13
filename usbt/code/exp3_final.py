"""Experiment 3: the finalists on every window.

Finalists = US v5 (live), India v8 on US, and the top configurations from exp2 (chosen on the
2012-2018 training window only). For each:
  * train / test / early windows, five start dates each (runner.PERIODS);
  * one continuous run 2012 -> today and one 2000 -> 2011, year by year against SPY, QQQ, SMH;
  * a fresh $10,000 start every quarter from 2019 Q1 to 2025 Q4, held to today: does it beat
    each index over its own holding period, and is its worst fall smaller?
Writes results/exp3_*.csv and results/curves.json (for the charts).
"""
import json
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd

import engine as X
import metrics as M
import runner as R
from exp2_search import materialise

M_ = None


def _init():
    global M_
    M_ = X.Market(pd.read_pickle("../data/panel.pkl"))


def _fresh(args):
    lab, cfg, start = args
    r = X.run(M_, cfg, start=start)
    c = r["curve"]
    b = M.bench_curves(M_, c.index)
    out = dict(label=lab, start=start, cagr=M.cagr(c), dd=M.maxdd(c))
    for k, v in b.items():
        out[f"cagr_{k}"] = M.cagr(v)
        out[f"dd_{k}"] = M.maxdd(v)
    e = c.index[min(len(c) - 1, c.index.searchsorted(c.index[0] + pd.DateOffset(months=12)) - 1)]
    out["r12"] = c[e] / c.iloc[0] - 1
    for k, v in b.items():
        out[f"r12_{k}"] = v[e] / v.iloc[0] - 1
    return out


def _cont(args):
    lab, cfg, start, end = args
    r = X.run(M_, cfg, start=start, end=end)
    s = M.score(M_, r["curve"])
    return lab, start, r["curve"], s["table"], {k: v for k, v in s.items() if k != "table"}, r["expo"]


def finalists(k=6):
    top = pd.read_csv("../results/exp2_top.csv")
    out = [("US v5 (live)", {}), ("India v8 on US", dict(R.INDIA))]
    for n, row in top.head(k).iterrows():
        out.append((f"Candidate {n + 1}", materialise(json.loads(row["cfg"]))))
    return out


if __name__ == "__main__":
    fin = finalists(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    extra = json.load(open("../results/extra_finalists.json")) if len(sys.argv) > 2 else []
    fin += [(e["label"], materialise(e["cfg"])) for e in extra]
    rn = R.Runner()
    rows = []
    for period in ("train", "test", "early"):
        res = rn.eval_many([c for _, c in fin], period)
        for (lab, c), a in zip(fin, res):
            print(period, R.fmt(lab, a), flush=True)
            rows.append(dict(period=period, label=lab, cfg=json.dumps(c, default=list), **a))
    pd.DataFrame(rows).to_csv("../results/exp3_periods.csv", index=False)

    pool = mp.get_context("fork").Pool(4, initializer=_init)
    jobs = [(lab, c, s, e) for lab, c in fin for s, e in (("2012-01-03", None), ("2020-01-02", None), ("2000-01-03", "2011-12-30"))]
    curves, years, summ = {}, [], []
    for lab, start, curve, table, s, expo in pool.map(_cont, jobs):
        key = f"{lab}|{start[:4]}"
        curves[key] = {"dates": [str(d.date()) for d in curve.index[::5]], "nav": [round(float(v), 2) for v in curve.values[::5]],
                       "expo": [round(float(v), 3) for v in expo.values[::5]]}
        t = table.copy()
        t["label"], t["run"] = lab, start[:4]
        years.append(t.reset_index().rename(columns={"index": "year"}))
        summ.append(dict(label=lab, run=start[:4], **s))
    m = X.Market(pd.read_pickle("../data/panel.pkl"))
    for s, e in (("2012-01-03", None), ("2020-01-02", None), ("2000-01-03", "2011-12-30")):
        i0 = m.dates.searchsorted(pd.Timestamp(s))
        i1 = len(m.dates) if e is None else m.dates.searchsorted(pd.Timestamp(e), side="right")
        for k in M.BENCH:
            v = m.E[k]["c"][i0:i1]
            v = v / v[~np.isnan(v)][0] * 10000
            curves[f"{k}|{s[:4]}"] = {"dates": [str(d.date()) for d in m.dates[i0:i1][::5]], "nav": [round(float(x), 2) for x in v[::5]]}
    json.dump(curves, open("../results/curves.json", "w"))
    pd.concat(years).to_csv("../results/exp3_years.csv", index=False)
    pd.DataFrame(summ).to_csv("../results/exp3_continuous.csv", index=False)

    starts = [str(d.date()) for d in pd.date_range("2020-01-01", "2025-12-31", freq="QS")]
    starts = [str(m.dates[m.dates.searchsorted(pd.Timestamp(s))].date()) for s in starts]
    fr = pool.map(_fresh, [(lab, c, s) for lab, c in fin for s in starts])
    fr = pd.DataFrame(fr)
    fr.to_csv("../results/exp3_fresh_starts.csv", index=False)
    for lab, g in fr.groupby("label", sort=False):
        beat_all = ((g.cagr > g.cagr_SPY) & (g.cagr > g.cagr_QQQ) & (g.cagr > g.cagr_SMH)).mean()
        dd_ok = ((g.dd > g.dd_SPY) & (g.dd > g.dd_QQQ) & (g.dd > g.dd_SMH)).mean()
        r12 = ((g.r12 > g.r12_SPY) & (g.r12 > g.r12_QQQ) & (g.r12 > g.r12_SMH)).mean()
        print(f"{lab:28s} fresh quarterly starts 2020-2025: beat all three to today {beat_all:.0%}, "
              f"first-12-months beat all {r12:.0%}, smaller worst fall than all three {dd_ok:.0%}, "
              f"median CAGR {g.cagr.median():.1%}")
