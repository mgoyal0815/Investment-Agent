"""Experiment 24 (clean S&P 500 universe): decision day, staggered tranches, adaptive slot count,
around System B. Luck-robust score on 2012-2019 (mean objective2 with and without the single biggest
winner); finalists on 2020-2026 and 2000-2011 with and without their top winner.
Tranches: k separate sub-accounts of $10,000/k, each deciding on a different session of the month."""
import itertools, json, multiprocessing as mp
import numpy as np
import pandas as pd
import engine as X, metrics as M, runner as R
from exp2_search import materialise

B = json.load(open("../system_B.json"))["cfg"]
MM = None
def _init():
    global MM
    MM = X.Market(pd.read_pickle("../data/panel.pkl"))
def mat(c):
    c = materialise(c)
    if c.get("lock_at") is None:
        c.pop("lock_at", None)
    return c
def curve(cfg, s, e, excl=None):
    days = cfg.get("tranche_days")
    if not days:
        c = dict(cfg, exclude=excl) if excl else cfg
        r = X.run(MM, mat(c), start=s, end=e, record=True)
        return r["curve"], r["lots"]
    tot, lots = None, []
    for d in days:
        c = dict(cfg, decide_day=d, capital=10000.0 / len(days))
        c.pop("tranche_days")
        if excl:
            c["exclude"] = excl
        r = X.run(MM, mat(c), start=s, end=e, record=True)
        tot = r["curve"] if tot is None else tot.add(r["curve"], fill_value=0)
        lots.append(r["lots"])
    return tot, pd.concat(lots)
def _pair(a):
    cfg, s, e = a
    cv, L = curve(cfg, s, e)
    top = L.groupby("sym").pnl.sum().idxmax() if len(L) else None
    s1 = M.score(MM, cv); s1.pop("table")
    cv2, _ = curve(cfg, s, e, [top] if top else None)
    s2 = M.score(MM, cv2); s2.pop("table")
    for x in (s1, s2):
        x["expo"], x["costs"] = np.nan, np.nan
    return s1, s2, top
def evaluate(pool, cfgs, per):
    s0, e0, starts = R.PERIODS[per]
    res = pool.map(_pair, [(c, s, e0) for c in cfgs for s in starts], chunksize=1)
    k = len(starts)
    return [(R.agg([x[0] for x in res[n*k:(n+1)*k]]), R.agg([x[1] for x in res[n*k:(n+1)*k]]),
             sorted({x[2] for x in res[n*k:(n+1)*k] if x[2]})) for n in range(len(cfgs))]
if __name__ == "__main__":
    grid = [("System B (last session)", B)]
    for d in (0, 5, 10, 15):
        grid.append((f"decide on session {d + 1} of the month", dict(B, decide_day=d)))
    for t in ((-1, 10), (0, 10), (-1, 5, 10, 15), (0, 5, 10, 15)):
        grid.append((f"tranches deciding on sessions {[x + 1 if x >= 0 else 'last' for x in t]}", dict(B, tranche_days=list(t))))
    for na in ((8, 12, 0.6), (8, 15, 0.5), (6, 10, 0.6), (10, 15, 0.5), (12, 8, 0.6)):
        grid.append((f"slots {na[0]} if breadth >= {na[2]:.0%} else {na[1]}", dict(B, N_adapt=list(na))))
    pool = mp.get_context("fork").Pool(4, initializer=_init)
    rows = []
    res = {p: evaluate(pool, [c for _, c in grid], p) for p in ("train", "test", "early")}
    for n, (lab, c) in enumerate(grid):
        r = {"label": lab}
        for p in res:
            a, b, tops = res[p][n]
            r.update({f"{p}_score": (R.objective2(a) + R.objective2(b)) / 2, f"{p}_cagr": a["cagr"], f"{p}_dd": a["maxdd"],
                      f"{p}_beat": a["beat_all"], f"{p}_cagr_wo": b["cagr"]})
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("train_score", ascending=False)
    df.to_csv("../results/exp24_all.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat", "cagr_wo")]
    print(df[cols].round(3).to_string(index=False))
