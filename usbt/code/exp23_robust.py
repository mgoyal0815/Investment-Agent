"""Experiment 23: luck-robust search around System B. Each configuration is scored on 2012-2019 as the
mean of runner.objective2 with all stocks and with its single biggest winner (in that start path)
removed, so a setting that only works because it held NVDA in 2016-17 does not win. Settings that
govern catching and keeping big winners: number of slots, exit width, decision cadence, gold slots,
idle-cash parking, adds, profit lock. Finalists tested on 2020-2026 and 2000-2011, also without
their top winner. Writes results/exp23_*."""
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
def _pair(a):
    cfg, s, e = a
    r = X.run(MM, mat(cfg), start=s, end=e, record=True)
    top = r["lots"].groupby("sym").pnl.sum().idxmax() if len(r["lots"]) else None
    s1 = M.score(MM, r["curve"]); s1.pop("table")
    c2 = dict(cfg, exclude=[top]) if top else cfg
    s2 = M.score(MM, X.run(MM, mat(c2), start=s, end=e)["curve"]); s2.pop("table")
    for x in (s1, s2):
        x["expo"], x["costs"] = np.nan, np.nan
    return s1, s2, top
def evaluate(pool, cfgs, per):
    s0, e0, starts = R.PERIODS[per]
    res = pool.map(_pair, [(c, s, e0) for c in cfgs for s in starts], chunksize=1)
    out = []
    k = len(starts)
    for n in range(len(cfgs)):
        rs = res[n * k:(n + 1) * k]
        a, b = R.agg([x[0] for x in rs]), R.agg([x[1] for x in rs])
        out.append((a, b, sorted({x[2] for x in rs if x[2]})))
    return out

if __name__ == "__main__":
    grid = []
    for N, buf, cad, gs, pk, ad, lk in itertools.product((8, 10, 15, 20), (1.5, 2.0, 3.0), ("monthly", "twice"), (0, 4),
                                                          ("leader_safe", None), ("india", None), (None, 0.3)):
        c = dict(B, N=N, buffer=buf, cadence=cad, gold_slots=gs, park=pk, adds=ad, lock_at=lk)
        grid.append((f"N {N}, exit top {int(N*buf)}, {cad}, gold {gs}, park {pk}, adds {ad}, lock {lk}", c))
    pool = mp.get_context("fork").Pool(4, initializer=_init)
    tr = evaluate(pool, [c for _, c in grid], "train")
    rows = []
    for (lab, c), (a, b, tops) in zip(grid, tr):
        rows.append(dict(label=lab, cfg=json.dumps(c, default=list), score=(R.objective2(a) + R.objective2(b)) / 2,
                         cagr=a["cagr"], dd=a["maxdd"], beat=a["beat_all"], cagr_wo_top=b["cagr"], dd_wo_top=b["maxdd"],
                         top=",".join(tops)))
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv("../results/exp23_train.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(df.drop(columns="cfg").head(10).round(3).to_string(index=False), flush=True)
    fin = [("System B", B)] + [(r.label, json.loads(r.cfg)) for _, r in df.head(3).iterrows()]
    out = []
    for per in ("train", "test", "early"):
        for (lab, c), (a, b, tops) in zip(fin, evaluate(pool, [c for _, c in fin], per)):
            print(per, R.fmt(lab[:44], a), f"| without top winner: CAGR {b['cagr']:.1%} DD {b['maxdd']:.1%} ({','.join(tops)})", flush=True)
            out.append(dict(period=per, label=lab, cagr=a["cagr"], dd=a["maxdd"], beat=a["beat_all"], cagr_wo=b["cagr"], dd_wo=b["maxdd"]))
    pd.DataFrame(out).to_csv("../results/exp23_final.csv", index=False)
