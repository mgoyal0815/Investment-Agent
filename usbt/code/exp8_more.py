"""Experiment 8: three more levers, judged by runner.objective2 (growth high, drawdown low, beat
the indexes), fixed before any test window was scored.

  A  weekly decisions instead of twice a month
  B  two systems side by side, $5,000 each, rebalanced to 50/50 every January
  C  walk-forward: each January, pick from a menu of 59 configurations the one with the best
     objective2 on all PAST years only, and run it for the coming year

Windows as before: train 2012-2019, test 2020-2026 and 2000-2011, five start dates each for A and B.
Writes results/exp8_*.csv and exp8_log.txt (stdout)."""
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import engine as X, metrics as M, runner as R
from exp2_search import materialise
from exp6_c3_grid import C3
from exp7_etf_pause import grid as grid7

T7 = pd.read_csv("../results/exp7_train.csv")
NEW = json.loads(T7[T7.label == "C3+index unless paused, no ETFs, pause any, breadth 30%"].iloc[0].cfg)
HYB = json.loads(open("../results/exp2b_best.json").read())["balanced"]["cfg"]
MM = None


def _init():
    global MM
    MM = X.Market(pd.read_pickle("../data/panel.pkl"))


def _run(args):
    cfg, start, end, cap = args
    r = X.run(MM, materialise(dict(cfg, capital=cap)), start=start, end=end)
    return r["curve"]


def combine(a, b):
    """50/50, rebalanced on the first session of each year, from two curves."""
    ra, rb = a.pct_change().fillna(0), b.pct_change().fillna(0)
    nav, wa, wb, out, yr = 1.0, 0.5, 0.5, [], None
    for d in a.index:
        if d.year != yr:
            yr, wa, wb = d.year, 0.5 * nav, 0.5 * nav
        wa *= 1 + ra[d]
        wb *= 1 + rb[d]
        nav = wa + wb
        out.append(nav)
    return pd.Series(out, index=a.index) * 10000


def pair_eval(pool, ca, cb, per):
    s0, e0, starts = R.PERIODS[per]
    ra = pool.map(_run, [(ca, s, e0, 5000.0) for s in starts])
    rb = pool.map(_run, [(cb, s, e0, 5000.0) for s in starts])
    out = []
    for x, y in zip(ra, rb):
        sc = M.score(MM_main, combine(x, y))
        sc.pop("table")
        sc["expo"], sc["costs"] = np.nan, np.nan
        out.append(sc)
    return R.agg(out)


def walk_forward(pool, menu, start, end, first_year):
    curves = pool.map(_run, [(c, start, end, 10000.0) for _, c in menu])
    yrs = [M.yearly(c) for c in curves]
    picks, rets = [], []
    last = curves[0].index[-1].year
    for Y in range(first_year, last + 1):
        best, bi = -1e9, None
        for k, c in enumerate(curves):
            past = c[c.index.year < Y]
            y = yrs[k][yrs[k].index < Y]
            b = M.bench_curves(MM_main, past.index)
            yb = {n: M.yearly(v) for n, v in b.items()}
            ba = float(((y > yb["SPY"]) & (y > yb["QQQ"]) & (y > yb["SMH"])).mean())
            o = M.cagr(past) + 0.5 * M.maxdd(past) + 0.1 * ba
            if o > best:
                best, bi = o, k
        picks.append((Y, menu[bi][0]))
        seg = curves[bi][curves[bi].index.year == Y]
        prev = curves[bi][curves[bi].index.year < Y].iloc[-1]
        rets.append(seg / prev)
    nav, parts = 10000.0, []
    for r in rets:
        parts.append(r * nav)
        nav = parts[-1].iloc[-1]
    return pd.concat(parts), picks


if __name__ == "__main__":
    MM_main = X.Market(pd.read_pickle("../data/panel.pkl"))
    rn = R.Runner()
    pool = mp.get_context("fork").Pool(4, initializer=_init)
    rows = []
    # ---- A: weekly
    A = [("New best (twice a month)", NEW), ("New best, weekly", dict(NEW, cadence="weekly")),
         ("Candidate 3", C3), ("Candidate 3, weekly", dict(C3, cadence="weekly")),
         ("US v5 (live)", {}), ("Hybrid A", HYB)]
    for per in ("train", "test", "early"):
        for (lab, c), a in zip(A, rn.eval_many([materialise(c) for _, c in A], per)):
            print(per, f"{R.objective2(a):+.3f}", R.fmt(lab, a), flush=True)
            rows.append(dict(part="A", period=per, label=lab, objective2=R.objective2(a), **a))
    # ---- B: 50/50 pairs
    B = [("New best + Hybrid A", NEW, HYB), ("New best + Candidate 3", NEW, C3), ("New best + v5 live", NEW, {})]
    for per in ("train", "test", "early"):
        for lab, ca, cb in B:
            a = pair_eval(pool, ca, cb, per)
            print(per, f"{R.objective2(a):+.3f}", R.fmt(lab, a), flush=True)
            rows.append(dict(part="B", period=per, label=lab, objective2=R.objective2(a), **a))
    pd.DataFrame(rows).to_csv("../results/exp8_ab.csv", index=False)
    # ---- C: walk-forward re-selection
    menu = [(l, c) for l, c in grid7()] + [("US v5 (live)", {}), ("India v8", R.INDIA), ("Hybrid A", HYB)]
    wf = {}
    for name, s, e, fy in (("2012 run", "2012-01-03", None, 2015), ("2000 run", "2000-01-03", "2011-12-30", 2003)):
        curve, picks = walk_forward(pool, menu, s, e, fy)
        sc = M.score(MM_main, curve)
        t = sc.pop("table")
        wf[name] = (curve, picks, sc, t)
        print(f"walk-forward {name} from {fy}: " + M.line("re-chosen each January", sc), flush=True)
        for Y, l in picks:
            print(f"   {Y}: {l}", flush=True)
        t.to_csv(f"../results/exp8_walkforward_{name[:4]}.csv")
    # yearly table for the main options + walk-forward
    spans = (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None))
    yrs = {}
    for lab, c in (("New best", NEW), ("New best, weekly", dict(NEW, cadence="weekly")), ("Candidate 3", C3), ("US v5 (live)", {})):
        yrs[lab] = pd.concat([M.yearly(x) for x in pool.map(_run, [(c, s, e, 10000.0) for s, e in spans])]) * 100
    parts = []
    for s, e in spans:
        x = pool.map(_run, [(NEW, s, e, 5000.0), (HYB, s, e, 5000.0)])
        parts.append(M.yearly(combine(*x)))
    yrs["New best + Hybrid A"] = pd.concat(parts) * 100
    full = MM_main.dates[MM_main.dates >= "2000-01-03"]
    for k, v in M.bench_curves(MM_main, full).items():
        yrs[k] = pd.concat([M.yearly(v[s:e]) for s, e in spans]) * 100
    t = pd.DataFrame(yrs).round(1)
    t.index.name = "year"
    t.to_csv("../results/exp8_years.csv")
    print(t.to_string())
