"""Experiment 9: (a) the leader tilt on its own on top of the current best, all windows;
(b) coordinate-ascent search around the current best over a wider space, scored with
runner.objective2 on 2012-2019 only, then tested on 2020-2026 and 2000-2011.
Writes results/exp9_*.csv; stdout to exp9_log.txt."""
import itertools
import json
import pandas as pd
import engine as X, metrics as M, runner as R
from exp2_search import materialise
from exp8_more import NEW

SPACE = dict(
    tilt_mult=[1.0, 1.5, 2.0, 3.0], tilt_corr=[0.6, 0.7, 0.8], tilt_leaders=[("SMH",), ("SMH", "QQQ")],
    size_mult=[1.0, 1.25, 1.5], cont_cut=[0.80, 0.85, 0.88, 0.92], rec_cut=[0.83, 0.90, 0.95],
    min_hold=[44, 63, 88, 126], N=[20, 30, 40], leader_lb=[63, 126, 189, 252], breadth_level=[0.25, 0.30, 0.35],
    pause_rules=[("breadth", "hyg", "large"), ("hyg", "large")], universe_n=[300, 500, 800, 1000],
    adds=[None, "v5"], zone_gate=[False, True], cadence=["twice", "monthly"], park=["leader_safe", "leader", None],
)


def climb(rn, base, passes=2):
    cur = dict(base)
    a = rn.eval_many([materialise(cur)], "train")[0]
    best = R.objective2(a)
    for p in range(passes):
        improved = False
        for k, vals in SPACE.items():
            cands = [dict(cur, **{k: v}) for v in vals if v != cur.get(k)]
            res = rn.eval_many([materialise(c) for c in cands], "train")
            for c, r in zip(cands, res):
                if R.objective2(r) > best + 1e-6:
                    best, cur, a, improved = R.objective2(r), c, r, True
            print(f"pass{p} {k} -> {cur.get(k)}  {best:+.3f}", R.fmt("", a), flush=True)
        if not improved:
            break
    return cur


if __name__ == "__main__":
    rn = R.Runner()
    # (a) tilt only
    tilt = [("Current best", NEW)]
    for mult, corr, ld in itertools.product([1.5, 2.0, 3.0], [0.6, 0.7, 0.8], [("SMH",), ("SMH", "QQQ")]):
        tilt.append((f"best + tilt x{mult} corr>={corr} {'+'.join(ld)}", dict(NEW, tilt_mult=mult, tilt_corr=corr, tilt_leaders=ld)))
    rows = []
    for per in ("train", "test", "early"):
        for (lab, c), a in zip(tilt, rn.eval_many([materialise(c) for _, c in tilt], per)):
            rows.append(dict(period=per, label=lab, objective2=R.objective2(a), cfg=json.dumps(c, default=list), **a))
    df = pd.DataFrame(rows)
    df.to_csv("../results/exp9_tilt.csv", index=False)
    piv = df.pivot_table(index="label", columns="period", values=["objective2", "cagr", "maxdd", "beat_all"]).round(3)
    print(piv.sort_values(("objective2", "train"), ascending=False).head(8).to_string(), flush=True)
    # (b) local search from the current best, training window only
    best = climb(rn, NEW)
    json.dump(best, open("../results/exp9_best.json", "w"), default=list, indent=1)
    fin = [("Current best", NEW), ("Local search (train-chosen)", best)]
    rows = []
    for per in ("train", "test", "early"):
        for (lab, c), a in zip(fin, rn.eval_many([materialise(c) for _, c in fin], per)):
            print(per, f"{R.objective2(a):+.3f}", R.fmt(lab, a), flush=True)
            rows.append(dict(period=per, label=lab, objective2=R.objective2(a), **a))
    pd.DataFrame(rows).to_csv("../results/exp9_final.csv", index=False)
    m = X.Market(pd.read_pickle("../data/panel.pkl"))
    spans = (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None))
    yrs = {lab: pd.concat([M.yearly(X.run(m, materialise(c), start=s, end=e)["curve"]) for s, e in spans]) * 100 for lab, c in fin}
    full = m.dates[m.dates >= "2000-01-03"]
    for k, v in M.bench_curves(m, full).items():
        yrs[k] = pd.concat([M.yearly(v[s:e]) for s, e in spans]) * 100
    t = pd.DataFrame(yrs).round(1)
    t.to_csv("../results/exp9_years.csv")
    bestidx = t[["SPY", "QQQ", "SMH"]].max(axis=1)
    for lab, _ in fin:
        w = (t[lab] > bestidx) & (t.index < 2026)
        print(lab, "beat all 3:", int(w.sum()), "/26", [y for y in t.index if w[y]], flush=True)
