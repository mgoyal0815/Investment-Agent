"""Experiment 10 (owner's option 1): worst fall may be as deep as SMH's. Random search over the
whole combined space (v5 family and India-slot family, index parking, leader tilt, bigger
positions, fewer holdings, regime weights), then coordinate ascent from the 3 best. Scored with
runner.objective3 on 2012-2019 only; finalists tested on 2020-2026 and 2000-2011."""
import json, random, sys
import pandas as pd
import engine as X, metrics as M, runner as R
from exp2_search import SPACE_SLOT, SPACE_V5, materialise
from exp8_more import NEW, HYB

EXTRA = dict(park=[None, "GLD", "SHY", "leader", "leader_safe"], leader_lb=[63, 126, 252],
             tilt_mult=[1.0, 2.0, 3.0], tilt_corr=[0.6, 0.7], tilt_leaders=[("SMH",), ("SMH", "QQQ")],
             breadth_level=[0.25, 0.30, 0.40], pause_mode=["any", "two"])
SV5 = dict(SPACE_V5, **EXTRA, size_mult=[1.0, 1.5, 2.0], max_pos=[None, 20, 12], N=[20, 30, 40],
           min_hold=[44, 88, 126], universe_n=[200, 300, 500, 1000])
SSL = dict(SPACE_SLOT, **{k: v for k, v in EXTRA.items() if not k.startswith("tilt")})


def sample(rng):
    if rng.random() < 0.5:
        c = dict(sizing="slot", universe="topN")
        sp = SSL
    else:
        c = dict(sizing="v5", exit="stop+rank")
        sp = SV5
    for k, v in sp.items():
        c[k] = rng.choice(v)
    return c


def climb(rn, cur, passes=2):
    sp = SSL if cur.get("sizing") == "slot" else SV5
    a = rn.eval_many([materialise(cur)], "train")[0]
    best = R.objective3(a)
    for p in range(passes):
        imp = False
        for k, vals in sp.items():
            cands = [dict(cur, **{k: v}) for v in vals if v != cur.get(k)]
            for c, r in zip(cands, rn.eval_many([materialise(c) for c in cands], "train")):
                if R.objective3(r) > best + 1e-6:
                    best, cur, a, imp = R.objective3(r), c, r, True
        print(f"  pass{p} {best:+.3f}", R.fmt("", a), flush=True)
        if not imp:
            break
    return cur, best


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = random.Random(11)
    raw = [NEW, HYB, {}] + [sample(rng) for _ in range(n)]
    rn = R.Runner()
    res = []
    for k in range(0, len(raw), 50):
        ch = raw[k:k + 50]
        for c, a in zip(ch, rn.eval_many([materialise(c) for c in ch], "train")):
            res.append(dict(cfg=json.dumps(c, default=list), objective3=R.objective3(a), **a))
        df = pd.DataFrame(res).sort_values("objective3", ascending=False)
        df.to_csv("../results/exp10_all.csv", index=False)
        print(k + len(ch), "best", round(df.objective3.iloc[0], 3), R.fmt("", df.iloc[0]), flush=True)
    seeds = [json.loads(c) for c in df.head(3).cfg]
    fin = [("Current best", NEW)]
    for n_, s in enumerate(seeds):
        c, o = climb(rn, s)
        fin.append((f"Option-1 pick {n_ + 1} (train {o:+.3f})", c))
    json.dump([{"label": l, "cfg": c} for l, c in fin], open("../results/exp10_finalists.json", "w"), default=list, indent=1)
    rows = []
    for per in ("train", "test", "early"):
        for (lab, c), a in zip(fin, rn.eval_many([materialise(c) for _, c in fin], per)):
            print(per, f"{R.objective3(a):+.3f}", R.fmt(lab, a), f"SMH-dd ok {a['dd_ok_SMH']:.0%}", flush=True)
            rows.append(dict(period=per, label=lab, objective3=R.objective3(a), **a))
    pd.DataFrame(rows).to_csv("../results/exp10_final.csv", index=False)
    m = X.Market(pd.read_pickle("../data/panel.pkl"))
    spans = (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None))
    yrs = {lab: pd.concat([M.yearly(X.run(m, materialise(c), start=s, end=e)["curve"]) for s, e in spans]) * 100 for lab, c in fin}
    full = m.dates[m.dates >= "2000-01-03"]
    for k, v in M.bench_curves(m, full).items():
        yrs[k] = pd.concat([M.yearly(v[s:e]) for s, e in spans]) * 100
    t = pd.DataFrame(yrs).round(1)
    t.to_csv("../results/exp10_years.csv")
    b = t[["SPY", "QQQ", "SMH"]].max(axis=1)
    for lab, _ in fin:
        w = (t[lab] > b) & (t.index < 2026)
        c = lambda a, z: int(w[(t.index >= a) & (t.index <= z)].sum())
        print(f"{lab:34s} beat all 3: 2000-11 {c(2000, 2011)}/12  2012-19 {c(2012, 2019)}/8  2020-25 {c(2020, 2025)}/6  total {int(w.sum())}/26", flush=True)
    print(t.to_string())
