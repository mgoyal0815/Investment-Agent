"""Experiment 22: the whole search again on the bias-free universe (point-in-time S&P 500
members). Random search over both families (v5 sizing and India-style slots, with index parking,
profit lock, adds, pause variants, rank-exit variants), then coordinate ascent from the best 3,
scored with runner.objective2 on 2012-2019 only; finalists tested on 2020-2026 and 2000-2011.
Writes results/exp22_*."""
import json, random, sys
import pandas as pd
import runner as R
from exp2_search import materialise

A = json.load(open("../system_A.json"))["cfg"]
SV5 = dict(overlay=["pause", "v5", "none"], intramonth=[True, False], zone_gate=[True, False], cadence=["daily", "twice", "monthly"],
           park=[None, "leader", "leader_safe", "SHY"], adds=[None, "v5"], exit=["stop", "stop+rank"], min_hold=[44, 88, 126],
           cont_cut=[0.80, 0.88, 0.95], rec_cut=[0.83, 0.95], N=[10, 20, 30], lock_at=[None, 0.3, 0.5, 1.0], lock_trail=[0.2, 0.25, 0.3],
           rank_losers_only=[False, True], rank_persist=[1, 2], breadth_level=[0.2, 0.3, 0.4], pause_mode=["any", "two"],
           base_mult=[1.0, 1.5, 2.0], risk_mult=[1.0, 1.5, 2.0], vol_mult=[1.0, 2.0], add_red_hi=[None, 3.0], ramp=[0, 3])
SSL = dict(N=[5, 8, 10, 12, 15, 20], score=["nifty", "robust"], trend=[True, False], exit=["rank", "stop+rank"], buffer=[1.0, 1.5, 2.0, 3.0],
           adds=["india", None], cadence=["twice", "monthly", "daily"], overlay=["pause", "v5", "none"], intramonth=[True, False],
           park=[None, "GLD", "SHY", "leader", "leader_safe"], gold_slots=[0, 2, 4], ramp=[0, 3], zone_gate=[False, True],
           rank_min_hold=[0, 21, 63], lock_at=[None, 0.3, 0.5, 1.0], lock_trail=[0.2, 0.25, 0.3], breadth_level=[0.2, 0.3, 0.4])


def sample(rng):
    if rng.random() < 0.5:
        c, sp = dict(sizing="v5"), SV5
    else:
        c, sp = dict(sizing="slot"), SSL
    for k, v in sp.items():
        c[k] = rng.choice(v)
    c["universe"] = "sp500"
    return c


def mat(c):
    c = materialise(c)
    if c.get("lock_at") is None:
        c.pop("lock_at", None)
    return c


def climb(rn, cur):
    sp = SSL if cur.get("sizing") == "slot" else SV5
    best = R.objective2(rn.eval_many([mat(cur)], "train")[0])
    for p in range(2):
        imp = False
        for k, vals in sp.items():
            cands = [dict(cur, **{k: v}) for v in vals if v != cur.get(k)]
            for c, a in zip(cands, rn.eval_many([mat(c) for c in cands], "train")):
                if R.objective2(a) > best + 1e-6:
                    best, cur, imp = R.objective2(a), c, True
        print(f"  pass{p} {best:+.3f}", flush=True)
        if not imp:
            break
    return cur, best


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = random.Random(23)
    seeds = [dict(A, universe="sp500"), dict(universe="sp500"), dict(R.INDIA, universe="sp500")]
    raw = seeds + [sample(rng) for _ in range(n)]
    rn = R.Runner()
    res = []
    for k in range(0, len(raw), 50):
        ch = raw[k:k + 50]
        for c, a in zip(ch, rn.eval_many([mat(c) for c in ch], "train")):
            res.append(dict(cfg=json.dumps(c, default=list), obj2=R.objective2(a), **a))
        df = pd.DataFrame(res).sort_values("obj2", ascending=False)
        df.to_csv("../results/exp22_all.csv", index=False)
        print(k + len(ch), "best", round(df.obj2.iloc[0], 3), R.fmt("", df.iloc[0]), flush=True)
    fin = [("System A on S&P 500", seeds[0]), ("v5 live on S&P 500", seeds[1]), ("India v8 on S&P 500", seeds[2])]
    for i, c in enumerate(json.loads(x) for x in df.head(3).cfg):
        cc, o = climb(rn, c)
        fin.append((f"S&P 500 pick {i + 1} (train {o:+.3f})", cc))
    json.dump([{"label": l, "cfg": c} for l, c in fin], open("../results/exp22_finalists.json", "w"), default=list, indent=1)
    rows = []
    for p in ("train", "test", "early"):
        for (lab, c), a in zip(fin, rn.eval_many([mat(c) for _, c in fin], p)):
            print(p, f"{R.objective2(a):+.3f}", R.fmt(lab, a), flush=True)
            rows.append(dict(period=p, label=lab, obj2=R.objective2(a), **a))
    pd.DataFrame(rows).to_csv("../results/exp22_final.csv", index=False)
