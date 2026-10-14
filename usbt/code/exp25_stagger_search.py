"""Experiment 25: the clean-universe search again with every configuration staggered inside ONE
account (part of the book reviewed on each of several sessions of the month) and scored without date
luck or single-stock luck: mean objective2 with and without the biggest winner, 2012-2019, five starts.
Random search over the slot family, coordinate ascent from the best two, finalists on all windows."""
import json, random, sys, multiprocessing as mp
import pandas as pd
import runner as R
from exp24_timing import evaluate, _init, mat

B = json.load(open("../system_B.json"))["cfg"]
SP = dict(N=[6, 8, 10, 12, 15, 20], score=["nifty", "robust"], trend=[True, False], exit=["rank", "stop+rank"],
          buffer=[1.5, 2.0, 3.0], adds=["india", None], overlay=["pause", "v5", "none"], park=[None, "SHY", "leader", "leader_safe", "GLD"],
          gold_slots=[0, 2, 4], zone_gate=[False, True], rank_min_hold=[0, 21, 63], lock_at=[None, 0.3, 0.5, 1.0],
          lock_trail=[0.2, 0.25, 0.3], breadth_level=[0.2, 0.3, 0.4], stagger=[[-1, 10], [-1, 5, 10, 15], [0, 10]],
          add_gain=[0.1, 0.2, 0.3], add_cap=[2.0, 3.0])
def sample(rng):
    c = dict(sizing="slot", universe="sp500", cadence="monthly", ramp=3)
    for k, v in SP.items():
        c[k] = rng.choice(v)
    return c
def score(a_b):
    a, b, _ = a_b
    return (R.objective2(a) + R.objective2(b)) / 2
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    rng = random.Random(31)
    cfgs = [dict(B, stagger=[-1, 10]), dict(B, stagger=[-1, 5, 10, 15])] + [sample(rng) for _ in range(n)]
    pool = mp.get_context("fork").Pool(4, initializer=_init)
    res = []
    for k in range(0, len(cfgs), 40):
        ch = cfgs[k:k + 40]
        for c, ab in zip(ch, evaluate(pool, ch, "train")):
            res.append(dict(cfg=json.dumps(c, default=list), score=score(ab), cagr=ab[0]["cagr"], dd=ab[0]["maxdd"],
                            beat=ab[0]["beat_all"], cagr_wo=ab[1]["cagr"]))
        df = pd.DataFrame(res).sort_values("score", ascending=False)
        df.to_csv("../results/exp25_all.csv", index=False)
        print(k + len(ch), "best", df.iloc[0][["score", "cagr", "dd", "beat", "cagr_wo"]].round(3).to_dict(), flush=True)
    picks = []
    for c0 in [json.loads(x) for x in df.head(2).cfg]:
        cur, best = c0, score(evaluate(pool, [c0], "train")[0])
        for p in range(2):
            imp = False
            for key, vals in SP.items():
                cands = [dict(cur, **{key: v}) for v in vals if v != cur.get(key)]
                for c, ab in zip(cands, evaluate(pool, cands, "train")):
                    if score(ab) > best + 1e-6:
                        best, cur, imp = score(ab), c, True
            print(f"  climb pass{p} {best:+.3f}", flush=True)
            if not imp:
                break
        picks.append(cur)
    fin = [("System B, one decision day", B), ("System B staggered last+11", dict(B, stagger=[-1, 10]))] + \
          [(f"staggered pick {i + 1}", c) for i, c in enumerate(picks)]
    json.dump([{"label": l, "cfg": c} for l, c in fin], open("../results/exp25_finalists.json", "w"), default=list, indent=1)
    rows = []
    for per in ("train", "test", "early"):
        for (lab, c), (a, b, tops) in zip(fin, evaluate(pool, [c for _, c in fin], per)):
            print(per, R.fmt(lab[:44], a), f"| w/o top winner {b['cagr']:.1%} ({','.join(tops)})", flush=True)
            rows.append(dict(period=per, label=lab, cagr=a["cagr"], dd=a["maxdd"], beat=a["beat_all"], cagr_wo=b["cagr"], dd_wo=b["maxdd"]))
    pd.DataFrame(rows).to_csv("../results/exp25_final.csv", index=False)
