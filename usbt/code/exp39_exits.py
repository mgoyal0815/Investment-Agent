"""Experiment 39: how System C protects gains, adds, holds, exits and sizes. New options (engine):
rank_ladder (sell half when out of the top 15, the rest out of the top rank_ladder*N), lock_half +
lock_trail2 (half out at the profit lock, the rest on a wider trail), lock_tiers (bigger winners may give
back more), rank_exempt_gain (big winners leave only on the lock), scale_in (ladder up), weight
(rank / inverse volatility / momentum sized slots). Coordinate ascent from System C on the owner's
yardstick, starts 2012-2016 stopping 2019 only; finalists judged on starts 2000-2008 (to 2011) and
2020-2025 (to today) and every monthly start 2000-2025."""
import json, multiprocessing as mp
import pandas as pd
import yardstick as Y
from exp34_yard_search import score

SC = json.load(open("../system_C.json"))["cfg"]
SPACE = dict(rank_ladder=[None, 2, 3, 4], lock_half=[False, True], lock_trail2=[0.3, 0.4, 0.5],
             lock_tiers=[None, [[1.0, 0.35]], [[0.6, 0.3], [1.5, 0.45]]], rank_exempt_gain=[None, 0.5, 1.0, 2.0],
             scale_in=[None, [0.5, 0.1, 1], [1 / 3, 0.1, 2], [0.5, 0.2, 1]], weight=[None, "rank", "invvol", "mom"],
             lock_at=[None, 0.2, 0.3, 0.5], lock_trail=[0.15, 0.2, 0.25, 0.3], buffer=[1.0, 1.5, 2.0, 3.0],
             add_gain=[0.1, 0.2, 0.3], add_cap=[2.0, 3.0, 4.0], N=[8, 10, 12])
PER = None


def ev(pool, c, p):
    st, end = PER[p]
    return Y.summary(Y.evaluate(pool, c, st, end))


if __name__ == "__main__":
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 2), "2019-12-31"),
           "test 2000-08": (Y.starts("2000-01-01", "2008-12-31", 2), "2011-12-30"),
           "test 2020-25": (Y.starts("2020-01-01", "2025-10-31", 1), None),
           "all starts": (Y.starts(), None)}
    log = []
    # 1) one change at a time from System C (what each idea does on its own)
    base = score(ev(pool, SC, "train"))
    print("System C train score", round(base, 3), flush=True)
    for k, vals in SPACE.items():
        for v in vals:
            if v == SC.get(k) or (v is None and k not in SC):
                continue
            c = dict(SC, **{k: v})
            s = ev(pool, c, "train")
            log.append(dict(step="single", key=k, val=json.dumps(v), score=score(s), beat=s["yrs_beat_SPY_QQQ"], smh3=s["r3_beat_SMH"],
                            dd_ok=s["dd_ok"], cagr=s["med_cagr"]))
            print("single", k, v, round(score(s), 3), round(s["yrs_beat_SPY_QQQ"], 3), round(s["med_cagr"], 3), flush=True)
    pd.DataFrame(log).to_csv("../results/exp39_single.csv", index=False)
    # 2) coordinate ascent (2 passes)
    cur, best = dict(SC), base
    for p in range(2):
        imp = False
        for k, vals in SPACE.items():
            for v in vals:
                if v == cur.get(k):
                    continue
                c = dict(cur, **{k: v})
                s = score(ev(pool, c, "train"))
                if s > best + 0.005:
                    best, cur, imp = s, c, True
                    print("  better", k, v, round(best, 3), flush=True)
        if not imp:
            break
    json.dump(cur, open("../results/exp39_best_cfg.json", "w"), indent=1, default=list)
    # 3) judge System C and the tuned one everywhere
    rows = []
    for lab, c in (("System C", SC), ("exp39 tuned", cur)):
        for p in PER:
            s = ev(pool, c, p)
            s.update(label=lab, period=p, score=score(s)); rows.append(s)
            print(p, lab, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(rows).to_csv("../results/exp39_final.csv", index=False)
