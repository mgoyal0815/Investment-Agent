"""Experiment 13: fix the leaks found in the 2012-2019 trade attribution (analyze_train.py).
  rank_min_hold  no rank exit before this many sessions (the rank exit killed 89% of positions
                 young, at a net loss): 0 / 21 / 44 / 63 / 88 / 126
  exit_score     "route": RobustPair within the name's route (a name that leaves its route has
                 no score and is sold); "pool": one RobustPair across the whole universe
  lock           None, or (lock_at, trail): once up lock_at, sell if it falls trail below its peak
  large_n        large-cap trend average for the buy pause: 50 / 100 / 200 sessions
All windows; pick by runner.objective2 on 2012-2019. Writes results/exp13_all.csv."""
import itertools, json
import pandas as pd
import runner as R
from exp2_search import materialise
from exp8_more import NEW

if __name__ == "__main__":
    cfgs = []
    for mh, es, lk, ln in itertools.product((0, 21, 44, 63, 88, 126), ("route", "pool"),
                                            (None, (0.5, 0.25), (1.0, 0.30)), (50, 100, 200)):
        c = dict(NEW, rank_min_hold=mh, exit_score=es, large_n=ln)
        if lk:
            c.update(lock_at=lk[0], lock_trail=lk[1])
        cfgs.append((f"rank hold {mh}, exit score {es}, lock {lk}, large {ln}", c))
    rn = R.Runner()
    res = {p: rn.eval_many([materialise(c) for _, c in cfgs], p) for p in ("train", "test", "early")}
    rows = []
    for k, (lab, c) in enumerate(cfgs):
        r = {"label": lab, "cfg": json.dumps(c, default=list)}
        for p in res:
            a = res[p][k]
            r.update({f"{p}_obj2": R.objective2(a), f"{p}_cagr": a["cagr"], f"{p}_dd": a["maxdd"], f"{p}_beat": a["beat_all"]})
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("train_obj2", ascending=False)
    df.to_csv("../results/exp13_all.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print(df[cols].head(15).round(3).to_string(index=False))
    print(df[df.label == "rank hold 0, exit score route, lock None, large 50"][cols].round(3).to_string(index=False))
