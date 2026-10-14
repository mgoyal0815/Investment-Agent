"""Experiment 21: the same systems on a universe with no look-ahead: only stocks that were S&P 500
members on that day (point-in-time). Breadth and the large-cap index still use the whole panel.
Writes results/exp21_all.csv."""
import json, itertools
import pandas as pd
import runner as R
from exp2_search import materialise
from exp8_more import NEW

A = json.load(open("../system_A.json"))["cfg"]
if __name__ == "__main__":
    cfgs = [("US v5 live (its own universe)", {}), ("System A (its own universe)", A),
            ("v5 live, S&P 500 only", dict(universe="sp500")),
            ("System A, S&P 500 only", dict(A, universe="sp500")),
            ("previous best (no lock/adds), S&P 500 only", dict(NEW, universe="sp500")),
            ("Candidate 3 core, S&P 500 only", dict(NEW, universe="sp500", park=None))]
    for lo, rp, la in itertools.product((False, True), (1, 2), (None, 0.3, 0.5)):
        c = dict(A, universe="sp500", rank_losers_only=lo, rank_persist=rp)
        if la is None:
            c.pop("lock_at", None); c.pop("lock_trail", None)
        else:
            c.update(lock_at=la, lock_trail=0.25)
        cfgs.append((f"A on S&P 500: losers-only {lo}, persist {rp}, lock {la}", c))
    rn = R.Runner()
    res = {p: rn.eval_many([materialise(c) for _, c in cfgs], p) for p in ("train", "test", "early")}
    rows = []
    for k, (lab, c) in enumerate(cfgs):
        r = {"label": lab, "cfg": json.dumps(c, default=list)}
        for p in res:
            a = res[p][k]
            r.update({f"{p}_obj2": R.objective2(a), f"{p}_cagr": a["cagr"], f"{p}_dd": a["maxdd"], f"{p}_beat": a["beat_all"]})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("../results/exp21_all.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print(df[cols].round(3).to_string(index=False))
