"""Experiment 14: the profit lock (exp13's winner) tuned, with adds to winners and a short rank hold.
lock_at 0.3/0.5/0.75/1.0 x lock_trail 0.15/0.2/0.25/0.3 x adds None/v5 x rank_min_hold 0/21.
All windows; pick by runner.objective2 on 2012-2019. Writes results/exp14_all.csv."""
import itertools, json
import pandas as pd
import runner as R
from exp2_search import materialise
from exp8_more import NEW

if __name__ == "__main__":
    cfgs = [("no lock (previous best)", NEW)]
    for la, lt, ad, rh in itertools.product((0.3, 0.5, 0.75, 1.0), (0.15, 0.2, 0.25, 0.3), (None, "v5"), (0, 21)):
        cfgs.append((f"lock {la:.0%}/{lt:.0%}, adds {ad}, rank hold {rh}", dict(NEW, lock_at=la, lock_trail=lt, adds=ad, rank_min_hold=rh)))
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
    df.to_csv("../results/exp14_all.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print(df[cols].head(15).round(3).to_string(index=False))
    print(df[df.label == "no lock (previous best)"][cols].round(3).to_string(index=False))
