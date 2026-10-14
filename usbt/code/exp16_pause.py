"""Experiment 16: the pause costs System A its losing years (root_years_A.csv). Variants:
  pause_needs_bear  a pause only counts while SPY is below its 200-session average
  safe_when         "paused": idle cash goes to Treasuries while paused (System A);
                    "bear": idle cash goes to Treasuries only while SPY is below its 200-session average
  breadth_level     0.30 (System A) / 0.40 / 0.20
  pause rules       System A's three / HYG + large-cap only
All windows; pick by runner.objective2 on 2012-2019. Writes results/exp16_all.csv."""
import itertools, json
import pandas as pd
import runner as R
from exp2_search import materialise

A = json.load(open("../system_A.json"))["cfg"]
if __name__ == "__main__":
    cfgs = []
    for nb, sw, bl, pr in itertools.product((False, True), ("paused", "bear"), (0.20, 0.30, 0.40),
                                            (("breadth", "hyg", "large"), ("hyg", "large"))):
        cfgs.append((f"needs bear {nb}, Treasuries when {sw}, breadth {bl:.0%}, rules {'+'.join(pr)}",
                     dict(A, pause_needs_bear=nb, safe_when=sw, breadth_level=bl, pause_rules=pr)))
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
    df.to_csv("../results/exp16_all.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print(df[cols].round(3).to_string(index=False))
