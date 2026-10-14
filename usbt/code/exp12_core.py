"""Experiment 12: index core with a fast trend exit, on the current best. A share core_w of the
account sits in the leading index ETF (SMH or QQQ by core_lb-session return) while that return is
above core_min and the ETF closes above its 50-session average; it exits at the next open when the
close falls below that average. The stock book is capped at the rest. All windows; pick by
runner.objective2 on 2012-2019. Writes results/exp12_all.csv, exp12_log.txt."""
import itertools, json
import pandas as pd
import runner as R
from exp2_search import materialise
from exp8_more import NEW

if __name__ == "__main__":
    cfgs = [("current best", NEW)]
    for w, mn, lb, sm in itertools.product((0.2, 0.3, 0.5, 0.7), (0.05, 0.15, 0.30), (63, 126), (1.0, 1.5)):
        cfgs.append((f"core {w:.0%} min {mn:.0%} lb {lb} size x{sm}", dict(NEW, core_w=w, core_min=mn, core_lb=lb, size_mult=sm)))
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
    df.to_csv("../results/exp12_all.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print(df[cols].head(12).round(3).to_string(index=False))
    print(df[df.label == "current best"][cols].round(3).to_string(index=False))
