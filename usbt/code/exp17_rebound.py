"""Experiment 17: rebound mode for System A. After SPY falls rebound_trig from its 1-year high
(within the last 126 sessions) and closes back above its 50-session average, hold core_w of the
account in the leading index ETF (SPY/QQQ/SMH by core_lb-session return, if above its own
50-session average) and optionally loosen the Recovery cutoff. All windows; pick by objective2 on
2012-2019. Writes results/exp17_all.csv."""
import itertools, json
import pandas as pd
import runner as R
from exp2_search import materialise

A = json.load(open("../system_A.json"))["cfg"]
if __name__ == "__main__":
    cfgs = [("System A", A)]
    for trig, w, lb, rc, only in itertools.product((0.10, 0.15, 0.20), (0.3, 0.5, 0.7), (63, 126), (None, 0.83), (True, False)):
        cfgs.append((f"rebound {trig:.0%}, core {w:.0%}, lb {lb}, rec cut {rc}, {'QQQ/SMH only' if only else 'any index'}",
                     dict(A, core_mode="rebound", rebound_trig=trig, core_w=w, core_lb=lb, rec_cut_rebound=rc, core_only_smh_qqq=only)))
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
    df.to_csv("../results/exp17_all.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print(df[cols].head(15).round(3).to_string(index=False))
    print(df[df.label == "System A"][cols].round(3).to_string(index=False))
