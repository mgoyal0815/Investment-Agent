"""Experiment 20: the rank exit sells re-bought winners on a short dip (NVDA Dec 2023, SMCI).
rank_persist 1/2/3 consecutive reviews outside the top 1.5N; rank_losers_only; N 30/40; and the
universe width 500/800/1000 crossed with them. All windows. Writes results/exp20_all.csv."""
import itertools, json
import pandas as pd
import runner as R
from exp2_search import materialise

A = json.load(open("../system_A.json"))["cfg"]
if __name__ == "__main__":
    cfgs = []
    for rp, lo, n, un in itertools.product((1, 2, 3), (False, True), (30, 40), (500, 800, 1000)):
        cfgs.append((f"persist {rp}, losers only {lo}, exit top {int(1.5*n)}, universe {un}",
                     dict(A, rank_persist=rp, rank_losers_only=lo, N=n, universe_n=un)))
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
    df.to_csv("../results/exp20_all.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print(df[cols].round(3).to_string(index=False))
