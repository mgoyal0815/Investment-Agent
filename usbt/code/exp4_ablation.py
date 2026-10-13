"""Experiment 4: which parts of Candidate 3 matter? Remove each change from US v5 one at a time,
on all three windows. Also add each to v5 alone. Writes results/exp4_ablation.csv."""
import json
import pandas as pd
import runner as R
from exp2_search import materialise

C3 = json.loads(pd.read_csv("../results/exp2_top.csv").iloc[2]["cfg"])
PARTS = {
    "top-500 universe": (dict(universe="topN", universe_n=500), dict(universe="adv50")),
    "India buy pause (no regime overlay)": (dict(overlay="pause", intramonth=False), dict(overlay="v5", intramonth=True)),
    "decide twice a month": (dict(cadence="twice"), dict(cadence="daily")),
    "rank sell (outside top 45)": (dict(exit="stop+rank", N=30), dict(exit="stop")),
    "no v5 adds": (dict(adds=None), dict(adds="v5")),
    "no pullback-band entry": (dict(zone_gate=False), dict(zone_gate=True)),
    "recovery cutoff 0.95": (dict(rec_cut=0.95), dict(rec_cut=0.83)),
}
if __name__ == "__main__":
    rn = R.Runner()
    cfgs = [("Candidate 3", C3), ("US v5 (live)", {})]
    for k, (on, off) in PARTS.items():
        cfgs.append((f"C3 without {k}", dict(C3, **off)))
        cfgs.append((f"v5 with only {k}", dict(R.V5, **on)))
    rows = []
    for period in ("train", "test", "early"):
        res = rn.eval_many([materialise(c) for _, c in cfgs], period)
        for (lab, c), a in zip(cfgs, res):
            print(period, R.fmt(lab, a), flush=True)
            rows.append(dict(period=period, label=lab, **a))
    pd.DataFrame(rows).to_csv("../results/exp4_ablation.csv", index=False)
