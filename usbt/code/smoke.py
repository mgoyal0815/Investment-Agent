import sys, time, pandas as pd
import engine as X, metrics as M
P = pd.read_pickle("../data/panel.pkl"); m = X.Market(P)
for lab, cfg in [("v5", {}), ("india", dict(universe="topN", score="nifty", sizing="slot", exit="rank", adds="india",
                 cadence="twice", overlay="pause", park="GLD", gold_slots=4, ramp=0, adv_cap=0.01, zone_gate=False))]:
    t = time.time(); r = X.run(m, cfg, start="2012-01-03", end="2018-12-31")
    print(lab, f"{time.time()-t:.1f}s", M.line(lab, M.score(m, r["curve"])), "expo", round(r["expo"].mean(),2))
