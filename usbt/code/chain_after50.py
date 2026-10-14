"""After exp50: take its best Bayesian pick (by 2012-16 starts), run the acceptance test vs System E; if accepted,
freeze it as System F and start round 2 (bayes_round.py) from it. Otherwise stop and report."""
import json, multiprocessing as mp, subprocess
import pandas as pd
import yardstick as Y, accept
import exp50_bayes_all as B

L = pd.read_csv("../results/exp50_optuna.csv").sort_values("obj", ascending=False)
base = {k: v[0] for k, v in B.F.items()}
best = next(json.loads(r.x) for _, r in L.iterrows() if json.loads(r.x) != base)
cand = B.build(best)
E = json.load(open("../system_E.json"))["cfg"]
Y._init("../data/panel_idx.pkl")
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel_idx.pkl",))
ok, checks, tab = accept.decide(pool, E, cand)
tab.to_csv("../results/exp50_accept.csv")
changes = {k: v for k, v in best.items() if v != base[k]}
print("CANDIDATE", json.dumps(changes)); print(tab.round(3).to_string()); print(checks)
if ok:
    json.dump(dict(name="System F", frozen="2026-10-15", description="System E + Bayesian round 1 (exp50) changes, accepted by accept.py",
                   changes=changes, cfg=cand), open("../system_F.json", "w"), indent=1, default=list)
    print("ROUND1 ACCEPT -> System F; round 2 after the range sweeps", flush=True)
    pool.close()
    pass   # round 2 now waits for the full-range sweeps (exp51) - owner, 15 Oct 2026
else:
    print("ROUND1 REJECT", flush=True)
