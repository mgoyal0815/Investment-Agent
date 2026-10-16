"""Placebo for ranking fragility (owner, 16 Oct 2026): a 5% blend of acceleration into Q's ranking cut its one run
2000-2025 from $2.63M to $0.99M. Blend pure random noise at small weights into Q's rank; if Q's honest score
(calendar-robust, 70.2%) drops sharply, Q's result also depends on its exact ranking, not only on its rules."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [(f"Random noise seed {sd}, weight 0.05", dict(Q, score_mix={f"noise{sd}": 0.05})) for sd in (1, 2, 3)]
    G += [("Random noise seed 1, weight 0.02", dict(Q, score_mix={"noise1": 0.02})), ("Random noise seed 1, weight 0.2", dict(Q, score_mix={"noise1": 0.2}))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a48_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a48_rank_placebo.csv", index=False)
