"""New ideas on System Q (owner, 16 Oct 2026), calendar-robust yardstick, Q = 70.2%:
momentum quality ('frog in the pan'), momentum acceleration and residual momentum blended into Q's ranking
(score_mix weight w: rank = (Q rank + w x feature percentile) / (1 + w)); turn-of-month index holding during pauses.
(Short-term mean reversion, the fifth idea, was tested standalone in meanrev.py: <= 2% a year after costs, rejected.)"""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [(f"{nm} blended into rank, weight {w}", dict(Q, score_mix={k: w})) for k, nm in (("fip", "Momentum quality"), ("accel", "Acceleration"), ("resmom", "Residual momentum")) for w in (0.5, 1.0)]
    G += [("Turn-of-month: paused money holds the leading index", dict(Q, tom_park=True))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a47_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a47_new_ideas.csv", index=False)
