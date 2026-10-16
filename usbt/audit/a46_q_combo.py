"""Combination of the small, flat gains from a45 (each within ~0.5 pt of Q): sell outside top 1.7x, OBV trend >= 1.0,
profit-lock trail 30%. Checks whether near-ties add up or are noise. Calendar-robust yardstick; Q = 70.2%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [("1.7x + OBV 1.0 + trail 30%", dict(Q, buffer=1.7, obv_min=1.0, lock_trail=0.3)),
         ("1.7x + trail 30%", dict(Q, buffer=1.7, lock_trail=0.3))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); heads.append(dict(test=lab, **h))
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a46_q_combo.csv", index=False)
