"""Neighbours of the lead from a41: the large-cap pause vote counts only when the large-cap index is x% below its
50-day average (P + 2% = 70.2% vs P 68.2%, calendar-robust). Plateau check and one combination."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    P = json.load(open("../system_P.json"))["cfg"]
    G = [(f"Large-cap vote needs {x*100:g}% below its 50-day", dict(P, large_margin=x)) for x in (0.01, 0.015, 0.025, 0.03)]
    G += [("2% margin + paused money in leading index above its 200-day", dict(P, large_margin=0.02, park_sma=200))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a42_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a42_p_margin.csv", index=False)
