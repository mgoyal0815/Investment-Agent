"""Leveraged parking with an account-level guard (owner, 16 Oct 2026: 2x allowed only if the worst fall does not get
worse). No 2x while the account is more than x below its own peak. Calendar-robust yardstick; R = 71.3%, worst -48%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    R = json.load(open("../system_R.json"))["cfg"]
    G = [(f"2x above {n}-day, off when account {int(x*100)}%+ below peak", dict(R, lev_park=n, lev_dd=x)) for n, x in ((200, 0.10), (200, 0.15), (50, 0.10), (50, 0.15), (200, 0.05))]
    heads = []
    for lab, c in G:
        h, Rr = RB.rscore(pool, c); Rr.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        Rr.to_csv("../audit/a59_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a59_r_lev_dd.csv", index=False)
