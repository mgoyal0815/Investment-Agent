"""2x parking with the fixes from a60 (owner, 16 Oct 2026: allowed only if the worst fall does not get worse).
The deepest 2x falls came from new accounts parked in 2x QQQ at the 2000 top: no 2x in a new account's first 126
sessions, and/or at most a capped share of the account in 2x. Calendar-robust yardstick; R = 71.3%, worst -48.1%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    R = json.load(open("../system_R.json"))["cfg"]
    G = [("2x above 200d, none in first 126 sessions", dict(R, lev_park=200, lev_after=126)),
         ("2x above 50d, none in first 126 sessions", dict(R, lev_park=50, lev_after=126)),
         ("2x above 200d, cap 25% + none in first 126", dict(R, lev_park=200, lev_cap=0.25, lev_after=126)),
         ("2x above 200d, cap 15% + none in first 126", dict(R, lev_park=200, lev_cap=0.15, lev_after=126)),
         ("2x above 200d, none in first 252 sessions", dict(R, lev_park=200, lev_after=252))]
    heads = []
    for lab, c in G:
        h, Rr = RB.rscore(pool, c); Rr.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        Rr.to_csv("../audit/a61_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a61_r_lev2.csv", index=False)
