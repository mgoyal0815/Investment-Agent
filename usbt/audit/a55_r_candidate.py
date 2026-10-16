"""Candidate System R (owner, 16 Oct 2026): Q + the two gains that held up honestly - no rank sells while SPY is deep
below its 1-year high (plateau 15-25%: same win rate, worst fall -46/-47% vs -51%) and the giants slot drawn from the
top 10 most-traded instead of 20 (71.2%). Combination and neighbours, calendar-robust yardstick, Q = 70.2%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [("R: crash 20% + giants top 10", dict(Q, crash_no_rank=0.20, giant_top=10)),
         ("crash 15% + giants top 10", dict(Q, crash_no_rank=0.15, giant_top=10)),
         ("crash 25% + giants top 10", dict(Q, crash_no_rank=0.25, giant_top=10)),
         ("crash 20% + giants top 15", dict(Q, crash_no_rank=0.20, giant_top=15))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a55_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a55_r_candidate.csv", index=False)
