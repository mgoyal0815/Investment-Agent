"""System Q, round 1 (owner, 16 Oct 2026): Q loses 2006 and 2011 on every calendar because its stocks fell while the
indexes rose (momentum reversals of May 2006 / Aug 2011). Tests on the stock picks, calendar-robust yardstick. Q = 70.2%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [(f"Skip the most volatile {int((1-v)*100)}% of stocks at entry", dict(Q, vol_max_pct=v)) for v in (0.9, 0.8, 0.7)]
    G += [("Add v5's ratcheting stop to the rank sell", dict(Q, exit="stop+rank")),
          ("Buy only with more volume on up days (UDVR >= 1)", dict(Q, udvr_min=1.0)),
          ("Buy only while 6-month strength is rising", dict(Q, rs_rising=True))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a44_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a44_q_picks.csv", index=False)
