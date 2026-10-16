"""System P, round 1 (owner, 16 Oct 2026): P loses 2006/2011/2023 on every calendar and is a coin flip in 2002, 2007,
2010, 2013, 2014, 2019, 2021, mostly by 0-5 points with 10-30% of days paused in short Treasuries. Tests aimed at that
cash drag and the pause, under the calendar-robust yardstick. P = 68.2%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    P = json.load(open("../system_P.json"))["cfg"]
    G = [("Paused money stays in leading index while it is above its 200-day", dict(P, park_sma=200)),
         ("... above its 100-day", dict(P, park_sma=100)), ("... above its 50-day", dict(P, park_sma=50)),
         ("Large-cap vote needs 2% below its 50-day", dict(P, large_margin=0.02)),
         ("Large-cap vote on 150-day average", dict(P, large_n=150)),
         ("Large-cap vote on 100-day average", dict(P, large_n=100)),
         ("Pause without breadth vote", dict(P, pause_rules=("hyg", "large")))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a41_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a41_p_pause.csv", index=False)
