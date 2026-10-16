"""S2 = R + 2x parking only through QLD / SSO (when SMH leads, idle money stays in 1x SMH): the real USD fund lost 24%
in Apr 2010-Dec 2011 while SMH rose 10% (a63 trace). Real fund prices, calendar-robust, day-1 and held-back starts."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    R = json.load(open("../system_R.json"))["cfg"]
    S2 = dict(R, lev_park=200, lev_after=126, lev_real=True, lev_only=["SPY", "QQQ"])
    heads = []
    for lab, c, day in (("S2 real QLD/SSO only, day-1 starts", S2, 1), ("S2 real QLD/SSO only, held-back 15th starts", S2, 15)):
        h, Rr = RB.rscore(pool, c, day=day); Rr.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        Rr.to_csv("../audit/a64_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a64_s2.csv", index=False)
