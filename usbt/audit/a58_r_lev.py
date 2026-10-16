"""Leveraged index ETFs in System R (owner, 16 Oct 2026: allowed only if the worst fall does not get worse).
Idle money R parks in the leading index goes into its synthetic 2x fund (engine exp78) while conditions hold;
never during a pause. Calendar-robust yardstick; R = 71.3%, worst fall -48%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    R = json.load(open("../system_R.json"))["cfg"]
    G = [("2x parking while leader above its 200-day", dict(R, lev_park=200)),
         ("2x parking while leader above its 50-day", dict(R, lev_park=50)),
         ("2x while leader above 50-day and VIX < 20", dict(R, lev_park=50, lev_vix=20)),
         ("2x while leader above 50-day, SPY above 200-day, VIX < 20", dict(R, lev_park=50, lev_vix=20, lev_spy=200))]
    heads = []
    for lab, c in G:
        h, Rr = RB.rscore(pool, c); Rr.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        Rr.to_csv("../audit/a58_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a58_r_lev.csv", index=False)
