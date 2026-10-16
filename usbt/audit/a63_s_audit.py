"""Audit of candidate System S = R + 2x parking (leader above its 200-day, none in a new account's first 126 sessions)
(owner, 16 Oct 2026). Calendar-robust yardstick: S with the real SSO / QLD / USD prices from their launch (synthetic
before), on the standard day-1 starts and on the held-back 15th-of-month starts; R on the held-back starts for
comparison. R day 1 = 71.3%, worst -48.1%; S synthetic day 1 = 74.8%, worst -49.1%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    R = json.load(open("../system_R.json"))["cfg"]; S = dict(R, lev_park=200, lev_after=126)
    G = [("S, real funds, day-1 starts", dict(S, lev_real=True), 1), ("S, real funds, held-back 15th starts", dict(S, lev_real=True), 15),
         ("S, synthetic, held-back 15th starts", S, 15), ("R, held-back 15th starts", R, 15)]
    heads = []
    for lab, c, day in G:
        h, Rr = RB.rscore(pool, c, day=day); Rr.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        Rr.to_csv("../audit/a63_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a63_s_audit.csv", index=False)
