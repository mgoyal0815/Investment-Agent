"""Owner's definition (16 Oct 2026): beat SPY and QQQ in each calendar year - from the start date to that December,
then every full calendar year. System R, calendar-robust (5 review calendars), day-1 and held-back 15th starts,
every monthly start 2000-2024, runs to Dec 2025. Also reports the first (partial) year on its own."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, numpy as np, pandas as pd, engine as X, yardstick as Y, robust as RB
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"


def first_year(a):
    cfg, s = a
    m = Y.MM; c = X.run(m, mat(cfg), start=str(s.date()), end=Y.STD_END)["curve"]
    e = c.index[c.index.year == s.year][-1]
    B = {t: pd.Series(m.E[t]["c"], index=m.dates).ffill() for t in ("SPY", "QQQ")}
    r = c[e] / c.iloc[0] - 1
    return bool(r > B["SPY"][e] / B["SPY"][s] - 1 and r > B["QQQ"][e] / B["QQQ"][s] - 1)


if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    R = json.load(open("../system_R.json"))["cfg"]
    heads = []
    for day in (1, 15):
        h, Rr = RB.rscore(pool, R, day=day); Rr.insert(0, "starts", f"day {day}"); heads.append(dict(starts=f"day {day}", **h))
        Rr.to_csv("../audit/a67_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
    fy = []
    for c in RB.calendars(R):
        fy += pool.map(first_year, [(c, s) for s in Y.starts(step=3)])
    print({"first partial year won (start to December), all 5 calendars": round(float(np.mean(fy)), 3)}, flush=True)
    pd.DataFrame(heads).to_csv("../audit/a67_r_calendar.csv", index=False)
