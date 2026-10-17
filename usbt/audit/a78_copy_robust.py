"""Owner, 17 Oct 2026: before calling the 75% goal met for a copy-on-day-one investor, score the RUNNING account the
fair way. A copy tracks the running account (a76/a77: 90-98% invested on day one, years within a few points), so the
running account's record is the copier's record. Running R1 accounts from 2000-01-03 to 2025-12-31 on each of the 5
review calendars (robust.calendars) x (R1's tie-break + 8 random tie orders as the luck range). Measures (judge):
calendar years beating SPY and QQQ, rolling 3-year windows beating SMH, worst fall vs SPY and QQQ."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, numpy as np, pandas as pd, yardstick as Y, robust as RB, engine as X
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"
R1 = json.load(open("../system_R1.json"))["cfg"]


def one(a):
    cal, sd = a
    c = dict(cal) if sd is None else dict(cal, tie_key=None, tie_seed=sd)
    cv = X.run(Y.MM, mat(c), start="2000-01-03", end="2025-12-31")["curve"]
    j = Y.judge(cv, Y.MM)
    y = cv.groupby(cv.index.year).last(); y0 = pd.concat([pd.Series([cv.iloc[0]], index=[1999]), y]).pct_change().dropna()
    return dict(stagger=str(cal["stagger"]), tie="R1 (6-month strength)" if sd is None else sd, cal_win=j["cal_win"], r3_win=j["r3_win"],
                dd=j["dd"], dd_ok=j["dd_ok"], cagr=j["cagr"], end=float(cv.iloc[-1]), cal_2016_25=float(np.mean([
                    y0[k] > Y.MM.E["SPY"]["c"][Y.MM.dates.get_loc(cv.index[cv.index.year == k][-1])] / Y.MM.E["SPY"]["c"][Y.MM.dates.get_loc(cv.index[cv.index.year == k - 1][-1])] - 1 and
                    y0[k] > Y.MM.E["QQQ"]["c"][Y.MM.dates.get_loc(cv.index[cv.index.year == k][-1])] / Y.MM.E["QQQ"]["c"][Y.MM.dates.get_loc(cv.index[cv.index.year == k - 1][-1])] - 1
                    for k in range(2016, 2026)])))


if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    jobs = [(c, sd) for c in RB.calendars(R1) for sd in (None,) + RB.TIE_SEEDS]
    T = pd.DataFrame(pool.map(one, jobs, chunksize=1))
    T.to_csv("a78_copy_robust.csv", index=False)
    a, s = T[T.tie == "R1 (6-month strength)"], T[T.tie != "R1 (6-month strength)"]
    S = {"R1 tie-break, 5 calendars": {k: dict(mean=float(a[k].mean()), lo=float(a[k].min()), hi=float(a[k].max())) for k in ("cal_win", "cal_2016_25", "r3_win", "dd", "cagr")},
         "random tie orders, 40 runs": {k: dict(median=float(s[k].median()), lo=float(s[k].min()), hi=float(s[k].max())) for k in ("cal_win", "cal_2016_25", "r3_win", "dd", "cagr")},
         "dd_ok share": float(T.dd_ok.mean())}
    json.dump(dict(summary=S, provenance=RB.provenance(panel=PATH, system_file="system_R1.json")), open("a78_copy_robust.json", "w"), indent=1)
    print(json.dumps(S, indent=1)); print(a[["stagger", "cal_win", "cal_2016_25", "r3_win", "dd", "cagr", "end"]].round(3).to_string(index=False))
