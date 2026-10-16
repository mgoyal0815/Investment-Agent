"""Start-year table (owner, 16 Oct 2026): System R started with $10,000 on the first session of each January 2000-2024
(single account, its own calendar), every following calendar year 2000-2025 compared with SPY and QQQ. 1 = R beat both."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, numpy as np, pandas as pd, engine as X, yardstick as Y
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"


def job(y):
    m = Y.MM; R = json.load(open("../system_R.json"))["cfg"]
    s = m.dates[m.dates.searchsorted(pd.Timestamp(f"{y}-01-01"))]
    c = X.run(m, mat(R), start=str(s.date()), end=Y.STD_END)["curve"]
    yr = c.groupby(c.index.year).last(); base = pd.concat([pd.Series({y - 1: c.iloc[0]}), yr]); ret = (base / base.shift(1) - 1).dropna()
    return y, ret


if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    m = Y.MM
    B = {t: pd.Series(m.E[t]["c"], index=m.dates).ffill()[:"2025-12-31"] for t in ("SPY", "QQQ")}
    by = lambda s: s.groupby(s.index.year).last().pct_change()
    best = pd.concat([by(B["SPY"]), by(B["QQQ"])], axis=1).max(axis=1)
    best[2000] = max(B["SPY"]["2000"].iloc[-1] / B["SPY"]["2000"].iloc[0], B["QQQ"]["2000"].iloc[-1] / B["QQQ"]["2000"].iloc[0]) - 1
    rows = {}
    for y, ret in pool.map(job, range(2000, 2025)):
        rows[y] = {yy: int(ret[yy] > best[yy]) for yy in ret.index if yy <= 2025}
    T = pd.DataFrame(rows).T.sort_index(); T = T[sorted(T.columns)]
    T["won"] = T.sum(1); T["years"] = T.drop(columns="won").notna().sum(1); T["share"] = (T.won / T.years).round(2)
    T.to_csv("../audit/a66_start_year.csv")
    pd.set_option("display.width", 250)
    show = T.drop(columns=["won", "years", "share"]).replace({1: "W", 0: "."}).fillna("")
    show.columns = [str(c)[2:] for c in show.columns]
    print(pd.concat([show, T[["won", "years", "share"]]], axis=1).to_string())
    print("\nShare of years won, all starts: %.3f" % (T.won.sum() / T.years.sum()))
