"""Which calendar years does System Q lose on every review calendar? One run 2000-2025 per calendar (second review on
sessions 5/8/11/14/17), yearly return vs the better of SPY/QQQ, and P's exposure in each year (11th-session calendar)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, numpy as np, pandas as pd, engine as X, yardstick as Y, robust as RB
from exp24_timing import mat
Y._init("../data/panel_ndx.pkl"); m = Y.MM
P = json.load(open("../system_Q.json"))["cfg"]
B = {t: pd.Series(m.E[t]["c"], index=m.dates).ffill()["1999-12-31":"2025-12-31"] for t in ("SPY", "QQQ", "SMH")}
yr = lambda s: s.groupby(s.index.year).last().pct_change().dropna()
best = pd.concat([yr(B["SPY"]), yr(B["QQQ"])], axis=1).max(axis=1)
out = {}
for c in RB.calendars(P):
    r = X.run(m, mat(c), start="2000-01-03", end="2025-12-31", record=(c["stagger"] == [-1, 10]))
    cv = r["curve"]; y = cv.groupby(cv.index.year).last() / cv.groupby(cv.index.year).last().shift(1).fillna(cv.iloc[0]) - 1
    out[f"s{c['stagger'][1]+1}"] = y.reindex(best.index)
    if "daily" in r: d = r["daily"]; d.index = pd.to_datetime(d.index)
T = pd.DataFrame(out); T["best index"] = best; T["SMH"] = yr(B["SMH"])
cols = [k for k in out]; T["calendars won"] = (T[cols].gt(T["best index"], axis=0)).sum(1)
T["avg gap"] = T[cols].mean(1) - T["best index"]
g = d.groupby(d.index.year)
T["paused % (s11)"] = g.paused.mean(); T["in stocks % (s11)"] = g.apply(lambda x: (x.stocks / x.nav).mean())
T.to_csv("../audit/a43_q_years.csv")
pd.set_option("display.width", 250)
show = T.copy(); show[cols + ["best index", "SMH", "avg gap", "paused % (s11)", "in stocks % (s11)"]] *= 100
print(show.round(1).to_string())
print("\nyears won on all 5 calendars:", int((T["calendars won"] == 5).sum()), " on none:", int((T["calendars won"] == 0).sum()),
      " average share:", round(T["calendars won"].mean() / 5, 3))
