"""Where System O's yearly gap vs the better of SPY/QQQ comes from, one run 2000-2025: share of days paused,
average share in stocks / parked index / short Treasuries, and the stock sleeve's own return."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, pickle, numpy as np, pandas as pd, engine as X, yardstick as Y
from exp24_timing import mat
Y._init("../data/panel_ndx.pkl"); m = Y.MM
r = X.run(m, mat(json.load(open("../system_O.json"))["cfg"]), start="2000-01-03", end="2025-12-31", record=True)
pickle.dump(r, open("/tmp/claude-0/-home-user/611e8ebf-cd90-5f23-b4dc-f40b7c4b7cdf/scratchpad/runO.pkl", "wb"))
d = r["daily"]; d.index = pd.to_datetime(d.index)
B = {t: pd.Series(m.E[t]["c"], index=m.dates).ffill() for t in ("SPY", "QQQ", "SMH")}
yr = lambda s: s.groupby(s.index.year).last() / s.groupby(s.index.year).first().shift(0) - 1
rows = []
for y, g in d.groupby(d.index.year):
    nav = g.nav; prev = d.nav[d.index < g.index[0]]
    base = prev.iloc[-1] if len(prev) else nav.iloc[0]
    sy = B["SPY"][str(y)]; qy = B["QQQ"][str(y)]
    sp_prev = B["SPY"][B["SPY"].index < g.index[0]]; qq_prev = B["QQQ"][B["QQQ"].index < g.index[0]]
    s_ret = sy.iloc[-1] / (sp_prev.iloc[-1] if len(sp_prev) else sy.iloc[0]) - 1; q_ret = qy.iloc[-1] / (qq_prev.iloc[-1] if len(qq_prev) else qy.iloc[0]) - 1
    rows.append(dict(year=y, O=nav.iloc[-1] / base - 1, SPY=s_ret, QQQ=q_ret, paused=g.paused.mean(), stocks=(g.stocks / g.nav).mean(),
                     parked=(g.parked / g.nav).mean(), shy=(g.shy / g.nav).mean(), cash=(g.cash / g.nav).mean(), npos=g.npos.mean()))
T = pd.DataFrame(rows).set_index("year"); T["gap"] = T.O - T[["SPY", "QQQ"]].max(axis=1)
T.round(3).to_csv("../audit/a30_o_years.csv"); pd.set_option("display.width", 200); print((T * 100).round(1).to_string())
