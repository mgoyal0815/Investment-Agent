"""Why does System A lose the years it loses? For every year: account return, the stock book's own
return (flow-free), the money split (stocks / leading-index ETF / Treasuries / cash), days paused,
number of holdings, and the index returns. Then for the stock book in each losing year, the best
and worst names. Writes results/root_years_A.csv."""
import json
import numpy as np
import pandas as pd
import engine as X
from exp2_search import materialise

A = materialise(json.load(open("../system_A.json"))["cfg"])
m = X.Market(pd.read_pickle("../data/panel.pkl"))
parts, lots = [], []
for s, e in (("2000-01-03", "2011-12-30"), ("2012-01-03", None)):
    r = X.run(m, A, start=s, end=e, record=True)
    parts.append(r["daily"]); lots.append(r["lots"])
D = pd.concat(parts); L = pd.concat(lots)
idx = {k: pd.Series(m.E[k]["c"], index=m.dates) for k in ("SPY", "QQQ", "SMH")}
rows = []
for y, g in D.groupby(D.index.year):
    prev = D[D.index.year == y - 1]
    nav0 = prev.nav.iloc[-1] if len(prev) else g.nav.iloc[0]
    u0 = prev.unit.iloc[-1] if len(prev) else g.unit.iloc[0]
    row = dict(year=y, account=g.nav.iloc[-1] / nav0 - 1, stock_book=g.unit.iloc[-1] / u0 - 1,
               in_stocks=(g.stocks / g.nav).mean(), in_index_etf=((g.parked - g.shy) / g.nav).mean(),
               in_treasuries=(g.shy / g.nav).mean(), cash=(g.cash / g.nav).mean(), paused=g.paused.mean(),
               holdings=g.npos.mean())
    for k, v in idx.items():
        vv = v[v.index.year <= y]
        row[k] = vv[vv.index.year == y].iloc[-1] / vv[vv.index.year == y - 1].iloc[-1] - 1 if (vv.index.year == y - 1).any() else np.nan
    row["best_index"] = max(row["SPY"], row["QQQ"], row["SMH"])
    row["gap"] = row["account"] - row["best_index"]
    rows.append(row)
T = pd.DataFrame(rows).set_index("year")
T.to_csv("../results/root_years_A.csv")
pd.set_option("display.width", 220)
print((T * 100).round(1).drop(columns=["holdings"]).assign(holdings=T.holdings.round(0)).to_string())
L["exit_year"] = pd.to_datetime(L.exit).dt.year
lose = T[T.gap < 0].index
print("\nLOSING YEARS: closed-position P&L by exit reason")
print(L[L.exit_year.isin(lose)].pivot_table(index="exit_year", columns="why", values="pnl", aggfunc="sum").round(0).to_string())
