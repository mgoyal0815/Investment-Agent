"""Where does the yearly gap to the better of SPY/QQQ come from? (owner, 15 Oct 2026)
  system - best index  =  w x (our stocks - average S&P stock)          stock picking
                        + w x (average S&P stock - best index)          concentration (giants vs the average stock)
                        + (1-w) x (idle money's return - best index)    exposure / cash drag
w = average share of the account in stocks that year; 'our stocks' = the stock book's own flow-free return;
'average S&P stock' = equal-weight point-in-time S&P 500 members."""
import json, sys
import numpy as np
import pandas as pd
import engine as X, metrics as M
from exp24_timing import mat

SYS = sys.argv[1] if len(sys.argv) > 1 else "../system_E.json"
m = X.Market(pd.read_pickle("../data/panel.pkl"))
P, D = m.P, m.dates
r = X.run(m, mat(json.load(open(SYS))["cfg"]), start="2000-01-03", record=True)
d = r["daily"]
# equal-weight S&P 500 (members as of the previous day)
ret = P["ret1"].astype(float)
ew = np.array([np.nanmean(np.where(m.sp_member[i - 1], ret[i], np.nan)) if i > 0 else 0 for i in range(len(D))])
EW = pd.Series(np.cumprod(1 + np.nan_to_num(ew)), index=D)
spy, qqq = (pd.Series(m.E[k]["c"], index=D) for k in ("SPY", "QQQ"))
rows = []
for y in range(2000, 2026):
    s = d[d.index.year == y]
    if len(s) < 200:
        continue
    prev = d[d.index < s.index[0]]
    a = prev.iloc[-1] if len(prev) else s.iloc[0]
    R = s.nav.iloc[-1] / a.nav - 1
    Rs = s.unit.iloc[-1] / a.unit - 1
    w = float((s.stocks / s.nav).mean())
    t0 = prev.index[-1] if len(prev) else s.index[0]
    t1 = s.index[-1]
    g = lambda x: x[t1] / x[t0] - 1
    Rew, Rspy, Rqqq = g(EW), g(spy), g(qqq)
    B = max(Rspy, Rqqq)
    Rrest = (R - w * Rs) / (1 - w) if w < 0.999 else 0.0
    rows.append(dict(year=y, system=R, best_index=B, gap=R - B, in_stocks=w, our_stocks=Rs, avg_stock=Rew,
                     picking=w * (Rs - Rew), concentration=w * (Rew - B), cash_drag=(1 - w) * (Rrest - B), idle_money=Rrest,
                     paused_days=float(s.paused.mean())))
T = pd.DataFrame(rows).set_index("year")
T.to_csv(f"../results/diag_gap_{SYS.split('/')[-1][:-5]}.csv")
pct = T.copy()
for c in T.columns:
    if c not in ("in_stocks", "paused_days"):
        pct[c] = (T[c] * 100).round(1)
pct["in_stocks"] = (T.in_stocks * 100).round(0); pct["paused_days"] = (T.paused_days * 100).round(0)
pd.set_option("display.width", 250)
print(pct.to_string())
L = T[T.gap < 0]
print(f"\n{len(L)} losing years; average gap {L.gap.mean()*100:.1f} pts = picking {L.picking.mean()*100:.1f} + concentration {L.concentration.mean()*100:.1f} + cash drag {L.cash_drag.mean()*100:.1f}")
W = T[T.gap >= 0]
print(f"{len(W)} winning years; average gap {W.gap.mean()*100:.1f} pts = picking {W.picking.mean()*100:.1f} + concentration {W.concentration.mean()*100:.1f} + cash drag {W.cash_drag.mean()*100:.1f}")
