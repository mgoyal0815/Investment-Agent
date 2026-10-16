"""Root cause, part 3: which 12-month windows does the 11th-session calendar win that the 10th loses?
One run 2000-2025 each; rolling 12-month windows starting every month vs the better of SPY/QQQ."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, numpy as np, pandas as pd, engine as X, yardstick as Y
from exp24_timing import mat
Y._init("../data/panel_ndx.pkl"); m = Y.MM
O = json.load(open("../system_O.json"))["cfg"]
B = {t: pd.Series(m.E[t]["c"], index=m.dates).ffill() for t in ("SPY", "QQQ")}
C, R = {}, {}
for d in (9, 10):
    r = X.run(m, mat(dict(O, stagger=[-1, d])), start="2000-01-03", end="2025-12-31", record=True); C[d] = r["curve"]; R[d] = r
idx = C[9].index; ms = idx.to_series().groupby(idx.to_period("M")).min()
rows = []
for s in ms:
    e = s + pd.DateOffset(years=1)
    if e > idx[-1]: break
    e = idx[idx.searchsorted(e)]
    b = max(B["SPY"][e] / B["SPY"][s], B["QQQ"][e] / B["QQQ"][s]) - 1
    rows.append(dict(start=s.date(), s10=C[9][e] / C[9][s] - 1, s11=C[10][e] / C[10][s] - 1, best=b))
W = pd.DataFrame(rows); W["win10"] = W.s10 > W.best; W["win11"] = W.s11 > W.best
W.to_csv("a36_why11_periods.csv", index=False)
print("windows:", len(W), "won by 10:", W.win10.sum(), "won by 11:", W.win11.sum())
F = W[W.win11 != W.win10]; F["who"] = np.where(F.win11, "11 only", "10 only")
pd.set_option("display.width", 200); print((F.assign(s10=F.s10 * 100, s11=F.s11 * 100, best=F.best * 100).round(1)).to_string(index=False))
for d in (9, 10):
    t = R[d]["trades"]; t["yr"] = pd.to_datetime(t.date).dt.year
    print(f"\nsession {d+1}: trades {len(t)}, per year {len(t)/26:.0f}")
