"""What do indicators on System J's own account say when it is low? (owner, 15 Oct 2026)
One run of J from 2000 on the standard panel. Indicators on J's daily account value (and on its stock book's
flow-free unit value): 14-day RSI, MACD histogram, distance from its 50/200-day averages, drawdown from peak,
and J's 63-day return minus the better of SPY/QQQ. For each indicator bucket: what J did over the next 21 and
63 sessions, and by how much it beat or lagged the better of SPY and QQQ over the same window."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, numpy as np, pandas as pd, engine as X, yardstick as Y
from exp24_timing import mat
Y._init("../data/panel.pkl"); m = Y.MM
J = json.load(open("../system_J.json"))["cfg"]
r = X.run(m, mat(J), start="2000-01-03", record=True)
d = r["daily"]; nav = d.nav; D = nav.index
spy = pd.Series(m.E["SPY"]["c"], index=m.dates).reindex(D); qqq = pd.Series(m.E["QQQ"]["c"], index=m.dates).reindex(D)
def rsi(s, n=14):
    x = s.diff(); up = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)
def macdh(s):
    mc = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean(); return (mc - mc.ewm(span=9, adjust=False).mean()) / s
F = pd.DataFrame(index=D)
F["RSI14"] = rsi(nav)
F["MACD hist (% of value)"] = macdh(nav) * 100
F["vs 50-day avg (%)"] = (nav / nav.rolling(50).mean() - 1) * 100
F["vs 200-day avg (%)"] = (nav / nav.rolling(200).mean() - 1) * 100
F["below peak (%)"] = (nav / nav.cummax() - 1) * 100
best63 = np.maximum(spy / spy.shift(63) - 1, qqq / qqq.shift(63) - 1)
F["last 63d vs best index (pts)"] = ((nav / nav.shift(63) - 1) - best63) * 100
for n in (21, 63):
    F[f"J next {n}d (%)"] = (nav.shift(-n) / nav - 1) * 100
    F[f"J minus best index next {n}d (pts)"] = ((nav.shift(-n) / nav - 1) - np.maximum(spy.shift(-n) / spy - 1, qqq.shift(-n) / qqq - 1)) * 100
F["paused"] = d.paused.astype(bool)
F = F.iloc[200:]
out = []
for ind in ["RSI14", "MACD hist (% of value)", "vs 50-day avg (%)", "vs 200-day avg (%)", "below peak (%)", "last 63d vs best index (pts)"]:
    q = pd.qcut(F[ind], 5, labels=["lowest 20%", "low", "middle", "high", "highest 20%"], duplicates="drop")
    g = F.groupby(q, observed=True)
    t = pd.DataFrame({"range": g[ind].agg(lambda x: f"{x.min():.1f} to {x.max():.1f}"), "days": g.size(),
                      "J next 21d": g["J next 21d (%)"].mean(), "J next 63d": g["J next 63d (%)"].mean(),
                      "vs best index 21d": g["J minus best index next 21d (pts)"].mean(), "vs best index 63d": g["J minus best index next 63d (pts)"].mean(),
                      "share beating index 63d": g["J minus best index next 63d (pts)"].apply(lambda x: (x > 0).mean()), "paused share": g["paused"].mean()})
    t.insert(0, "indicator", ind); out.append(t.reset_index(names="bucket"))
R = pd.concat(out); R.to_csv("../audit/a9_book_signals.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 100)
print(R.round(2).to_string(index=False))
print("\nAll days: J next 63d %.2f%%, vs best index %.2f pts, beats index %.0f%% of the time" % (F["J next 63d (%)"].mean(), F["J minus best index next 63d (pts)"].mean(), (F["J minus best index next 63d (pts)"] > 0).mean() * 100))
