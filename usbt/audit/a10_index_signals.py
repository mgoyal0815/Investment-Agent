"""What were the index indicators saying, and did they tell when the market would rebound? (owner, 15 Oct 2026)
Indicators on SPY and QQQ (RSI14, MACD histogram, distance from 50/200-day averages, fall from 1-year high),
VIX, and S&P 500 breadth (share of that day's members above their 200-day average). For each indicator bucket:
the better index's next 21/63 sessions, and System J's next 63 sessions vs that index; all days, and only the
days when J itself was low (more than 2% below its own 50-day average)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, numpy as np, pandas as pd, engine as X, yardstick as Y
from exp24_timing import mat
Y._init("../data/panel.pkl"); m = Y.MM
J = json.load(open("../system_J.json"))["cfg"]
d = X.run(m, mat(J), start="2000-01-03", record=True)["daily"]; nav = d.nav; D = nav.index
px = {t: pd.Series(m.E[t]["c"], index=m.dates).reindex(D) for t in ("SPY", "QQQ")}
def rsi(s, n=14):
    x = s.diff(); up = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)
def macdh(s):
    mc = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean(); return (mc - mc.ewm(span=9, adjust=False).mean()) / s * 100
F = pd.DataFrame(index=D)
for t, s in px.items():
    F[f"{t} RSI14"] = rsi(s); F[f"{t} MACD hist (%)"] = macdh(s)
    F[f"{t} vs 50-day (%)"] = (s / s.rolling(50).mean() - 1) * 100; F[f"{t} vs 200-day (%)"] = (s / s.rolling(200).mean() - 1) * 100
    F[f"{t} below 1-year high (%)"] = (s / s.rolling(252, min_periods=60).max() - 1) * 100
F["VIX"] = pd.Series(m.P["VIX"], index=m.dates).reindex(D)
F["breadth: % of S&P above 200-day"] = pd.Series(m.breadth_pit, index=m.dates).reindex(D) * 100
fwd = lambda s, n: s.shift(-n) / s - 1
for n in (21, 63):
    F[f"best index next {n}d (%)"] = np.maximum(fwd(px["SPY"], n), fwd(px["QQQ"], n)) * 100
F["J vs best index next 63d (pts)"] = (fwd(nav, 63) - np.maximum(fwd(px["SPY"], 63), fwd(px["QQQ"], 63))) * 100
F["J low"] = nav / nav.rolling(50).mean() - 1 < -0.02
F = F.iloc[252:].dropna(subset=["best index next 63d (%)"])
inds = [c for c in F.columns if not c.startswith(("best index", "J "))]
rows = []
for ind in inds:
    for scope, G in (("all days", F), ("J low", F[F["J low"]])):
        q = pd.qcut(F[ind], 5, labels=["lowest 20%", "low", "middle", "high", "highest 20%"], duplicates="drop").reindex(G.index)
        g = G.groupby(q, observed=True)
        t = pd.DataFrame({"range": F.groupby(pd.qcut(F[ind], 5, labels=["lowest 20%", "low", "middle", "high", "highest 20%"], duplicates="drop"), observed=True)[ind].agg(lambda x: f"{x.min():.1f} to {x.max():.1f}"),
                          "days": g.size(), "index next 21d": g["best index next 21d (%)"].mean(), "index next 63d": g["best index next 63d (%)"].mean(),
                          "index up next 63d": g["best index next 63d (%)"].apply(lambda x: (x > 0).mean()),
                          "J vs index 63d": g["J vs best index next 63d (pts)"].mean(), "J beats index 63d": g["J vs best index next 63d (pts)"].apply(lambda x: (x > 0).mean())})
        t.insert(0, "scope", scope); t.insert(0, "indicator", ind); rows.append(t.reset_index(names="bucket"))
R = pd.concat(rows); R.to_csv("../audit/a10_index_signals.csv", index=False)
pd.set_option("display.width", 260, "display.max_rows", 300)
print(R[R.scope == "all days"].round(2).to_string(index=False))
print(R[R.scope == "J low"].round(2).to_string(index=False))
print("\nAll days: index next 63d %.2f%%, up %.0f%% of the time; J low on %.0f%% of days" % (F["best index next 63d (%)"].mean(), (F["best index next 63d (%)"] > 0).mean() * 100, F["J low"].mean() * 100))
