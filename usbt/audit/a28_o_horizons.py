"""(1) How often System K beats both SPY and QQQ over 1, 2, 3 and 5-year windows starting each month (one run from 2000).
(2) First look at a second strategy run beside K: a monthly ETF rotation, blended with K at fixed weights and
rebalanced monthly. Rotation: at each month-end close, rank the candidates by 6-month return; hold the top k equally
from the next session, each only while it is above its 200-day average (otherwise that share sits in SHY).
  R1: top 1 of SPY, QQQ, SMH
  R2: top 2 of QQQ, SMH and the sector ETFs (XLK XLY XLF XLE XLV XLI XLB XLU XLP IGV XLC)
  R3: top 3 of the same
Uses the standard panel; no costs on the monthly ETF switches (a few dollars a year)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import numpy as np, pandas as pd, yardstick as Y
Y._init("../data/panel.pkl"); m = Y.MM; D = m.dates
cur = pd.read_csv("../results/final_O_curves.csv", index_col=0, parse_dates=True)
K = cur["System O"].dropna()
px = {t: pd.Series(m.E[t]["c"], index=D).ffill() for t in ("SPY", "QQQ", "SMH", "SHY")}
sy = list(m.syms)
for t in ("XLK", "XLY", "XLF", "XLE", "XLV", "XLI", "XLB", "XLU", "XLP", "IGV", "XLC"):
    if t in sy:
        px[t] = pd.Series(m.P["Cf"][:, sy.index(t)].astype(float), index=D).ffill()
P = pd.DataFrame(px).loc[K.index[0]:]
ret = P.pct_change().fillna(0.0)
shy = ret["SHY"].where(P["SHY"].notna(), 0.0)

def rotation(cands, k):
    me = P.index.to_series().groupby(P.index.to_period("M")).max().values     # month-end sessions
    w = pd.DataFrame(0.0, index=P.index, columns=list(cands) + ["SHY"])
    m126 = P[list(cands)] / P[list(cands)].shift(126) - 1
    s200 = P[list(cands)] > P[list(cands)].rolling(200).mean()
    cur_w = {"SHY": 1.0}
    idx = list(P.index)
    for d in P.index:
        w.loc[d, list(cur_w)] = list(cur_w.values())
        if d in me:
            r = m126.loc[d].dropna().sort_values(ascending=False)
            top = list(r.index[:k]); cur_w = {}
            for t in top:
                key = t if s200.loc[d, t] else "SHY"
                cur_w[key] = cur_w.get(key, 0) + 1 / k
            if len(top) < k:
                cur_w["SHY"] = cur_w.get("SHY", 0) + (k - len(top)) / k
    w = w.shift(1).fillna(0.0); w.iloc[0, -1] = 1.0                         # decided at the close, held from the next session
    r_ = (w[list(cands)] * ret[list(cands)].fillna(0)).sum(1) + w["SHY"] * shy
    return (1 + r_).cumprod() * 10000
sect = ["QQQ", "SMH"] + [t for t in ("XLK", "XLY", "XLF", "XLE", "XLV", "XLI", "XLB", "XLU", "XLP", "IGV", "XLC") if t in P]
R_unused = {"R1 top 1 of SPY/QQQ/SMH": rotation(["SPY", "QQQ", "SMH"], 1), "R2 top 2 sectors+QQQ+SMH": rotation(sect, 2), "R3 top 3 sectors+QQQ+SMH": rotation(sect, 3)}

def blend(a, b, w):
    ra, rb = a.pct_change().fillna(0), b.pct_change().fillna(0)
    out, val, wa = [], 10000.0, w
    va, vb = val * w, val * (1 - w)
    mon = a.index.to_period("M")
    for k_, d in enumerate(a.index):
        va *= 1 + ra.iloc[k_]; vb *= 1 + rb.iloc[k_]
        if k_ + 1 < len(a.index) and mon[k_ + 1] != mon[k_]:
            t = va + vb; va, vb = t * w, t * (1 - w)
        out.append(va + vb)
    return pd.Series(out, index=a.index)

spy, qqq = P["SPY"], P["QQQ"]
def scores(c, name):
    c = c.reindex(P.index).ffill()
    yr = c.groupby(c.index.year).last(); yr = yr.pct_change().fillna(yr.iloc[0] / c.iloc[0] - 1)
    by = lambda s: s.groupby(s.index.year).last().pct_change().fillna(s.groupby(s.index.year).last().iloc[0] / s.iloc[0] - 1)
    best = pd.concat([by(spy), by(qqq)], axis=1).max(axis=1)
    y = (yr - best).loc[2000:2025]
    starts = P.index.to_series().groupby(P.index.to_period("M")).min().values
    out = dict(system=name, calendar_years_beating=f"{int((y > 0).sum())}/26", calendar_share=float((y > 0).mean()))
    for H in (1, 2, 3, 5):
        wins = []
        for s in starts:
            e = s + pd.DateOffset(years=H)
            if e > P.index[-1]: break
            e = P.index[P.index.searchsorted(e) - 1] if e not in P.index else e
            wins.append(c[e] / c[s] - 1 > max(spy[e] / spy[s] - 1, qqq[e] / qqq[s] - 1))
        out[f"{H}y windows"] = float(np.mean(wins))
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    out["growth/yr"] = float((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1); out["worst fall"] = float((c / c.cummax() - 1).min())
    return out

rows = [scores(K, "System O"), scores(spy / spy.iloc[0] * 1e4, "SPY"), scores(qqq / qqq.iloc[0] * 1e4, "QQQ")]
T = pd.DataFrame(rows); T.to_csv("../audit/a28_o_horizons.csv", index=False)
pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))
