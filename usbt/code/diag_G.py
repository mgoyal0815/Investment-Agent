"""System G post-mortem (owner, 15 Oct 2026): what its own trades and daily log say about the losing years.
1 exits: what sold stocks did next, by reason   2 losing years: where the gap came from, and which index winners
it missed and why   3 pauses: what the market did while it sat out   4 lot outcomes: holding time and peak give-back."""
import json, pickle, sys
import numpy as np, pandas as pd
import yardstick as Y
Y._init("../data/panel.pkl"); m = Y.MM; P, D = m.P, m.dates
R = pickle.load(open(sys.argv[1], "rb"))
T, L, d = R["trades"], R["lots"], R["daily"]
Cf, J, di = P["Cf"], {s: j for j, s in enumerate(m.syms)}, {x: k for k, x in enumerate(D)}
spy, qqq = np.asarray(m.E["SPY"]["c"], float), np.asarray(m.E["QQQ"]["c"], float)
def fwd(a, i, n): k = min(i + n, len(D) - 1); return a[k] / a[i] - 1
pd.set_option("display.width", 250, "display.max_rows", 300)

print("== 1. SELLS: what the stock did next (vs SPY), by reason")
rows = []
for t in L[L.exit.notna()].itertuples():
    i = di.get(pd.Timestamp(t.exit)); j = t.j
    if i is None or i + 5 >= len(D): continue
    rows.append(dict(why=t.why, year=pd.Timestamp(t.exit).year, exit_ret=t.exit_ret, peak=t.peak_ret, held=t.held,
                     n63=fwd(Cf[:, j], i, 63) - fwd(spy, i, 63), n126=fwd(Cf[:, j], i, 126) - fwd(spy, i, 126), n252=fwd(Cf[:, j], i, 252) - fwd(spy, i, 252)))
S = pd.DataFrame(rows)
print(S.groupby("why").agg(n=("n63", "size"), exit_ret=("exit_ret", "median"), peak=("peak", "median"), held_days=("held", "median"),
                           next63_vs_spy=("n63", "mean"), next126_vs_spy=("n126", "mean"), next252_vs_spy=("n252", "mean"),
                           share_beat_spy_126=("n126", lambda x: (x > 0).mean())).round(3).to_string())
print("rank sells of stocks that were winners at exit (exit_ret>30%):")
w = S[(S.why == "rank") & (S.exit_ret > 0.3)]
print(f"  n={len(w)}, next126 vs SPY {w.n126.mean():.3f}, beat SPY {(w.n126>0).mean():.2f}")

print("\n== 4. LOTS: give-back from peak and holding time")
L2 = L[L.exit.notna()].copy(); L2["giveback"] = L2.peak_ret - L2.exit_ret
for lo, hi in ((-9, 0), (0, .2), (.2, .5), (.5, 1), (1, 99)):
    g = L2[(L2.peak_ret > lo) & (L2.peak_ret <= hi)]
    print(f"  peak {lo:>4}-{hi:<3}: n={len(g):4d}  median exit {g.exit_ret.median():6.2f}  median give-back {g.giveback.median():.2f}  median held {g.held.median():.0f}d")
L2.to_csv("../results/diagG_lots.csv", index=False)

print("\n== 3. PAUSES: market return while paused vs not, by year (SPY, sum of daily returns)")
sr = pd.Series(spy, index=D).pct_change().reindex(d.index)
py = pd.DataFrame({"paused_share": d.paused.groupby(d.index.year).mean(), "spy_while_paused": sr[d.paused.astype(bool)].groupby(sr[d.paused.astype(bool)].index.year).sum(),
                   "spy_while_on": sr[~d.paused.astype(bool)].groupby(sr[~d.paused.astype(bool)].index.year).sum(),
                   "in_stocks": (d.stocks / d.nav).groupby(d.index.year).mean()}).round(3)
print(py.to_string())

print("\n== 2. LOSING YEARS: which of the year's 15 best large S&P stocks (by index weight proxy) did G hold, and why not")
cur = d.nav
yr = cur.groupby(cur.index.year).last().pct_change()
yr.iloc[0] = cur.groupby(cur.index.year).last().iloc[0] / cur.iloc[0] - 1
by = lambda a: pd.Series(a, index=D).groupby(D.year).last()
bs, bq = by(spy).pct_change(), by(qqq).pct_change()
held = {}
for t in L.itertuples():
    e1 = pd.Timestamp(t.exit) if pd.notna(t.exit) else D[-1]
    held.setdefault(t.sym, []).append((pd.Timestamp(t.entry), e1))
medv = P["medv"]
out = []
for y in range(2000, 2026):
    if not (yr[y] < max(bs[y], bq[y])): continue
    i0 = np.searchsorted(D, pd.Timestamp(f"{y}-01-01")) - 1; i1 = np.searchsorted(D, pd.Timestamp(f"{y}-12-31"), side="right") - 1
    mem = m.sp_member[i0] & ~m.is_etf
    big = np.where(mem, medv[i0], np.nan); top = np.argsort(-np.nan_to_num(big, nan=-1))[:100]   # 100 most traded members = the giants
    rr = Cf[i1, top] / Cf[i0, top] - 1
    best = top[np.argsort(-np.nan_to_num(rr, nan=-9))[:15]]
    for j in best:
        s = m.syms[j]; days = sum(max(0, (min(b, D[i1]) - max(a, D[i0])).days) for a, b in held.get(s, []))
        # why not: share of month-end decision days the stock passed each gate
        idx = [k for k in range(i0, i1) if D[k].month != D[k + 1].month]
        adx = np.mean([P["ADX"][k, j] >= 25 and P["PDI_GT"][k, j] for k in idx]); obv = np.mean([P["OBVT"][k, j] >= 0 for k in idx])
        trend = np.mean([bool(P["cont"][k, j]) for k in idx]); pz = np.mean([bool(d.paused.asof(D[k])) for k in idx])
        rk = np.nanmean([pd.Series(np.where(m.sp_member[k] & ~m.is_etf, P["rs126"][k], np.nan)).rank(pct=True).iloc[j] for k in idx])
        out.append(dict(year=y, sym=s, stock_ret=round(Cf[i1, j] / Cf[i0, j] - 1, 2), days_held=days, rs_pctile=round(rk, 2),
                        adx_ok=round(adx, 2), obv_ok=round(obv, 2), trend_ok=round(trend, 2), mkt_paused=round(pz, 2)))
O = pd.DataFrame(out); O.to_csv("../results/diagG_missed.csv", index=False)
for y, g in O.groupby("year"):
    print(f"\n{y}: G {yr[y]*100:.1f}%  SPY {bs[y]*100:.1f}%  QQQ {bq[y]*100:.1f}%  held {int((g.days_held>0).sum())} of 15 best giants")
    print(g.drop(columns="year").to_string(index=False))
print("\nAcross losing years, the 15 best giants:")
print(f"  held at all {(O.days_held>0).mean():.2f}; avg 6m-strength percentile {O.rs_pctile.mean():.2f}; passed ADX gate {O.adx_ok.mean():.2f}; OBV gate {O.obv_ok.mean():.2f}; trend gate {O.trend_ok.mean():.2f}; market paused {O.mkt_paused.mean():.2f}")
