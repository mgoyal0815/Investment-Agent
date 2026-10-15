"""Deep trade review of System K (owner, 15 Oct 2026). One run from 3 Jan 2000 on the standard panel.
Writes audit/k_trades_*.csv and audit/k_trades_summary.json, prints the headline tables."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, pickle, numpy as np, pandas as pd, engine as X, yardstick as Y
from exp24_timing import mat
Y._init("../data/panel.pkl"); m = Y.MM
K = json.load(open("../system_K.json"))["cfg"]
r = X.run(m, mat(K), start="2000-01-03", record=True)
pickle.dump({k: r[k] for k in ("trades", "lots", "daily", "curve")}, open("/tmp/claude-0/-home-user/611e8ebf-cd90-5f23-b4dc-f40b7c4b7cdf/scratchpad/runK.pkl", "wb"))
T, L, d = r["trades"], r["lots"].copy(), r["daily"]
D = m.dates; di = {x: k for k, x in enumerate(D)}
spy = pd.Series(m.E["SPY"]["c"], index=D); qqq = pd.Series(m.E["QQQ"]["c"], index=D)
out = {}
pd.set_option("display.width", 250, "display.max_rows", 200)

# ---- 1. every closed position (lot)
L["entry"] = pd.to_datetime(L.entry); L["exit"] = pd.to_datetime(L.exit)
C = L[L.exit.notna()].copy()
C["ret"] = C.exit_ret; C["year"] = C.exit.dt.year; C["win"] = C.pnl > 0
spy_ret = lambda a, b: spy.asof(b) / spy.asof(a) - 1
C["spy_same_period"] = [spy_ret(a, b) for a, b in zip(C.entry, C.exit)]
C["beat_spy"] = C.ret > C.spy_same_period
w, l_ = C[C.win], C[~C.win]
summ = dict(positions=len(C), open_now=int(L.exit.isna().sum()), win_rate=float(C.win.mean()),
            avg_win=float(w.ret.mean()), avg_loss=float(l_.ret.mean()), median_win=float(w.ret.median()), median_loss=float(l_.ret.median()),
            profit_factor=float(w.pnl.sum() / -l_.pnl.sum()), avg_days_held=float(C.held.mean()), median_days_held=float(C.held.median()),
            beat_spy_same_period=float(C.beat_spy.mean()), total_costs=float(r["costs"]),
            buys=int((T.side == "BUY").sum()), sells=int((T.side == "SELL").sum()),
            top10_share_of_profit=float(C.nlargest(10, "pnl").pnl.sum() / C.pnl.sum()),
            top1pct_share_of_profit=float(C.nlargest(max(1, len(C) // 100), "pnl").pnl.sum() / C.pnl.sum()))
out["summary"] = summ
print("== 1. CLOSED POSITIONS"); print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summ.items()}, indent=1))

by_why = C.groupby("why").agg(n=("ret", "size"), win_rate=("win", "mean"), avg_ret=("ret", "mean"), median_ret=("ret", "median"),
                              avg_peak=("peak_ret", "mean"), avg_days=("held", "mean"), pnl=("pnl", "sum")).round(3)
print("\nBy exit reason:"); print(by_why.to_string()); by_why.to_csv("../audit/k_trades_by_exit.csv")

hb = pd.cut(C.held, [0, 21, 63, 126, 252, 10000], labels=["<1 month", "1-3 months", "3-6 months", "6-12 months", ">1 year"])
by_hold = C.groupby(hb, observed=True).agg(n=("ret", "size"), win_rate=("win", "mean"), avg_ret=("ret", "mean"), pnl=("pnl", "sum")).round(3)
print("\nBy holding time:"); print(by_hold.to_string()); by_hold.to_csv("../audit/k_trades_by_holding.csv")

by_year = C.groupby("year").agg(closed=("ret", "size"), win_rate=("win", "mean"), avg_ret=("ret", "mean"), pnl=("pnl", "sum"),
                                beat_spy=("beat_spy", "mean")).round(3)
by_year.to_csv("../audit/k_trades_by_year.csv")

best = C.nlargest(20, "pnl")[["sym", "entry", "exit", "held", "ret", "peak_ret", "pnl", "why"]]
worst = C.nsmallest(20, "pnl")[["sym", "entry", "exit", "held", "ret", "peak_ret", "pnl", "why"]]
best.to_csv("../audit/k_trades_best.csv", index=False); worst.to_csv("../audit/k_trades_worst.csv", index=False)
print("\nBest 20:"); print(best.round(3).to_string(index=False)); print("\nWorst 20:"); print(worst.round(3).to_string(index=False))
syms = C.groupby("sym").agg(trades=("ret", "size"), pnl=("pnl", "sum"), win_rate=("win", "mean")).sort_values("pnl")
syms.to_csv("../audit/k_trades_by_symbol.csv")
print("\nMost-traded names:"); print(syms.sort_values("trades", ascending=False).head(12).round(3).to_string())
print("\nBiggest losing names:"); print(syms.head(10).round(0).to_string())

# give-back from peak
C["giveback"] = C.peak_ret - C.ret
gb = C.groupby(pd.cut(C.peak_ret, [-9, 0, 0.2, 0.5, 1, 99], labels=["never up", "peak 0-20%", "peak 20-50%", "peak 50-100%", "peak >100%"]), observed=True)
gbt = gb.agg(n=("ret", "size"), median_peak=("peak_ret", "median"), median_exit=("ret", "median"), median_giveback=("giveback", "median")).round(3)
print("\nGive-back from peak:"); print(gbt.to_string()); gbt.to_csv("../audit/k_trades_giveback.csv")

# ---- 2. where the money sat each year
d = d.copy(); d["index_parked"] = d.parked - d.shy
yr = d.groupby(d.index.year).apply(lambda x: pd.Series(dict(
    stocks=(x.stocks / x.nav).mean(), index_etf=(x.index_parked / x.nav).mean(), treasuries=(x.shy / x.nav).mean(),
    cash=(x.cash / x.nav).mean(), paused_days=x.paused.mean(), avg_positions=x.npos.mean()))).round(3)
cal = r["curve"].groupby(r["curve"].index.year).last().pct_change()
cal.iloc[0] = r["curve"].groupby(r["curve"].index.year).last().iloc[0] / r["curve"].iloc[0] - 1
by = lambda s: s.groupby(s.index.year).last().pct_change()
yr["K"] = cal.round(3); yr["SPY"] = by(spy).round(3); yr["QQQ"] = by(qqq).round(3)
yr = yr.join(by_year[["closed", "win_rate", "pnl"]])
print("\n== 2. WHERE THE MONEY SAT, BY YEAR"); print(yr.to_string()); yr.to_csv("../audit/k_trades_exposure_by_year.csv")

# ---- 3. pause episodes and crash-trigger index holds
p = d.paused.astype(bool); edges = p.ne(p.shift()).cumsum()
eps = []
for _, g in d[p].groupby(edges[p]):
    a, b = g.index[0], g.index[-1]
    eps.append(dict(start=a.date(), end=b.date(), sessions=len(g), spy=spy.asof(b) / spy.asof(a) - 1, K=g.nav.iloc[-1] / g.nav.iloc[0] - 1,
                    index_held_days=int((g.index_parked > 0.05 * g.nav).sum())))
E = pd.DataFrame(eps); E.to_csv("../audit/k_trades_pauses.csv", index=False)
long_ = E[E.sessions >= 20]
print(f"\n== 3. PAUSES: {len(E)} episodes, {len(long_)} lasting 20+ sessions; SPY during pauses summed {E.spy.sum():.2f}, K {E.K.sum():.2f}")
print(long_.round(3).to_string(index=False))
rsi = m.etf_rsi("SPY"); sp = m.E["SPY"]["c"]
trig = pd.Series([(rsi[i] <= 25) or (i > 21 and sp[i] <= np.nanmax(sp[i - 20:i + 1]) * 0.82) for i in range(len(D))], index=D)
holds = []
last = None
for t in trig[trig].index:
    if t < d.index[0] or (last is not None and (D.get_loc(t) - D.get_loc(last)) < 21) or not bool(d.paused.asof(t)):
        continue
    last = t; k = D.get_loc(t); e = D[min(k + 21, len(D) - 1)]
    kind = "RSI <= 25" if rsi[k] <= 25 else "18% fall"
    holds.append(dict(trigger=t.date(), kind=kind, spy_next_21=spy[e] / spy[t] - 1, qqq_next_21=qqq[e] / qqq[t] - 1))
H = pd.DataFrame(holds); H.to_csv("../audit/k_trades_crash_holds.csv", index=False)
print(f"\nCrash-trigger index holds while paused: {len(H)}; SPY next 21 sessions: average {H.spy_next_21.mean():.3f}, up {(H.spy_next_21 > 0).mean():.0%}")
print(H.round(3).to_string(index=False))

# ---- 4. restarts (Coppock / cyclicals vs defensives) fired during pauses
rs, _ = m.ra_series("cd100")
cop = m.coppock
fires = []
for k in range(300, len(D)):
    if D[k] < d.index[0] or not bool(d.paused.asof(D[k])):
        continue
    c1 = cop[k] == cop[k] and cop[k] < 0 and cop[k] > cop[k - 21] and cop[k - 21] <= cop[k - 42]
    c2 = (not m.ra_below(rs, 100, k)) and m.ra_below(rs, 100, k - 1)
    if c1 or c2:
        e = D[min(k + 42, len(D) - 1)]
        fires.append(dict(date=D[k].date(), signal="Coppock" if c1 else "cyclicals/defensives", spy_next_42=spy[e] / spy[D[k]] - 1))
Fz = pd.DataFrame(fires); Fz.to_csv("../audit/k_trades_restarts.csv", index=False)
print(f"\n== 4. RESTARTS during pauses: {len(Fz)}; SPY next 42 sessions average {Fz.spy_next_42.mean():.3f}, up {(Fz.spy_next_42 > 0).mean():.0%}")
print(Fz.groupby("signal").spy_next_42.agg(["size", "mean", lambda x: (x > 0).mean()]).round(3).to_string())

json.dump(dict(summary=summ, crash_holds=len(H), crash_hold_avg=float(H.spy_next_21.mean()), restarts=len(Fz),
               restart_avg=float(Fz.spy_next_42.mean()), pauses=len(E)), open("../audit/k_trades_summary.json", "w"), indent=1)
