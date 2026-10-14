"""Experiment 32 (diagnostic, no leverage): could concentrated S&P 500 stock picks beat the best index
in its own boom years? Monthly, equal weight, top-k by 126-day return among point-in-time S&P 500 members
above their 200-day; variants restrict to names that move with the leading index (corr >= 0.6)."""
import numpy as np
import pandas as pd
import engine as X

m = X.Market(pd.read_pickle("../data/panel.pkl"))
P = m.P; D = m.dates
Cf = P["Cf"]; mem = m.sp_member
r1 = np.nan_to_num(P["ret1"].astype(float))
mom = np.full_like(Cf, np.nan); mom[126:] = Cf[126:] / Cf[:-126] - 1
above = Cf > P["SMA200"]
IDX = {k: pd.Series(m.E[k]["c"], index=D) for k in ("SPY", "QQQ", "SMH")}
me = m.month_end


def run(k=5, corr=None, leader_only=False):
    nav = np.ones(len(D)); w = np.zeros(Cf.shape[1])
    for i in range(1, len(D)):
        nav[i] = nav[i - 1] * (1 + (w * r1[i]).sum())
        if me[i]:
            ok = mem[i] & above[i] & ~np.isnan(mom[i]) & ~m.is_etf
            if corr:
                ldr = max(IDX, key=lambda t: IDX[t].iloc[i] / IDX[t].iloc[max(i - 126, 0)])
                c = P["corr_" + ldr][i] if ldr != "SPY" else np.ones(Cf.shape[1])
                ok &= np.nan_to_num(c) >= corr
            s = np.where(ok, mom[i], -np.inf)
            top = np.argsort(-s)[:k]; top = top[np.isfinite(s[top])]
            w = np.zeros(Cf.shape[1]); w[top] = 1 / k if len(top) else 0
    return pd.Series(nav, index=D)


def yearly(s):
    ye = s.groupby(s.index.year).last(); p = ye.shift(1); p.iloc[0] = s.iloc[0]; return ye / p - 1


out = {}
for lab, kw in {"top5": dict(k=5), "top10": dict(k=10), "top5 leader-corr0.6": dict(k=5, corr=0.6), "top10 leader-corr0.6": dict(k=10, corr=0.6),
                "top3": dict(k=3)}.items():
    out[lab] = yearly(run(**kw).loc["2000-01-03":])
Y = pd.DataFrame({**out, **{k: yearly(v.loc["2000-01-03":]) for k, v in IDX.items()}})
mx = Y[["SPY", "QQQ", "SMH"]].max(axis=1)
pd.set_option("display.width", 220)
print((Y * 100).round(1).to_string())
print("\nbeat all 3, 2001-2025:")
for c in out:
    print(c, f"{(Y.loc[2001:2025, c] > mx.loc[2001:2025]).mean():.0%}")

# hindsight ceiling: in each year pick whichever of these rules (or System C) did best that year
C = pd.read_csv("../results/final_System_C_curve2000.csv", index_col=0, parse_dates=True).nav
Y["System C"] = yearly(C)
best = Y[list(out) + ["System C"]].max(axis=1)
print("hindsight best-of-6 rules beats all 3:", f"{(best.loc[2001:2025] > mx.loc[2001:2025]).mean():.0%}")
print("years even hindsight loses:", list(best.loc[2001:2025][best.loc[2001:2025] <= mx.loc[2001:2025]].index))
Y.to_csv("../results/exp32_potential_years.csv")
