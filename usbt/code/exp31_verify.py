"""Experiment 31: verify exp30 candidates in the real engine. While the boom switch is off the account
runs System C (engine, whole shares, costs); when it turns on, System C is sold at that day's close and
the account holds the leveraged ETF slice + System C... (frac<1: the rest stays in a System C run sized to
the remainder). When it turns off, System C restarts with the whole account (no ramp)."""
import json, sys
import numpy as np
import pandas as pd
import engine as X
import exp26_lev as E
import exp30_boom as B
from exp2_search import materialise

D = E.D
SYS = json.load(open("../system_C.json"))["cfg"]
m = X.Market(pd.read_pickle("../data/panel.pkl"))


def c_run(start, end, capital, ramp):
    cfg = materialise(dict(SYS, capital=capital, ramp=ramp))
    r = X.run(m, cfg, start=str(start.date()), end=str(end.date()))
    return r["curve"]


def chained(params, start="2000-01-03", cost=0.002):
    L, frac = params["L"], params["frac"]
    st = B.run(**params, states=True).loc[start:]
    rL = E.RET[("SMH", L)]
    # segments of constant state; the state set at close t applies from t+1
    eff = st.shift(1).fillna(False).astype(bool)
    seg_id = (eff != eff.shift(1)).cumsum()
    nav = pd.Series(np.nan, index=eff.index)
    v = 10000.0
    first = True
    for _, g in eff.groupby(seg_id):
        a, b = g.index[0], g.index[-1]
        on = bool(g.iloc[0])
        if not first:
            v *= 1 - cost
        if on:
            lev_part = v * frac * (1 + rL.loc[a:b]).cumprod()
            if frac < 1:
                cpart = c_run(a, b, v * (1 - frac), 0).reindex(g.index).ffill()
            else:
                cpart = 0.0
            seg = lev_part + cpart
        else:
            seg = c_run(a, b, v, 3 if first else 0).reindex(g.index).ffill()
        nav.loc[a:b] = seg.values
        v = float(seg.iloc[-1]); first = False
    return nav, st


CANDS = {
    "A: 2x, 2/3 slice, hi10, x15": dict(L=2, sma=200, rel="spy", dec="W", hi=0.1, spyf=False, x=0.15, fast=None, frac=0.67),
    "B: 2x, 2/3 slice, hi20, SPY filter": dict(L=2, sma=200, rel="spy", dec="W", hi=0.2, spyf=True, x=None, fast=None, frac=0.67),
    "C: 3x, 2/3 slice, hi20, SPY filter, x15": dict(L=3, sma=200, rel="spy", dec="W", hi=0.2, spyf=True, x=0.15, fast=None, frac=0.67),
    "D: 3x, full, hi20, x20": dict(L=3, sma=200, rel="spy", dec="W", hi=0.2, spyf=False, x=0.2, fast=None, frac=1.0),
}

if __name__ == "__main__":
    rows, curves = [], {}
    for lab, p in CANDS.items():
        nav, st = chained(p)
        curves[lab] = nav
        approx = B.run(**p)
        print(f"{lab}: switches {int((st != st.shift(1)).sum())}, time in lev {st.mean():.0%}", flush=True)
        for per, (a, b) in B.PER.items():
            for kind, s in (("engine", nav), ("approx", approx)):
                c = E.card(s, a, b, lab=f"{lab} [{kind}]"); c["per"] = per; rows.append(c)
    R = pd.DataFrame(rows); R.to_csv("../results/exp31_verify.csv", index=False)
    pd.DataFrame(curves).to_csv("../results/exp31_curves.csv")
    W = B.table(R)
    pd.set_option("display.width", 280, "display.max_columns", 20, "display.max_colwidth", 60)
    print(W[B.COLS].round(3))
    for lab, s in curves.items():
        y = pd.DataFrame({"sys": E.yearly(s), **{k: E.yearly(E.IDX[k].loc["2000":]) for k in E.IDX}})
        y["win"] = y.sys > y[["SPY", "QQQ", "SMH"]].max(axis=1)
        print("\n", lab); print((y.drop(columns="win") * 100).round(1).assign(win=y.win).T.to_string())
