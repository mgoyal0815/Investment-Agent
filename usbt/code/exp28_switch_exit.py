"""Experiment 28: exp27's switch plus DAILY crash exits on the leveraged leg (entry on the weekly/monthly
decision, exit any day): SMH below a fast average, a trailing stop on the sleeve, a volatility cap.
After an exit the money goes back to System C's stocks (or cash) until the next entry decision."""
import itertools, sys
import numpy as np
import pandas as pd
import exp26_lev as E
from exp27_switch import C, rC, px, D

WK = set(pd.Series(D, index=D).groupby([D.year, D.isocalendar().week.values]).last().values)
MO = set(pd.Series(D, index=D).groupby([D.year, D.month]).last().values)
smh = px.SMH
rv20 = smh.pct_change().rolling(20).std() * np.sqrt(252)
mom = {lb: px / px.shift(lb) - 1 for lb in (63, 126)}
CASH = E.RC


def run(L=2, sma=150, band=0.05, lb=126, rel="spy", dec="M", fast=50, trail=None, vcap=None, after="C", cool=0, cost=0.002):
    rL = E.RET[("SMH", L)].values
    ma = smh.rolling(sma).mean().values
    fm = smh.rolling(fast).mean().values if fast else None
    s, m6 = smh.values, mom[lb]
    rs = (m6.SMH > m6.SPY).values if rel == "spy" else ((m6.SMH > m6[["SPY", "QQQ"]].max(axis=1)).values if rel == "both" else np.ones(len(D), bool))
    vv = rv20.values
    days = MO if dec == "M" else WK
    state, peak, val, last_exit = False, 1.0, 1.0, -999
    r_out = np.zeros(len(D))
    rc, cash = rC.values, CASH.values
    i0 = D.searchsorted(pd.Timestamp("2000-01-03"))
    for i in range(len(D)):
        # return of today from yesterday's state
        if i > 0:
            r_out[i] = rL[i] if state else (rc[i] if after == "C" or not was_lev_exit else cash[i])
            if state:
                val *= 1 + rL[i]; peak = max(peak, val)
        prev = state
        if state:   # daily exits
            ex = (fm is not None and s[i] < fm[i]) or (trail and val / peak - 1 < -trail) or (vcap and vv[i] > vcap)
            if ex:
                state, last_exit = False, i
        if D[i] in days and not np.isnan(ma[i]) and i - last_exit >= cool:
            on = s[i] > ma[i] * (1 + band) and rs[i] and not (vcap and vv[i] > vcap) and not (fm is not None and s[i] < fm[i])
            if on and not state:
                state, val, peak = True, 1.0, 1.0
            elif not on and state:
                state = False
        was_lev_exit = False
        if prev != state and i > 0:
            r_out[i] -= cost
    r = pd.Series(r_out, index=D).loc["2000-01-03":]
    return (1 + r).cumprod()


PER = {"train": ("2012-01-03", "2019-12-31"), "early": ("2000-01-03", "2011-12-30"), "test": ("2020-01-02", None), "all": ("2000-01-03", None)}

if __name__ == "__main__":
    rows = []
    grid = itertools.product((2, 3), (150, 200), (0.0, 0.05), ("spy", "both", "none"), ("W", "M"), (None, 20, 50),
                             (None, 0.15, 0.25), (None, 0.45, 0.6))
    for L, sma, band, rel, dec, fast, trail, vcap in grid:
        nav = run(L, sma, band, 126, rel, dec, fast, trail, vcap)
        lab = f"L{L} sma{sma} b{band} {rel} {dec} fast{fast} trail{trail} vcap{vcap}"
        for per, (a, b) in PER.items():
            c = E.card(nav, a, b, lab=lab); c["per"] = per; rows.append(c)
    R = pd.DataFrame(rows); R.to_csv("../results/exp28_switch_exit.csv", index=False)
    W = R.pivot(index="lab", columns="per", values=["beat_all", "cagr", "dd"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    pd.set_option("display.width", 260, "display.max_columns", 20, "display.max_colwidth", 60)
    cols = ["beat_all_train", "beat_all_early", "beat_all_test", "beat_all_all", "cagr_train", "dd_train", "cagr_all", "dd_all", "dd_early", "dd_test"]
    W["sel"] = W.beat_all_train + 0.5 * W.cagr_train + 0.5 * W.dd_train
    print(W.sort_values("sel", ascending=False)[cols].head(25).round(3))
    print("\nby all-period beat-all with all-period DD > -40%")
    print(W[W.dd_all > -0.40].sort_values("beat_all_all", ascending=False)[cols].head(20).round(3))
