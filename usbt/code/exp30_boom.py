"""Experiment 30: hold the 3x/2x semiconductor ETF only in a confirmed boom: SMH above its 200-day,
SMH leading SPY over 6 months, SMH within h of its 52-week high, SPY above its 200-day; exit any day
SMH falls x from its 63-day high (or below its 50-day). Otherwise System C. One account."""
import itertools
import numpy as np
import pandas as pd
import exp26_lev as E
from exp27_switch import C, rC, px, D

WK = set(pd.Series(D, index=D).groupby([D.year, D.isocalendar().week.values]).last().values)
MO = set(pd.Series(D, index=D).groupby([D.year, D.month]).last().values)
smh, spy = px.SMH.values, px.SPY.values
cash, rc = E.RC.values, rC.values
m126 = px / px.shift(126) - 1
RS = {"spy": (m126.SMH > m126.SPY).values, "both": (m126.SMH > m126[["SPY", "QQQ"]].max(axis=1)).values, "none": np.ones(len(D), bool)}
MA = {n: px.SMH.rolling(n).mean().values for n in (50, 100, 150, 200)}
SPYMA = px.SPY.rolling(200).mean().values
HI252 = (px.SMH / px.SMH.rolling(252, min_periods=150).max()).values
HI63 = (px.SMH / px.SMH.rolling(63).max()).values


def run(L=3, sma=200, rel="spy", dec="W", hi=None, spyf=False, x=None, fast=None, frac=1.0, cost=0.002, start="2000-01-03", states=False):
    rL = E.RET[("SMH", L)].values
    n = len(D); out = np.zeros(n); on = False; st = np.zeros(n, bool)
    days = MO if dec == "M" else WK
    ma = MA[sma]
    for i in range(1, n):
        out[i] = (frac * rL[i] + (1 - frac) * rc[i]) if on else rc[i]
        if np.isnan(ma[i]):
            continue
        new = on
        if on and ((x and HI63[i] < 1 - x) or (fast and smh[i] < MA[fast][i])):
            new = False
        elif D[i] in days:
            new = (smh[i] > ma[i] and RS[rel][i] and (hi is None or HI252[i] > 1 - hi) and (not spyf or spy[i] > SPYMA[i])
                   and not (x and HI63[i] < 1 - x) and not (fast and smh[i] < MA[fast][i]))
        if new != on:
            out[i] -= cost * frac
        on = new; st[i] = on
    if states:
        return pd.Series(st, index=D)
    return (1 + pd.Series(out, index=D).loc[start:]).cumprod()


PER = {"train": ("2012-01-03", "2019-12-31"), "early": ("2000-01-03", "2011-12-30"), "test": ("2020-01-02", None), "all": ("2000-01-03", None)}


def table(R):
    W = R.pivot(index="lab", columns="per", values=["beat_all", "cagr", "dd"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    return W


COLS = ["beat_all_train", "beat_all_early", "beat_all_test", "beat_all_all", "cagr_train", "dd_train", "cagr_all", "dd_all", "dd_early", "dd_test"]

if __name__ == "__main__":
    rows = []
    for L, sma, rel, dec, hi, spyf, x, fast, frac in itertools.product((2, 3), (150, 200), ("spy", "both"), ("W", "M"), (None, 0.1, 0.2),
                                                                       (False, True), (None, 0.1, 0.15, 0.2), (None, 50), (1.0, 0.67)):
        nav = run(L, sma, rel, dec, hi, spyf, x, fast, frac)
        lab = f"L{L} f{frac} sma{sma} {rel} {dec} hi{hi} spyf{spyf} x{x} fast{fast}"
        for per, (a, b) in PER.items():
            c = E.card(nav, a, b, lab=lab); c["per"] = per; rows.append(c)
    R = pd.DataFrame(rows); R.to_csv("../results/exp30_boom.csv", index=False)
    W = table(R)
    pd.set_option("display.width", 280, "display.max_columns", 20, "display.max_colwidth", 70)
    W["sel"] = W.beat_all_train + 0.5 * W.cagr_train + 0.5 * W.dd_train
    print(W.sort_values("sel", ascending=False)[COLS].head(20).round(3))
    print("\nall-period DD better than -55% (SPY's)")
    print(W[W.dd_all > -0.55].sort_values(["beat_all_all", "cagr_all"], ascending=False)[COLS].head(25).round(3))
