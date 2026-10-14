"""Experiment 29: one account. Risk-on (SMH trend + leadership): hold a volatility-targeted amount of the
3x semiconductor ETF (SOXL; synthetic before 2010) and put the rest in System C's stocks or cash.
Risk-off: System C. Optional bear filter: SPY below its 200-day -> cash. No margin, no options:
exposure above 1x comes only from holding part of the account in the 3x ETF."""
import itertools
import numpy as np
import pandas as pd
import exp26_lev as E
from exp27_switch import C, rC, px, D

WK = set(pd.Series(D, index=D).groupby([D.year, D.isocalendar().week.values]).last().values)
MO = set(pd.Series(D, index=D).groupby([D.year, D.month]).last().values)
smh, spy = px.SMH.values, px.SPY.values
r3 = E.RET[("SMH", 3)].values
cash = E.RC.values
rc = rC.values
m126 = (px / px.shift(126) - 1)
RS = {"spy": (m126.SMH > m126.SPY).values, "both": (m126.SMH > m126[["SPY", "QQQ"]].max(axis=1)).values,
      "none": np.ones(len(D), bool)}
RV = {n: (px.SMH.pct_change().rolling(n).std() * np.sqrt(252)).values for n in (20, 63)}
MA = {n: px.SMH.rolling(n).mean().values for n in (50, 100, 150, 200)}
SPYMA = px.SPY.rolling(200).mean().values


def run(sma=200, band=0.0, rel="spy", dec="W", tv=0.6, Lmax=3.0, rvn=20, rest="C", bear=False, fast=None, cost=0.002, start="2000-01-03"):
    n = len(D)
    out = np.zeros(n)
    w3 = 0.0        # weight of the 3x ETF
    on = False
    days = MO if dec == "M" else WK
    ma, rv = MA[sma], RV[rvn]
    fm = MA[fast] if fast else None
    for i in range(1, n):
        rest_r = rc[i] if rest == "C" else cash[i]
        if bear and spy[i - 1] < SPYMA[i - 1] and not on:
            rest_r = cash[i]
        out[i] = w3 * r3[i] + (1 - w3) * (rest_r if on else (cash[i] if (bear and spy[i - 1] < SPYMA[i - 1]) else rc[i]))
        # drift of the weight inside the day
        if w3 > 0:
            g3 = w3 * (1 + r3[i]); w3 = g3 / (1 + out[i])
        if np.isnan(ma[i]):
            continue
        new_on = on
        if on and fm is not None and smh[i] < fm[i]:
            new_on = False
        if D[i] in days:
            new_on = smh[i] > ma[i] * (1 + band) and RS[rel][i] and not (fm is not None and smh[i] < fm[i])
        tgt = min(Lmax, tv / max(rv[i], 1e-6)) / 3.0 if new_on else 0.0
        if new_on != on or (new_on and D[i] in days and abs(tgt - w3) > 0.1 * max(tgt, 0.05)) or (new_on and abs(tgt - w3) > 0.5 * tgt):
            out[i] -= cost * abs(tgt - w3) * 2 if new_on == on else cost
            w3 = tgt
        on = new_on
    r = pd.Series(out, index=D).loc[start:]
    return (1 + r).cumprod()


PER = {"train": ("2012-01-03", "2019-12-31"), "early": ("2000-01-03", "2011-12-30"), "test": ("2020-01-02", None), "all": ("2000-01-03", None)}

if __name__ == "__main__":
    rows = []
    grid = itertools.product((150, 200), (0.0, 0.05), ("spy", "both"), ("W", "M"), (0.4, 0.6, 0.8, 1.0), (2.0, 3.0), (20, 63),
                             ("C", "cash"), (False, True), (None, 50))
    for sma, band, rel, dec, tv, Lmax, rvn, rest, bear, fast in grid:
        nav = run(sma, band, rel, dec, tv, Lmax, rvn, rest, bear, fast)
        lab = f"sma{sma} b{band} {rel} {dec} tv{tv} L{Lmax} rv{rvn} rest={rest} bear={bear} fast{fast}"
        for per, (a, b) in PER.items():
            c = E.card(nav, a, b, lab=lab); c["per"] = per; rows.append(c)
    R = pd.DataFrame(rows); R.to_csv("../results/exp29_voltarget.csv", index=False)
    W = R.pivot(index="lab", columns="per", values=["beat_all", "cagr", "dd"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    pd.set_option("display.width", 280, "display.max_columns", 20, "display.max_colwidth", 70)
    cols = ["beat_all_train", "beat_all_early", "beat_all_test", "beat_all_all", "cagr_train", "dd_train", "cagr_all", "dd_all", "dd_early", "dd_test"]
    W["sel"] = W.beat_all_train + 0.5 * W.cagr_train + 0.5 * W.dd_train
    print(W.sort_values("sel", ascending=False)[cols].head(20).round(3))
    print("\nall-period DD better than -45%")
    print(W[W.dd_all > -0.45].sort_values(["beat_all_all", "cagr_all"], ascending=False)[cols].head(20).round(3))
