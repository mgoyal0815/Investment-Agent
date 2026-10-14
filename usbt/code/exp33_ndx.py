"""Experiment 33: widen the universe to point-in-time S&P 500 OR Nasdaq-100 members (NDX from 2004).
System C and nearby variants, one run 2000-2026 each, calendar-year beat-all, worst fall."""
import json, multiprocessing as mp, sys
import numpy as np
import pandas as pd
import engine as X, metrics as M
from exp24_timing import mat

SC = json.load(open("../system_C.json"))["cfg"]
MM = None


def _init(path):
    global MM
    MM = X.Market(pd.read_pickle(path))


def one(a):
    lab, cfg, s, e = a
    r = X.run(MM, mat(cfg), start=s, end=e)
    sc = M.score(MM, r["curve"])
    t = sc.pop("table")
    return lab, sc, t, r["curve"]


def grid():
    g = [("System C, S&P 500 only", SC), ("System C, S&P 500 + Nasdaq-100", dict(SC, universe="sp500+ndx"))]
    for N in (5, 7, 12):
        g.append((f"N{N}, S&P+NDX", dict(SC, universe="sp500+ndx", N=N)))
    for b in (1.0, 2.0, 3.0):
        g.append((f"buffer {b}, S&P+NDX", dict(SC, universe="sp500+ndx", buffer=b)))
    g.append(("no gold slots, S&P+NDX", dict(SC, universe="sp500+ndx", gold_slots=0)))
    g.append(("score nifty, S&P+NDX", dict(SC, universe="sp500+ndx", score="nifty")))
    g.append(("no profit lock, S&P+NDX", dict(SC, universe="sp500+ndx", lock_at=None)))
    g.append(("ETFs in universe, S&P+NDX", dict(SC, universe="sp500+ndx", etf_in_universe=True)))
    return g


if __name__ == "__main__":
    pool = mp.get_context("fork").Pool(4, initializer=_init, initargs=("../data/panel_ndx.pkl",))
    G = grid()
    res = pool.map(one, [(l, c, "2000-01-03", None) for l, c in G], chunksize=1)
    rows, years = [], {}
    for lab, sc, t, cv in res:
        full = t.loc[2000:2025]
        win = (full.sys > full[["SPY", "QQQ", "SMH"]].max(axis=1))
        rows.append(dict(label=lab, cagr=sc["cagr"], maxdd=sc["maxdd"], beat_all=win.mean(), beat_SPY=(full.sys > full.SPY).mean(),
                         beat_QQQ=(full.sys > full.QQQ).mean(), beat_SMH=(full.sys > full.SMH.fillna(-9)).mean(),
                         b0411=win.loc[2004:2011].mean(), b1219=win.loc[2012:2019].mean(), b2025=win.loc[2020:2025].mean()))
        years[lab] = t.sys
    df = pd.DataFrame(rows)
    df.to_csv("../results/exp33_ndx.csv", index=False)
    Y = pd.DataFrame(years); Y.to_csv("../results/exp33_ndx_years.csv")
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    print((Y * 100).round(1).T.to_string())
