"""Owner, 17 Oct 2026: copy the running System R1 account, scaled to $10,000, on the first session of every month
Jan 2016 - Dec 2025 (120 starts); (1) whole shares only, (2) fractional shares. Each copy then runs under R1's rules
to 31 Dec 2025 (engine init, mode continue; the reference's orders decided at the previous close are scaled too).
Reference: R1 run continuously from 2000-01-03. Scored with yardstick.judge (calendar years from the start to each
December, rolling 3-year windows vs SMH, worst fall) plus the first partial year on its own."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, math, multiprocessing as mp, numpy as np, pandas as pd, yardstick as Y, engine as X
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"
R1 = json.load(open("../system_R1.json"))["cfg"]
CAP, END = 10000.0, "2025-12-31"


def copy_init(m, ref, ip, s, frac):
    k = CAP / float(ref["curve"].iloc[-1]); J = {sym: j for j, sym in enumerate(m.syms)}
    rnd = (lambda x: math.floor(x * 1e6) / 1e6) if frac else math.floor
    cash, stocks, etfs = CAP, {}, {}
    for sym, q in ref["book"]["positions"].items():
        sf = q["sf"] or 1.0; px = float(m.P["Cf"][ip, J[sym]]) * float(m.P["SF"][ip, J[sym]]); n = rnd(q["adj_sh"] / sf * k)
        if n > 0 and n * px + X.tcost(n, n * px) <= cash:
            cash -= n * px + X.tcost(n, n * px)
            stocks[sym] = dict(shares=n, cost=q["cost"] * sf, peak=q["peak"] * sf, entry_date=q["entry"], tr=q["tr"], giant=q["giant"], adds=q["adds"], carried=False)
    for t, sh in ref["book"]["parked"].items():
        px = float(m.E[t]["c"][ip]); n = rnd(sh * k)
        if n > 0 and n * px + X.tcost(n, n * px) <= cash:
            cash -= n * px + X.tcost(n, n * px); etfs[t] = n
    pend = [dict(o, dollars=(o["dollars"] or 0) * k, min_dollars=(o.get("min_dollars") or 0) * k) for o in ref["pending"]]
    return dict(first_session=str(s.date()), mode="continue", cash=cash, stocks=stocks, etfs=etfs, pending=pend), 1 - cash / CAP, len(stocks)


def one(s):
    m = Y.MM; s = pd.Timestamp(s); ip = m.dates.get_loc(s) - 1; p = m.dates[ip]
    ref = X.run(m, mat(R1), start="2000-01-03", end=str(p.date()))
    e = m.dates[m.dates.searchsorted(pd.Timestamp(f"{s.year}-12-31"), side="right") - 1]
    out = []
    for frac in (False, True):
        init, inv, held = copy_init(m, ref, ip, s, frac)
        r = X.run(m, mat(dict(R1, init=init, fractional=frac, whole_etf=not frac)), start=str((s - pd.DateOffset(years=4)).date()), end=END)
        c = r["curve"].loc[p:].copy(); c.iloc[0] = CAP
        j = Y.judge(c, m)
        fy = float(c.loc[e] / CAP - 1)
        b = {t: float(m.E[t]["c"][m.dates.get_loc(e)] / m.E[t]["c"][ip] - 1) for t in ("SPY", "QQQ", "SMH")}
        out.append(dict(start=str(s.date()), frac=frac, invested_day1=inv, held_day1=held, first_year=fy, **{f"{t}_first": v for t, v in b.items()},
                        first_beat_both=fy > b["SPY"] and fy > b["QQQ"], first_beat_smh=fy > b["SMH"],
                        cal_win=j["cal_win"], cal_n=j["cal_n"], r3_win=j["r3_win"], dd=j["dd"], dd_ok=j["dd_ok"], cagr=j["cagr"], end_value=float(c.iloc[-1])))
    return out


if __name__ == "__main__":
    Y._init(PATH)
    st = [str(d.date()) for d in Y.starts(a="2016-01-01", b="2025-12-31", step=1, day=1)]
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    T = pd.DataFrame([r for rs in pool.map(one, st, chunksize=1) for r in rs])
    T.to_csv("a77_copy_monthly.csv", index=False)
    rows = []
    for frac, g in T.groupby("frac"):
        rows.append(dict(case="fractional" if frac else "whole shares", starts=len(g), invested_day1=g.invested_day1.mean(),
                         first_year_beat_SPY_QQQ=g.first_beat_both.mean(), first_year_beat_SMH=g.first_beat_smh.mean(),
                         calendar_years_beat_SPY_QQQ=(g.cal_win * g.cal_n).sum() / g.cal_n.sum(), r3_beat_SMH=g.r3_win.mean(),
                         worst_fall_ok=g.dd_ok.mean(), worst_fall=g.dd.min(), median_growth=g.cagr.median()))
    S = pd.DataFrame(rows); S.to_csv("a77_summary.csv", index=False)
    T["year"] = T.start.str[:4]
    Yr = T.groupby(["year", "frac"]).agg(first_year_median=("first_year", "median"), beat_both=("first_beat_both", "mean"), beat_smh=("first_beat_smh", "mean"),
                                         SPY=("SPY_first", "median"), QQQ=("QQQ_first", "median"), SMH=("SMH_first", "median")).unstack("frac")
    Yr.to_csv("a77_by_year.csv")
    json.dump(dict(env=Y.ENV, system="system_R1.json", capital=CAP, end=END, starts=st), open("a77_copy_monthly.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False)); print((Yr * 100).round(1).to_string())
