"""Owner, 17 Oct 2026: start a new account by buying the running system's portfolio that day, scaled to the new money,
instead of building positions from scratch. Reference: System R1 run continuously from 2000-01-03. For each year y,
the reference book at the last close of y-1 (stocks with cost, peak, entry date, half and giants flag; parked ETFs;
the orders decided at that close) is scaled to $10,000 in whole shares (fractional with --frac), bought at that close
with the normal trading costs, and the account then runs under R1's rules to the end of y (engine init, mode continue).
Compared with the fresh start (a73) and the reference account's own year."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, math, multiprocessing as mp, numpy as np, pandas as pd, yardstick as Y, engine as X
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"
R1 = json.load(open("../system_R1.json"))["cfg"]
CAP = 10000.0
FRAC = "--frac" in sys.argv


def one(y):
    m = Y.MM
    s = m.dates[m.dates.searchsorted(pd.Timestamp(f"{y}-01-01"))]
    p = m.dates[m.dates.get_loc(s) - 1]                                  # last close of the previous year
    e = m.dates[min(m.dates.searchsorted(pd.Timestamp(f"{y}-12-31"), side="right"), len(m.dates)) - 1]
    ref = X.run(m, mat(R1), start="2000-01-03", end=str(p.date()))
    navr = float(ref["curve"].iloc[-1]); k = CAP / navr
    ip = m.dates.get_loc(p); J = {sym: j for j, sym in enumerate(m.syms)}
    rnd = (lambda x: math.floor(x * 1e6) / 1e6) if FRAC else math.floor
    cash, stocks, spent = CAP, {}, 0.0
    for sym, q in ref["book"]["positions"].items():
        sf = q["sf"] or 1.0; raw = q["adj_sh"] / sf; px = float(m.P["Cf"][ip, J[sym]]) * float(m.P["SF"][ip, J[sym]])
        n = rnd(raw * k)
        if n <= 0:
            continue
        cash -= n * px + X.tcost(n, n * px)
        stocks[sym] = dict(shares=n, cost=q["cost"] * sf, peak=q["peak"] * sf, entry_date=q["entry"], tr=q["tr"], giant=q["giant"], adds=q["adds"], carried=False)
    etfs = {}
    for t, sh in ref["book"]["parked"].items():
        px = float(m.E[t]["c"][ip]); n = rnd(sh * k)
        if n > 0 and n * px + X.tcost(n, n * px) <= cash:
            cash -= n * px + X.tcost(n, n * px); etfs[t] = n
    pend = [dict(o, dollars=(o["dollars"] or 0) * k, min_dollars=(o.get("min_dollars") or 0) * k) for o in ref["pending"]]
    init = dict(first_session=str(s.date()), mode="continue", cash=cash, stocks=stocks, etfs=etfs, pending=pend)
    r = X.run(m, mat(dict(R1, init=init, fractional=FRAC, whole_etf=not FRAC)), start=f"{y - 4}-01-02", end=str(e.date()))
    c = r["curve"].loc[p:]
    c.iloc[0] = CAP                                                   # the money arrives at the previous close
    full = X.run(m, mat(R1), start="2000-01-03", end=str(e.date()))["curve"]
    return dict(year=y, mirror=float(c.iloc[-1] / CAP - 1), mirror_worst=float((c / c.cummax() - 1).min()), held=len(stocks),
                invested_day1=float(1 - cash / CAP), running_account=float(full.loc[e] / full.loc[p] - 1),
                **{t: float(m.E[t]["c"][m.dates.get_loc(e)] / m.E[t]["c"][ip] - 1) for t in ("SPY", "QQQ", "SMH")})


if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    T = pd.DataFrame(pool.map(one, range(2016, 2027), chunksize=1)).set_index("year")
    fresh = pd.read_csv("a73_calendar_years.csv", index_col=0)["R1 frac" if FRAC else "R1 whole"] if False else pd.read_csv("a73_calendar_years.csv", index_col=0)["R1 fractional" if FRAC else "R1 whole"]
    T.insert(1, "fresh_start", fresh)
    tag = "frac" if FRAC else "whole"
    T.to_csv(f"a76_mirror_start_{tag}.csv")
    json.dump(dict(env=Y.ENV, system="system_R1.json", capital=CAP, fractional=FRAC), open(f"a76_mirror_start_{tag}.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print((T * 100).round(1).to_string())
