"""Full backtest of a frozen system file (default ../system_A.json):
  * one continuous run from Jan 2000, one from Jan 2012, one from Jan 2020 (fresh $10,000 each)
  * year-by-year returns against SPY, QQQ, SMH
  * fresh $10,000 every quarter from 2000 to 2025, held to today and held 3 years
Writes results/final_<name>_*.csv and prints the summary."""
import json, sys, multiprocessing as mp
import numpy as np
import pandas as pd
import engine as X, metrics as M
from exp2_search import materialise

PATH = sys.argv[1] if len(sys.argv) > 1 else "../system_A.json"
SYSF = json.load(open(PATH))
NAME, CFG = SYSF["name"], materialise(SYSF["cfg"])
MM = None
def _init():
    global MM
    MM = X.Market(pd.read_pickle("../data/panel.pkl"))
def _fresh(a):
    s, e = a
    cv = X.run(MM, CFG, start=s, end=e)["curve"]
    b = M.bench_curves(MM, cv.index)
    return dict(start=s, end=str(cv.index[-1].date()), cagr=M.cagr(cv), dd=M.maxdd(cv),
                **{f"cagr_{k}": M.cagr(v) for k, v in b.items()}, **{f"dd_{k}": M.maxdd(v) for k, v in b.items()})
if __name__ == "__main__":
    m = X.Market(pd.read_pickle("../data/panel.pkl"))
    tag = NAME.replace(" ", "_")
    out = []
    for s in ("2000-01-03", "2012-01-03", "2020-01-02"):
        r = X.run(m, CFG, start=s)
        sc = M.score(m, r["curve"]); t = sc.pop("table")
        out.append(dict(run=f"from {s[:4]}", final=r["curve"].iloc[-1], costs=r["costs"], **sc))
        print(f"from {s[:4]}: $10,000 -> ${r['curve'].iloc[-1]:,.0f}  " + M.line("", sc))
        if s == "2000-01-03":
            pd.DataFrame({"date": r["curve"].index, "nav": r["curve"].values}).to_csv(f"../results/final_{tag}_curve2000.csv", index=False)
    pd.DataFrame(out).to_csv(f"../results/final_{tag}_runs.csv", index=False)
    spans = (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None))
    yrs = {NAME: pd.concat([M.yearly(X.run(m, CFG, start=s, end=e)["curve"]) for s, e in spans]) * 100}
    full = m.dates[m.dates >= "2000-01-03"]
    for k, v in M.bench_curves(m, full).items():
        yrs[k] = pd.concat([M.yearly(v[s:e]) for s, e in spans]) * 100
    t = pd.DataFrame(yrs).round(1)
    t["beat all 3"] = (t[NAME] > t[["SPY", "QQQ", "SMH"]].max(axis=1)) & (t.index < 2026)
    t.to_csv(f"../results/final_{tag}_years.csv")
    print(t.to_string())
    print("beat all 3:", int(t["beat all 3"].sum()), "of 26 full years")
    qs = [str(m.dates[m.dates.searchsorted(d)].date()) for d in pd.date_range("2000-01-01", "2025-10-01", freq="QS")]
    pool = mp.get_context("fork").Pool(4, initializer=_init)
    held = pool.map(_fresh, [(s, None) for s in qs])
    three = pool.map(_fresh, [(s, str((pd.Timestamp(s) + pd.DateOffset(years=3)).date())) for s in qs if s < "2023-10"])
    for lab, rows in (("held to today", held), ("held 3 years", three)):
        f = pd.DataFrame(rows)
        f.to_csv(f"../results/final_{tag}_starts_{lab.replace(' ', '_')}.csv", index=False)
        f["era"] = pd.cut(pd.to_datetime(f.start).dt.year, [1999, 2011, 2019, 2026], labels=["2000-11", "2012-19", "2020-25"])
        g = f.groupby("era", observed=True)
        print(f"\nFRESH QUARTERLY STARTS, {lab}")
        print(pd.DataFrame({"starts": g.size(), "median CAGR": g.cagr.median(), "worst CAGR": g.cagr.min(),
                            "worst fall": g.dd.min(),
                            "beat SPY": g.apply(lambda x: (x.cagr > x.cagr_SPY).mean()),
                            "beat all 3": g.apply(lambda x: (x.cagr > x[["cagr_SPY", "cagr_QQQ", "cagr_SMH"]].max(axis=1)).mean()),
                            "fall < SMH's": g.apply(lambda x: (x.dd > x.dd_SMH).mean()),
                            "fall < all 3": g.apply(lambda x: (x.dd > x[["dd_SPY", "dd_QQQ", "dd_SMH"]].max(axis=1)).mean())}).round(3).to_string())
