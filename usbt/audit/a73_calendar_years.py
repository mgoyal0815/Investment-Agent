"""Apples to apples (owner, 17 Oct 2026): System R1 started fresh with $10,000 on the first session of each year and
stopped at that year's last session, vs SPY / QQQ / SMH over the same dates (total return). Whole shares (the live
rule) and fractional shares. Research panel (panel_ndx.pkl, to 2026-10-12); 2026 runs to the panel's last session."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, hashlib, multiprocessing as mp, numpy as np, pandas as pd, yardstick as Y, engine as X
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"
R1 = json.load(open("../system_R1.json"))["cfg"]


def one(a):
    y, frac = a
    m = Y.MM
    s = m.dates[m.dates.searchsorted(pd.Timestamp(f"{y}-01-01"))]
    e = m.dates[min(m.dates.searchsorted(pd.Timestamp(f"{y}-12-31"), side="right"), len(m.dates)) - 1]
    c = X.run(m, mat(dict(R1, fractional=frac, whole_etf=not frac)), start=str(s.date()), end=str(e.date()))["curve"]
    return dict(year=y, frac=frac, start=str(s.date()), end=str(e.date()), r1=float(c.iloc[-1] / 10000 - 1), worst=float((c / c.cummax() - 1).min()),
                **{k: float(m.E[k]["c"][m.dates.get_loc(e)] / m.E[k]["c"][m.dates.get_loc(s)] - 1) for k in ("SPY", "QQQ", "SMH")})


if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    T = pd.DataFrame(pool.map(one, [(y, f) for y in range(2016, 2027) for f in (False, True)], chunksize=1))
    W = T[~T.frac].set_index("year"); F = T[T.frac].set_index("year")
    out = pd.DataFrame({"R1 whole": W.r1, "R1 fractional": F.r1, "SPY": W.SPY, "QQQ": W.QQQ, "SMH": W.SMH, "R1 worst fall": W.worst, "end": W.end})
    out.to_csv("../audit/a73_calendar_years.csv")
    json.dump(dict(env=Y.ENV, system="system_R1.json", capital=10000), open("../audit/a73_calendar_years.json", "w"), indent=1)
    print((out.drop(columns="end") * 100).round(1).to_string()); print(out.end.iloc[-1])
