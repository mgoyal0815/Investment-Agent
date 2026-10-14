"""Experiment 44: full check of the combination spotted in exp43 (trend-hold + MACD filter + profit lock at
+50%), which was found by looking at the check periods, so it needs every monthly start and a year table
before it can count. Compared with System D."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat

SD = json.load(open("../system_D.json"))["cfg"]
G = {"System D": SD, "D + trend-hold + MACD + lock 50%": dict(SD, trend_hold=True, macd_pos=True, lock_at=0.5),
     "D + trend-hold + lock 50%": dict(SD, trend_hold=True, lock_at=0.5), "D + lock 50%": dict(SD, lock_at=0.5)}
if __name__ == "__main__":
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    PER = {"starts 2000-11": (Y.starts("2000-01-01", "2011-12-31"), None), "starts 2012-19": (Y.starts("2012-01-01", "2019-12-31"), None),
           "starts 2020-25": (Y.starts("2020-01-01", "2025-10-31"), None), "every start": (Y.starts(), None)}
    rows, years = [], {}
    for lab, c in G.items():
        for p, (st, e) in PER.items():
            s = Y.summary(Y.evaluate(pool, c, st, e))
            rows.append(dict(label=lab, period=p, beat=s["yrs_beat_SPY_QQQ"], cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], ddok=s["dd_ok"],
                             cagr=s["med_cagr"], worst=s["worst_dd"]))
        cv = X.run(Y.MM, mat(c), start="2000-01-03")["curve"]
        years[lab] = M.yearly(cv) * 100
        years[lab].loc["CAGR"] = M.cagr(cv) * 100; years[lab].loc["worst fall"] = M.maxdd(cv) * 100
        print(lab, flush=True)
    R = pd.DataFrame(rows); R.to_csv("../results/exp44_candidate.csv", index=False)
    b = M.bench_curves(Y.MM, cv.index)
    for k, v in b.items():
        years[k] = M.yearly(v) * 100; years[k].loc["CAGR"] = M.cagr(v) * 100; years[k].loc["worst fall"] = M.maxdd(v) * 100
    Yt = pd.DataFrame(years).round(1); Yt.to_csv("../results/exp44_years.csv")
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False)); print(Yt.to_string())
