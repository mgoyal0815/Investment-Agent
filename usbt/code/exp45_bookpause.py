"""Experiment 45: pause on our own stock book's stress instead of the market (owner, 15 Oct 2026).
book_pause: dd = the book's fall from its 63-session high that pauses; ma = the book below its n-session
average pauses; mode: replace (book only) / and (market signal AND book stress) / or (either).
Every monthly start 2000-2025, split by start era, plus one run from 2000 with exposure and key years."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat

E = json.load(open("../system_E.json"))["cfg"]
G = [("System E (market pause)", E),
     ("book fall >10% (replace)", dict(E, book_pause=dict(dd=0.10, mode="replace"))),
     ("book fall >15% (replace)", dict(E, book_pause=dict(dd=0.15, mode="replace"))),
     ("book below 50d avg (replace)", dict(E, book_pause=dict(ma=50, mode="replace"))),
     ("book below 100d avg (replace)", dict(E, book_pause=dict(ma=100, mode="replace"))),
     ("market AND book fall >10%", dict(E, book_pause=dict(dd=0.10, mode="and"))),
     ("market AND book below 50d", dict(E, book_pause=dict(ma=50, mode="and"))),
     ("market AND book below 100d", dict(E, book_pause=dict(ma=100, mode="and"))),
     ("market AND book fall >5%", dict(E, book_pause=dict(dd=0.05, mode="and")))]

if __name__ == "__main__":
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    rows = []
    for lab, c in G:
        T = Y.evaluate(pool, c, Y.starts())
        r = {"label": lab}
        for era, (a, b) in {"2000-11": (2000, 2011), "2012-19": (2012, 2019), "2020-25": (2020, 2025), "all": (2000, 2025)}.items():
            s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
            r.update({f"{era}|beat": s["yrs_beat_SPY_QQQ"], f"{era}|ddok": s["dd_ok"]})
        s = Y.summary(T)
        r.update(cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], medcagr=s["med_cagr"], worst=s["worst_dd"])
        rr = X.run(Y.MM, mat(c), start="2000-01-03", record=True)
        d, y = rr["daily"], M.yearly(rr["curve"])
        r.update(cagr2000=M.cagr(rr["curve"]), dd2000=M.maxdd(rr["curve"]), in_stocks=(d.stocks / d.nav).mean(), paused=d.paused.mean(),
                 **{str(k): y[k] for k in (2002, 2008, 2009, 2019, 2020, 2022, 2023)})
        rows.append(r)
        print(lab, flush=True)
    df = pd.DataFrame(rows); df.to_csv("../results/exp45_bookpause.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print(df.round(3).to_string(index=False))
