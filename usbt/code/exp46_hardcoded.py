"""Experiment 46: rules that were hard-coded or inherited and never questioned under the owner's yardstick
(code review, 15 Oct 2026): the pause tightening the sell rank to the top 10, gold taking 4 of 10 slots,
re-entry waiting for the next decision day after a pause, where idle money is parked, decision cadence,
the 3-month ramp, the trend gate on new buys. One change at a time on System E; every monthly start."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat

E = json.load(open("../system_E.json"))["cfg"]
G = [("System E", E),
     ("pause does NOT tighten the sell rank", dict(E, pause_tight_exit=False)),
     ("no gold slots", dict(E, gold_slots=0)),
     ("2 gold slots", dict(E, gold_slots=2)),
     ("re-enter the day a pause ends", dict(E, reenter_fast=True)),
     ("idle money always in leading index", dict(E, park="leader")),
     ("idle money in SPY", dict(E, park="SPY")),
     ("idle money in QQQ", dict(E, park="QQQ")),
     ("no 3-month ramp", dict(E, ramp=0)),
     ("weekly decisions, no stagger", {**{k: v for k, v in E.items() if k != "stagger"}, "cadence": "weekly"}),
     ("4 tranches (sessions last,5,10,15)", dict(E, stagger=[-1, 5, 10, 15])),
     ("no trend gate on new buys", dict(E, trend=False)),
     ("no gold + re-enter fast + no tight exit", dict(E, gold_slots=0, reenter_fast=True, pause_tight_exit=False))]

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
        r.update(cagr2000=M.cagr(rr["curve"]), dd2000=M.maxdd(rr["curve"]), in_stocks=(d.stocks / d.nav).mean(),
                 **{str(k): y[k] for k in (2008, 2009, 2019, 2020, 2022, 2023)})
        rows.append(r); print(lab, flush=True)
        pd.DataFrame(rows).to_csv("../results/exp46_hardcoded.csv", index=False)
    pd.set_option("display.width", 320, "display.max_columns", 40)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
