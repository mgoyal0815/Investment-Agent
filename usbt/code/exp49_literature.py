"""Experiment 49 (literature, 15 Oct 2026), on System E + index hurdle, clean signals, panel_idx:
 - industry momentum (Moskowitz & Grinblatt 1999): buy only inside the top-k industries (sector ETFs, stocks
   assigned point-in-time by correlation, sectors.py), or blend the stock score with its industry's momentum;
 - momentum-crash guard (Daniel & Moskowitz 2016): after a 2-year market decline, in a rebound (SPY above its
   50-day), rank by 3-month recovery instead of 6-month momentum;
 - risk-managed momentum (Barroso & Santa-Clara 2015): scale the stock book by its own volatility;
 - intermediate momentum (Novy-Marx 2012) and 12-1 momentum blended into the rank.
Every monthly start 2000-2025, split into 2012-16 'choosing' starts and the rest."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat

E = json.load(open("../system_E.json"))["cfg"]
EH = dict(E, index_hurdle=True)
G = [("E + hurdle", EH),
     ("+ top-3 industries", dict(EH, sector_top_k=3)), ("+ top-4 industries", dict(EH, sector_top_k=4)),
     ("+ top-5 industries", dict(EH, sector_top_k=5)), ("+ top-6 industries", dict(EH, sector_top_k=6)),
     ("+ top-5 industries (3-month)", dict(EH, sector_top_k=5, sector_lb=63)),
     ("+ industry blend 0.25", dict(EH, sector_w=0.25)),
     ("+ crash guard", dict(EH, dm_guard=True)),
     ("+ own-vol target 30% hard", dict(EH, vol_target=0.3, vol_hard=True)),
     ("+ own-vol target 40% hard", dict(EH, vol_target=0.4, vol_hard=True)),
     ("+ 12-1 momentum blend 0.5", dict(EH, score_mix={"mom12_1": 0.5})),
     ("+ 12-7 momentum blend 0.25", dict(EH, score_mix={"mom12_7": 0.25})),
     ("+ top-5 industries + crash guard", dict(EH, sector_top_k=5, dm_guard=True)),
     ("+ top-4 industries + crash guard", dict(EH, sector_top_k=4, dm_guard=True))]

if __name__ == "__main__":
    PATH = "../data/panel_idx.pkl"
    Y._init(PATH)
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    rows = []
    for lab, c in G:
        T = Y.evaluate(pool, c, Y.starts())
        r = {"label": lab}
        for era, (a, b) in {"choose 2012-16": (2012, 2016), "2000-11": (2000, 2011), "2017-19": (2017, 2019), "2020-25": (2020, 2025), "all": (2000, 2025)}.items():
            s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
            r.update({f"{era}|beat": s["yrs_beat_SPY_QQQ"], f"{era}|ddok": s["dd_ok"]})
        s = Y.summary(T)
        r.update(cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], medcagr=s["med_cagr"], worst=s["worst_dd"])
        rr = X.run(Y.MM, mat(c), start="2000-01-03")
        y = M.yearly(rr["curve"])
        r.update(cagr2000=M.cagr(rr["curve"]), dd2000=M.maxdd(rr["curve"]), **{str(k): y[k] for k in (2003, 2009, 2019, 2020, 2008, 2022, 2023)})
        rows.append(r); print(lab, flush=True)
        pd.DataFrame(rows).to_csv("../results/exp49_literature.csv", index=False)
    print("DONE", flush=True)
