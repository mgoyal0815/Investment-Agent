"""Experiment 40: why System D missed 2023-24 and fixes for it. Diagnosis: the breadth buy pause fired in a
narrow market (half of the days in 2023, a third in mid 2024) while the mega caps rose, so half the book
sat in Treasuries (SHY) and only 5-7 of 10 slots were filled. Fixes tested: park in the leading index
during a pause unless SPY is below its 200-day (safe_when=b200); pause only in a real bear market
(pause_needs_bear); two signals must agree (pause_mode=two); drop the breadth signal; a minimum hold.
Judged on the owner's yardstick on every period, plus the 2023 and 2024 calendar years."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat
from exp34_yard_search import score

SD = json.load(open("../system_D.json"))["cfg"]
G = [("System D", SD),
     ("D, idle cash in leading index unless SPY<200d", dict(SD, safe_when="b200")),
     ("D, pause only when SPY<200d", dict(SD, pause_needs_bear=True)),
     ("D, pause needs two signals", dict(SD, pause_mode="two")),
     ("D, no breadth signal", dict(SD, pause_rules=("hyg", "large"))),
     ("D, no breadth + idle cash in index", dict(SD, pause_rules=("hyg", "large"), safe_when="b200")),
     ("D, pause only SPY<200d + idle in index", dict(SD, pause_needs_bear=True, safe_when="b200")),
     ("D, breadth line 10%", dict(SD, breadth_level=0.1)),
     ("D, min hold 42 sessions", dict(SD, rank_min_hold=42))]

if __name__ == "__main__":
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 2), "2019-12-31"),
           "test 2000-08": (Y.starts("2000-01-01", "2008-12-31", 2), "2011-12-30"),
           "test 2020-25": (Y.starts("2020-01-01", "2025-10-31", 1), None), "all starts": (Y.starts(), None)}
    rows = []
    for lab, c in G:
        r = {"label": lab}
        for p, (st, e) in PER.items():
            s = Y.summary(Y.evaluate(pool, c, st, e))
            r.update({f"{p}|beat": s["yrs_beat_SPY_QQQ"], f"{p}|ddok": s["dd_ok"], f"{p}|cagr": s["med_cagr"], f"{p}|worst": s["worst_dd"],
                      f"{p}|score": score(s)})
        cv = X.run(Y.MM, mat(c), start="2000-01-03")["curve"]
        yr = M.yearly(cv)
        r.update({"2023": yr.get(2023), "2024": yr.get(2024), "2025": yr.get(2025), "2011": yr.get(2011), "cagr_2000": M.cagr(cv), "dd_2000": M.maxdd(cv)})
        rows.append(r)
        print(lab, {k: round(v, 3) for k, v in r.items() if k != "label" and v is not None}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("../results/exp40_narrow.csv", index=False)
