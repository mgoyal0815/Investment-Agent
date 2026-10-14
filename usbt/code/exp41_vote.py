"""Experiment 41: buy pause by a vote of market indicators (owner's idea): SPY / QQQ / SMH below their
50/100/200-session averages, HYG below its 63-session average (credit stress), TLT above its 63-session
average (flight to safety), and the existing equal-weight large-cap and breadth signals. Pause when at
least k of the n signals agree. System D otherwise. Chosen on starts 2012-2016 (stopping 2019) with the
yardstick score; judged on starts 2000-2008, 2020-2025 and every start; plus calendar 2008/2011/2022/2023/2024."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat
from exp34_yard_search import score

SD = json.load(open("../system_D.json"))["cfg"]
SETS = {"SPY/QQQ/SMH 200d": ("spy200", "qqq200", "smh200"),
        "SPY/QQQ/SMH 200d + HYG": ("spy200", "qqq200", "smh200", "hyg63"),
        "SPY/QQQ/SMH 200d + HYG + TLT": ("spy200", "qqq200", "smh200", "hyg63", "tlt_up"),
        "SPY/QQQ/SMH 100d + HYG": ("spy100", "qqq100", "smh100", "hyg63"),
        "SPY/QQQ/SMH 50d + HYG": ("spy50", "qqq50", "smh50", "hyg63"),
        "SPY/QQQ 200d + HYG + large": ("spy200", "qqq200", "hyg63", "large"),
        "D's signals + SPY/QQQ 200d": ("breadth", "hyg", "large", "spy200", "qqq200"),
        "SPY/QQQ/SMH 200d + HYG + TLT + large": ("spy200", "qqq200", "smh200", "hyg63", "tlt_up", "large")}


def grid():
    g = [("System D (breadth/HYG/large, any 1)", SD)]
    for lab, r in SETS.items():
        for k in range(1, len(r) + 1):
            g.append((f"{lab}, {k} of {len(r)}", dict(SD, pause_rules=r, pause_k=k)))
    return g


if __name__ == "__main__":
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 2), "2019-12-31"),
           "test 2000-08": (Y.starts("2000-01-01", "2008-12-31", 2), "2011-12-30"),
           "test 2020-25": (Y.starts("2020-01-01", "2025-10-31", 1), None), "all starts": (Y.starts(), None)}
    rows = []
    for lab, c in grid():
        r = {"label": lab}
        for p, (st, e) in PER.items():
            s = Y.summary(Y.evaluate(pool, c, st, e))
            r.update({f"{p}|score": score(s), f"{p}|beat": s["yrs_beat_SPY_QQQ"], f"{p}|smh3": s["r3_beat_SMH"], f"{p}|ddok": s["dd_ok"],
                      f"{p}|cagr": s["med_cagr"], f"{p}|worst": s["worst_dd"]})
        rr = X.run(Y.MM, mat(c), start="2000-01-03", record=True)
        yr = M.yearly(rr["curve"])
        r.update({str(y): yr.get(y) for y in (2002, 2008, 2011, 2022, 2023, 2024)})
        r.update(paused=rr["daily"].paused.mean(), cagr2000=M.cagr(rr["curve"]), dd2000=M.maxdd(rr["curve"]))
        rows.append(r)
        print(lab, round(r["train|score"], 3), round(r["all starts|beat"], 3), round(r["all starts|ddok"], 3), flush=True)
        pd.DataFrame(rows).to_csv("../results/exp41_vote.csv", index=False)
