"""System Q as five sub-accounts (owner, 16 Oct 2026): $10,000 split into five $2,000 accounts, each running Q with its
second review on a different session (5th, 8th, 11th, 14th, 17th; month-end review in all). No rebalancing between
them; whole shares and costs in each. The account is the sum. Scored from every monthly start 2000-2024 (day 1, and the
held-back 15th), runs to 31 Dec 2025, with the standard yardstick. Each sub-account is also scored alone for comparison."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, engine as X, yardstick as Y, robust as RB
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"


def job(a):
    cfg, s = a
    curves = [X.run(Y.MM, mat(dict(c, capital=2000.0)), start=str(s.date()), end=Y.STD_END)["curve"] for c in RB.calendars(cfg)]
    out = {"ensemble": Y.judge(sum(curves), Y.MM)}
    for k, cv in enumerate(curves):
        out[f"cal{k}"] = Y.judge(cv, Y.MM)
    return out


if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    rows = []
    for day in (1, 15):
        res = pool.map(job, [(Q, s) for s in Y.starts(day=day)], chunksize=2)
        for key in ["ensemble"] + [f"cal{k}" for k in range(5)]:
            T = pd.DataFrame([r[key] for r in res]); T.to_csv(f"../audit/a49_{key}_d{day}.csv", index=False)
            s = Y.summary(T); e = {era: Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])["yrs_beat_SPY_QQQ"]
                                   for era, (a, b) in {"00-11": (2000, 2011), "12-19": (2012, 2019), "20-24": (2020, 2024)}.items()}
            rows.append(dict(account=key, starts=f"day {day}", beat=s["yrs_beat_SPY_QQQ"], cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"],
                             ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"], **{f"beat {k}": v for k, v in e.items()}))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv("../audit/a49_ensemble.csv", index=False)
