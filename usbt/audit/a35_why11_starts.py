"""Root cause, part 2: the same rules give near-equal calendar years for review sessions 9-13 (a34), so where does the
11th session's yardstick edge come from? Score every monthly start (day 1, 2000-2024, runs to Dec 2025) with the
second review on session 10, 11 and 12 and compare start by start: which starts flip, and why."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    O = json.load(open("../system_O.json"))["cfg"]
    out = {}
    for d in (9, 10, 11):
        T = Y.evaluate(pool, dict(O, stagger=[-1, d]), Y.starts()); T["session"] = d + 1; out[d] = T
        print(d + 1, round(Y.summary(T)["yrs_beat_SPY_QQQ"], 3), flush=True)
    pd.concat(out.values()).to_csv("../audit/a35_why11_starts.csv", index=False)
