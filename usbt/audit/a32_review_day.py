"""Robustness check (16 Oct 2026): System O with its second review moved to every session 4-19 of the month
(month-end review kept). O uses the 11th session (index 10), inherited from India v8 and never tuned. If O's score
holds only near the 11th, the tuned settings fit one trade path. Standard scoring: runs to 31 Dec 2025, every
second monthly start, all eras together."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    O = json.load(open("../system_O.json"))["cfg"]; ST = Y.starts(step=2)
    rows = []
    for d in range(3, 19):
        T = Y.evaluate(pool, dict(O, stagger=[-1, d]), ST); s = Y.summary(T)
        rows.append(dict(second_review_session=d + 1, beat=s["yrs_beat_SPY_QQQ"], smh3=s["r3_beat_SMH"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv("../audit/a32_review_day.csv", index=False)
