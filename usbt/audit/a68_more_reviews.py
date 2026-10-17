"""India round 11 on System R (owner, 17 Oct 2026): India gained from a third review of the whole book on the 4th-7th
session (and lost with four reviews). R keeps its two half-book reviews (month end + 11th session, shifted -6..+6 by the
robust yardstick); this adds whole-book review days (0-indexed session of the month, fixed across calendars).
Calendar-robust, day-1 starts every 3rd month 2000-2024, runs to Dec 2025."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
R = json.load(open("../system_R.json"))["cfg"]
V = {"R (2 half reviews)": R}
for k in (2, 3, 4, 5, 6, 7):
    V[f"R + whole review session {k + 1}"] = dict(R, extra_reviews=[k])
V["R + whole review session 16"] = dict(R, extra_reviews=[15])
V["R + whole reviews sessions 5 and 16 (4/month)"] = dict(R, extra_reviews=[4, 15])

if __name__ == "__main__":
    day = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    only = sys.argv[2:] or list(V)
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    rows = []
    for name in only:
        h, _ = RB.rscore(pool, V[name], day=day)
        rows.append(dict(test=name, starts=f"day {day}", **h))
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv(f"a68_more_reviews_day{day}.csv", index=False)
