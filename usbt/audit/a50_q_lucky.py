"""System Q on its original review calendar (month-end + 11th session), single $10,000 account, full yardstick:
every monthly start 2000-2024 (day 1 and the held-back 15th), runs to 31 Dec 2025. For comparison with the
calendar-robust score (70.2%), which averages 5 calendars on every third start."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    rows = []
    for day in (1, 15):
        T = Y.evaluate(pool, Q, Y.starts(day=day)); T.to_csv(f"../audit/a50_q_lucky_d{day}.csv", index=False)
        for era, (a, b) in {"every start": (2000, 2024), "2000-11": (2000, 2011), "2012-19": (2012, 2019), "2020-24": (2020, 2024)}.items():
            s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
            rows.append(dict(starts=f"day {day}", era=era, beat=s["yrs_beat_SPY_QQQ"], cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]))
    R = pd.DataFrame(rows); R.to_csv("../audit/a50_q_lucky.csv", index=False); print(R.round(3).to_string(index=False))
