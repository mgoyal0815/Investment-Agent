"""Owner, 16 Oct 2026: do not count 2026. Re-score System O, System N and live v5 with every run stopped at
31 Dec 2025 (starts to Dec 2024, so every start has at least one full year). Day-1 and day-15 starts."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
P = "../data/panel_ndx.pkl"; END = "2025-12-31"
if __name__ == "__main__":
    Y._init(P); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(P,))
    rows = []
    for lab, cfg in (("System O", json.load(open("../system_O.json"))["cfg"]), ("System N", json.load(open("../system_N.json"))["cfg"]), ("live v5", dict(universe="sp500"))):
        for day in (1, 15):
            T = Y.evaluate(pool, cfg, Y.starts(b="2024-12-31", day=day), end=END)
            T.to_csv(f"../audit/a29_{lab.replace(' ', '_')}_d{day}.csv", index=False)
            for era, (a, b) in {"every start": (2000, 2024), "2000-11": (2000, 2011), "2012-19": (2012, 2019), "2020-24": (2020, 2024)}.items():
                s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
                rows.append(dict(system=lab, starts=f"day {day}", era=era, beat=s["yrs_beat_SPY_QQQ"], cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]))
            print(pd.DataFrame(rows).tail(4).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv("../audit/a29_no2026.csv", index=False)
