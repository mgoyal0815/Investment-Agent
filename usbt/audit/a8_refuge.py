"""Is it gold, or any non-cash refuge? J with the pause money in SHY (J), GLD, GLD only in its uptrend, TLT, IEF.
Every second monthly start 2000-2025, standard panel. Also gold-only-after-2012 starts, where gold was flat to 2019."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
J = json.load(open("../system_J.json"))["cfg"]; ST = Y.starts(step=2)
rows = []
for lab, c in [("J (Treasuries, SHY)", J), ("gold (GLD)", dict(J, pause_park="GLD")), ("gold in uptrend only", dict(J, pause_park="GLD_trend")),
               ("long Treasuries (TLT)", dict(J, pause_park="TLT")), ("7-10y Treasuries (IEF)", dict(J, pause_park="IEF"))]:
    T = Y.evaluate(pool, c, ST)
    for era, (a, b) in {"all": (2000, 2025), "2000-11": (2000, 2011), "2012-19": (2012, 2019), "2020-25": (2020, 2025)}.items():
        s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
        rows.append(dict(refuge=lab, era=era, beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]))
    print(rows[-4], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a8_refuge.csv", index=False)
print(R.pivot_table(index="refuge", columns="era", values="beat").round(3).to_string()); print(R[R.era == "all"].round(3).to_string(index=False))
