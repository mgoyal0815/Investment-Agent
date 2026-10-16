"""OBV entry threshold around System M (panel_ndx), every second monthly start 2000-2025."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel_ndx.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
M = json.load(open("../system_M.json"))["cfg"]; ST = Y.starts(step=2)
rows = []
for v in (0.25, 0.75, 1.0):
    T = Y.evaluate(pool, dict(M, obv_min=v), ST); out = dict(test=f"OBV >= {v}")
    for era, (a, b) in {"all": (2000, 2025), "00-11": (2000, 2011), "12-19": (2012, 2019), "20-25": (2020, 2025)}.items():
        out[f"beat {era}"] = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])["yrs_beat_SPY_QQQ"]
    s = Y.summary(T); out.update(ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]); rows.append(out); print(out, flush=True)
pd.DataFrame(rows).to_csv("../audit/a24_m_obv.csv", index=False)
