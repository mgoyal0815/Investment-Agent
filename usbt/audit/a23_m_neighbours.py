"""Neighbouring settings around System M on panel_ndx, every second monthly start 2000-2025."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel_ndx.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
M = json.load(open("../system_M.json"))["cfg"]; ST = Y.starts(step=2)
def score(c):
    T = Y.evaluate(pool, c, ST); out = {}
    for era, (a, b) in {"all": (2000, 2025), "00-11": (2000, 2011), "12-19": (2012, 2019), "20-25": (2020, 2025)}.items():
        s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)]); out[f"beat {era}"] = s["yrs_beat_SPY_QQQ"]
    s = Y.summary(T); out.update(ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]); return out
rows = []
for n in (8, 9, 10):
    for b in (1.3, 1.5, 1.7):
        rows.append(dict(test=f"{n} holdings, buffer {b}" + (" (System M)" if (n, b) == (9, 1.5) else ""), **score(dict(M, N=n, buffer=b)))); print(rows[-1], flush=True)
for lab, c in [("ADX >= 20", dict(M, adx_min=20)), ("ADX >= 30", dict(M, adx_min=30)), ("OBV >= -0.5", dict(M, obv_min=-0.5)), ("OBV >= 0.5", dict(M, obv_min=0.5))]:
    rows.append(dict(test=lab, **score(c))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a23_m_neighbours.csv", index=False); print(R.round(3).to_string(index=False))
