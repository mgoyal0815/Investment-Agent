"""Audit 6: the same neighbouring-settings test for System G that H got (a4), on the standard (corrected) panel.
Every second monthly start 2000-2025."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
G = json.load(open("../system_G.json"))["cfg"]; H = json.load(open("../system_H.json"))["cfg"]
ST = Y.starts(step=2)
def score(c):
    s = Y.summary(Y.evaluate(pool, c, ST)); return dict(beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"])
tests = [("System G", G), ("System H", H)]
tests += [(f"G, Coppock window {d} sessions", dict(G, unpause_days=d)) for d in (14, 30, 42)]
tests += [(f"G, OBV trend >= {v}", dict(G, obv_min=v)) for v in (-0.5, 0.5, 1.0)]
tests += [("G without OBV filter", {k: v for k, v in G.items() if k != "obv_min"})]
tests += [("G without Coppock restart (System F + OBV)", {k: v for k, v in G.items() if k not in ("unpause_on", "unpause_days")})]
rows = []
for lab, c in tests:
    rows.append(dict(test=lab, **score(c))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a6_g_neighbours.csv", index=False); print(R.round(3).to_string(index=False))
