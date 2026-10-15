"""Neighbouring settings for 'RSI 25 or a steep fall (SPY X% below its N-day high)' on System J.
Every second monthly start 2000-2025, standard panel."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
J = json.load(open("../system_J.json"))["cfg"]; ST = Y.starts(step=2)
def score(c):
    s = Y.summary(Y.evaluate(pool, c, ST)); return dict(beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"])
rows = [dict(test="System J", **score(J))]; print(rows[-1], flush=True)
for p in (0.12, 0.15, 0.18):
    for n in (15, 20, 30):
        rows.append(dict(test=f"RSI 25 or SPY {int(p*100)}% below its {n}-day high", **score(dict(J, dip_fall=(p, n))))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a11_fall_neighbours.csv", index=False); print(R.round(3).to_string(index=False))
