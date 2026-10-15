"""Neighbouring settings for 'hold the leading index after SPY RSI <= r during a pause' on System I.
Every second monthly start 2000-2025, standard panel."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
I = json.load(open("../system_I.json"))["cfg"]; ST = Y.starts(step=2)
def score(c):
    s = Y.summary(Y.evaluate(pool, c, ST)); return dict(beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"])
rows = [dict(test="System I", **score(I))]; print(rows[-1], flush=True)
for r in (20, 22, 25, 27, 30):
    for h in (14, 21, 30):
        rows.append(dict(test=f"SPY RSI <= {r}, hold {h}", **score(dict(I, pause_dip_index_rsi=r, dip_hold=h)))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a7_dip_neighbours.csv", index=False); print(R.round(3).to_string(index=False))
