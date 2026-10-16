"""Neighbouring settings for the industry cap on System K: caps of 2, 3, 4 per industry, with the industry map built
from 3, 6 (standard) or 12 months of price co-movement (set SECTOR_FILE). Every second monthly start 2000-2025."""
import sys, os; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
K = json.load(open("../system_K.json"))["cfg"]; ST = Y.starts(step=2)
mp_ = os.environ.get("SECTOR_FILE", "../data/sector_idx.npy").split("/")[-1]
label = {"sector_idx.npy": "6 months", "sector_idx_63.npy": "3 months", "sector_idx_252.npy": "12 months"}[mp_]
def score(c):
    s = Y.summary(Y.evaluate(pool, c, ST)); return dict(beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"])
rows = []
if label == "6 months":
    rows.append(dict(map="-", cap="none (System K)", **score(K))); print(rows[-1], flush=True)
for k in (2, 3, 4):
    rows.append(dict(map=label, cap=k, **score(dict(K, max_per_sector=k)))); print(rows[-1], flush=True)
f = "../audit/a18_sector_neighbours.csv"
R = pd.DataFrame(rows); R.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
