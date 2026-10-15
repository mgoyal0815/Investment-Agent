"""Audit of System K, part 2. (a) Placebo for K's crash triggers: the 21-session index hold during pauses starts on
random dates, as many as the real trigger days (SPY RSI <= 25 or SPY 18% below its 20-day high), 16 seeds.
(b) The cyclicals/defensives restart inside K at 75/90/125/150-day averages (the H audit found it fragile).
Every second monthly start 2000-2025, standard panel."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, numpy as np, pandas as pd, yardstick as Y
PATH = "../data/panel.pkl"; Y._init(PATH); m = Y.MM
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
K = json.load(open("../system_K.json"))["cfg"]; ST = Y.starts(step=2)
def score(c):
    s = Y.summary(Y.evaluate(pool, c, ST)); return dict(beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"])
rsi = m.etf_rsi("SPY"); sp = m.E["SPY"]["c"]
real = [i for i in range(260, m.T) if rsi[i] <= 25 or sp[i] <= np.nanmax(sp[i - 20:i + 1]) * 0.82]
print("real trigger days:", len(real), flush=True)
rows = [dict(test="System K", **score(K)), dict(test="K without any crash trigger (I + hurdle only)", **score(dict(K, pause_dip_index_rsi=None, dip_fall=None)))]
print(rows, flush=True)
for seed in range(16):
    rd = set(np.random.default_rng(seed).choice(np.arange(260, m.T), len(real), replace=False).tolist())
    rows.append(dict(test=f"placebo seed {seed}", **score(dict(K, _dip_rand=rd)))); print(rows[-1], flush=True)
for n in (75, 90, 125, 150):
    rows.append(dict(test=f"cyc/def restart at {n}-day average", **score(dict(K, unpause_on=["coppock", f"ra_cd{n}"])))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a13_k_placebo.csv", index=False); print(R.round(3).to_string(index=False))
pl = R[R.test.str.startswith("placebo")]; k = R.iloc[0]; base = R.iloc[1]
print(f"\nK {k.beat:.3f}; no trigger {base.beat:.3f}; placebo mean {pl.beat.mean():.3f}, best {pl.beat.max():.3f}; K beats {int((pl.beat < k.beat).sum())} of 16")
