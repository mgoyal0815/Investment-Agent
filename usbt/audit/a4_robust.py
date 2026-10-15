"""Audit 4: is H's gain over G real or a product of searching many variants?
(a) placebo: replace the cyclicals/defensives restart with restarts on random dates, same count (176) and same
    42-session window, 16 seeds. If random dates do as well, the signal adds nothing beyond 'pause less'.
(b) neighbourhood: settings around H's (average 75-150 days, window 30-63 sessions). A real effect is a plateau,
    a fitted one is a lone spike.  Judged on every second monthly start 2000-2025 (155 starts)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, numpy as np, pandas as pd, yardstick as Y
PATH = "../data/panel.pkl"; Y._init(PATH); m = Y.MM
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
G = json.load(open("../system_G.json"))["cfg"]; H = json.load(open("../system_H.json"))["cfg"]
ST = Y.starts(step=2)
def score(c):
    T = Y.evaluate(pool, c, ST); s = Y.summary(T)
    return dict(beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"])
rows = [dict(test="System G", **score(G)), dict(test="System H", **score(H))]
print(rows[0], rows[1], flush=True)
for seed in range(16):
    rng = np.random.default_rng(seed)
    days = set(rng.choice(np.arange(260, m.T), 176, replace=False).tolist())
    c = dict(H, unpause_on=["coppock", "random"], _rand_days=days)
    rows.append(dict(test=f"placebo seed {seed}", **score(c))); print(rows[-1], flush=True)
for n in (75, 90, 125, 150):
    rows.append(dict(test=f"cyc/def {n}d, 42 sessions", **score(dict(H, unpause_on=["coppock", f"ra_cd{n}"])))); print(rows[-1], flush=True)
for d in (30, 35, 50, 63):
    rows.append(dict(test=f"cyc/def 100d, {d} sessions", **score(dict(H, unpause_days=d)))); print(rows[-1], flush=True)
rows.append(dict(test="Coppock only, 42 sessions (no cyc/def)", **score(dict(H, unpause_on=["coppock"])))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a4_robust.csv", index=False)
pl = R[R.test.str.startswith("placebo")]
h = R[R.test == "System H"].iloc[0]
print(R.round(3).to_string(index=False))
print(f"\nplacebo: beat mean {pl.beat.mean():.3f}, best {pl.beat.max():.3f}; H {h.beat:.3f}; H beats {int((pl.beat < h.beat).sum())} of {len(pl)} placebos")
