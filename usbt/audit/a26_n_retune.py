"""One-at-a-time retune of System N's remaining settings (panel_ndx), every second monthly start 2000-2025."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel_ndx.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
N = json.load(open("../system_N.json"))["cfg"]; ST = Y.starts(step=2)
def score(c):
    T = Y.evaluate(pool, c, ST); out = {}
    for era, (a, b) in {"all": (2000, 2025), "00-11": (2000, 2011), "12-19": (2012, 2019), "20-25": (2020, 2025)}.items():
        out[f"beat {era}"] = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])["yrs_beat_SPY_QQQ"]
    s = Y.summary(T); out.update(ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]); return out
G = [("System N", N)]
G += [(f"ADX >= {a}", dict(N, adx_min=a)) for a in (22, 28)]
G += [(f"Profit lock from +{int(a*100)}%", dict(N, lock_at=a)) for a in (0.4, 0.75)]
G += [(f"Profit lock trail {int(t*100)}%", dict(N, lock_trail=t)) for t in (0.20, 0.30)]
G += [(f"Restart window {d} sessions", dict(N, unpause_days=d)) for d in (35, 50)]
G += [("Restart on Coppock only (no cyclicals/defensives)", dict(N, unpause_on=["coppock"]))]
G += [(f"Crash trigger RSI <= {r}", dict(N, pause_dip_index_rsi=r)) for r in (22, 28)]
G += [(f"Crash hold {h} sessions", dict(N, dip_hold=h)) for h in (14, 30)]
G += [("No index hurdle", dict(N, index_hurdle=False))]
G += [(f"Winners exempt from rank sells from +{int(g*100)}%", dict(N, rank_exempt_gain=g)) for g in (0.75, 1.5)]
G += [("No trend hold", dict(N, trend_hold=False))]
G += [(f"Breadth pause level {int(b*100)}%", dict(N, breadth_level=b)) for b in (0.15, 0.25)]
rows = []
for lab, c in G:
    rows.append(dict(test=lab, **score(c))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a26_n_retune.csv", index=False)
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
