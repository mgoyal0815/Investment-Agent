"""Honest retune of System Q (owner, 16 Oct 2026). Almost every Q setting was tuned on one review calendar (the 11th
session). Each is moved one step either way and re-scored with the calendar-robust yardstick. Q = 70.2%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [(f"{n} slots", dict(Q, N=n)) for n in (8, 10)]
    G += [(f"ADX >= {a}", dict(Q, adx_min=a)) for a in (20, 30)]
    G += [(f"OBV trend >= {o}", dict(Q, obv_min=o)) for o in (0.0, 1.0)]
    G += [(f"Profit lock from +{int(x*100)}%", dict(Q, lock_at=x)) for x in (0.4, 0.75)]
    G += [(f"Profit lock trail {int(x*100)}%", dict(Q, lock_trail=x)) for x in (0.2, 0.3)]
    G += [(f"Winners exempt from +{int(x*100)}%", dict(Q, rank_exempt_gain=x)) for x in (0.5, 1.0)]
    G += [(f"Restart window {d} sessions", dict(Q, unpause_days=d)) for d in (30, 60)]
    G += [(f"Sell outside top {b}x", dict(Q, buffer=b)) for b in (1.3, 1.7)]
    G += [(f"Add to winners from +{int(x*100)}%", dict(Q, add_gain=x)) for x in (0.15, 0.25)]
    G += [("No index hurdle", dict(Q, index_hurdle=False))]
    G += [("Top-20 giants also skip ADX/volume gates in normal slots", dict(Q, giant_gate_exempt=20))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a45_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a45_q_retune.csv", index=False)
