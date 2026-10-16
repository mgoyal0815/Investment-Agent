"""One-at-a-time retune of System O's never-tuned settings (owner, 16 Oct 2026). New standard: runs stop at
31 Dec 2025, starts to Dec 2024, every second monthly start, all eras scored together (no split periods)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    O = json.load(open("../system_O.json"))["cfg"]; ST = Y.starts(step=2)
    def score(c):
        T = Y.evaluate(pool, c, ST); s = Y.summary(T)
        out = dict(beat=s["yrs_beat_SPY_QQQ"], smh3=s["r3_beat_SMH"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"])
        for era, (a, b) in {"00-11": (2000, 2011), "12-19": (2012, 2019), "20-24": (2020, 2024)}.items():
            out[f"beat {era}"] = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])["yrs_beat_SPY_QQQ"]
        return out
    G = [("System O", O)]
    G += [(f"Add to winners from +{int(g*100)}%", dict(O, add_gain=g)) for g in (0.10, 0.30)]
    G += [(f"Add size {s} slot", dict(O, add_size=s)) for s in (0.25, 1.0)]
    G += [(f"Add cap {c} slots", dict(O, add_cap=c)) for c in (2.0, 4.0)]
    G += [("No adding to winners", dict(O, adds=None))]
    G += [(f"Gold rule {g} slots", dict(O, gold_slots=g)) for g in (0, 2, 6)]
    G += [("Review month-end only", dict(O, stagger=[-1])), ("Review month-end + 6th session", dict(O, stagger=[-1, 5])),
          ("Review month-end + 16th session", dict(O, stagger=[-1, 15])), ("Review in thirds (end, 7th, 14th)", dict(O, stagger=[-1, 6, 13]))]
    G += [("Mid-month buying on", dict(O, intramonth=True))]
    G += [(f"Sell outside top {b}x holdings", dict(O, buffer=b)) for b in (1.3, 1.7)]
    G += [("Pause without credit vote", dict(O, pause_rules=("breadth", "large"))), ("Pause without large-cap vote", dict(O, pause_rules=("breadth", "hyg"))),
          ("Pause without breadth vote", dict(O, pause_rules=("hyg", "large"))), ("Pause needs 2 of 3 votes", dict(O, pause_k=2))]
    G += [(f"Large-cap vote on {n}-day average", dict(O, large_n=n)) for n in (100, 150, 200)]
    G += [(f"Large-cap vote needs {int(x*100)}% below its 50-day", dict(O, large_margin=x)) for x in (0.02, 0.04)]
    rows = []
    for lab, c in G:
        rows.append(dict(test=lab, **score(c))); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv("../audit/a31_o_retune.csv", index=False)
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(3).to_string(index=False))
