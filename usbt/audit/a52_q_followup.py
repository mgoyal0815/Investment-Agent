"""Follow-ups to a51 (owner, 16 Oct 2026), calendar-robust yardstick, Q = 70.2%: no rank sells in deep market falls at
neighbouring crash levels, and sector ETFs competing for slots (the a51 run was a no-op: engine fix)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [(f"No rank sells while SPY is {int(x*100)}%+ below its high", dict(Q, crash_no_rank=x)) for x in (0.15, 0.25)]
    G += [("Sector ETFs compete for slots like stocks (fixed)", dict(Q, etf_in_universe=True))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a52_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a52_q_followup.csv", index=False)
