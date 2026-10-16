"""System R, last structural ideas (owner, 16 Oct 2026), calendar-robust yardstick, R = 71.3%: fill empty slots every
Friday (no sells outside reviews), and the duplicate-slot fix (giants pick no longer also takes a normal slot)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    R = json.load(open("../system_R.json"))["cfg"]
    G = [("R + duplicate-slot fix", dict(R, dedupe=True)), ("R + weekly fill of empty slots", dict(R, fill_weekly=True)),
         ("R + both", dict(R, fill_weekly=True, dedupe=True))]
    heads = []
    for lab, c in G:
        h, Rr = RB.rscore(pool, c); Rr.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        Rr.to_csv("../audit/a57_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a57_r_fill.csv", index=False)
