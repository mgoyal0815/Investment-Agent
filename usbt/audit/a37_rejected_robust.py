"""Step 2 (owner, 16 Oct 2026): the strongest previously rejected ideas, plus a fix aimed at the root cause
(score smoothing), re-scored against System O's honest baseline (62.4%) with the calendar-robust yardstick."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    O = json.load(open("../system_O.json"))["cfg"]
    G = [("Rank on 5-session average score", dict(O, score_smooth=5)),
         ("Rank on 10-session average score", dict(O, score_smooth=10)),
         ("Permanent QQQ core 20%", dict(O, core_w=0.2, core_mode="fixed", core_etf="QQQ")),
         ("Permanent QQQ core 33%", dict(O, core_w=0.33, core_mode="fixed", core_etf="QQQ")),
         ("Leading-index core 25% while strong", dict(O, core_w=0.25)),
         ("Giants path: 2 slots for top-20 giants", dict(O, giant_slots=2, giant_top=20)),
         ("9 slots in thirds + 5-session average score", dict(O, stagger=[-1, 6, 13], score_smooth=5))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a37_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a37_rejected_robust.csv", index=False)
