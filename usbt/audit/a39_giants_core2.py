"""Round 3 (owner, 16 Oct 2026): combine the best giants setting (1 slot, top 20) with a QQQ core, and test a core that
is held only while QQQ is above its 200-day average (core_trend), to cut the -67/-70% worst falls of the a38 combos."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    O = json.load(open("../system_O.json"))["cfg"]
    g = lambda s: dict(O, giant_slots=s, giant_top=20)
    q = lambda c, w, t=False: dict(c, core_w=w, core_mode="fixed", core_etf="QQQ", core_trend=t)
    G = [("Giants 1 + QQQ core 20%", q(g(1), 0.2)), ("Giants 1 + QQQ core 33%", q(g(1), 0.33)),
         ("Giants 1 + QQQ core 20% above 200d", q(g(1), 0.2, True)), ("Giants 1 + QQQ core 33% above 200d", q(g(1), 0.33, True)),
         ("Giants 2 + QQQ core 33% above 200d", q(g(2), 0.33, True))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a39_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a39_giants_core2.csv", index=False)
