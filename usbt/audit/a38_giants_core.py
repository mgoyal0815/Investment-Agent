"""Follow-up to a37 (owner, 16 Oct 2026): the two leads under the calendar-robust yardstick - the giants path
(67.5%) and a permanent QQQ core (65.4% at 33%). Neighbouring settings and combinations, vs System O 62.4%."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    O = json.load(open("../system_O.json"))["cfg"]
    g = lambda s, t: dict(O, giant_slots=s, giant_top=t)
    q = lambda c, w: dict(c, core_w=w, core_mode="fixed", core_etf="QQQ")
    G = [("Giants 1 slot, top 20", g(1, 20)), ("Giants 3 slots, top 20", g(3, 20)),
         ("Giants 2 slots, top 10", g(2, 10)), ("Giants 2 slots, top 30", g(2, 30)),
         ("Giants 2/20 + QQQ core 20%", q(g(2, 20), 0.2)), ("Giants 2/20 + QQQ core 33%", q(g(2, 20), 0.33)),
         ("QQQ core 50%", q(O, 0.5)), ("Giants 2/20 in thirds", dict(g(2, 20), stagger=[-1, 6, 13]))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a38_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a38_giants_core.csv", index=False)
