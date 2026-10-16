"""Structure grid under the calendar-robust yardstick (robust.py): more stocks per review and more review dates,
to cut the calendar luck found in a32. Base: System O. Gold rule kept at the same share of slots."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    O = json.load(open("../system_O.json"))["cfg"]
    def v(n, st): return dict(O, N=n, stagger=st, gold_slots=round(n * 4 / 9))
    G = [("System O (9 slots, 2 halves)", O),
         ("10 slots, 2 halves", v(10, [-1, 10])), ("12 slots, 2 halves", v(12, [-1, 10])),
         ("9 slots, 3 thirds", v(9, [-1, 6, 13])), ("12 slots, 3 thirds", v(12, [-1, 6, 13])),
         ("12 slots, 4 quarters", v(12, [-1, 4, 9, 14])), ("16 slots, 4 quarters", v(16, [-1, 4, 9, 14])),
         ("8 slots, 4 quarters", v(8, [-1, 4, 9, 14])), ("15 slots, 3 thirds", v(15, [-1, 6, 13]))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); h = dict(test=lab, **h); heads.append(h)
        R.to_csv("../audit/a33_structure_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in h.items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a33_structure.csv", index=False)
