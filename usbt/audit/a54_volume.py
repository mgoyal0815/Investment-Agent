"""Volume strategies on System Q (owner, 16 Oct 2026), calendar-robust yardstick, Q = 70.2%. Features: volfeat.py."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, os, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB, volfeat
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    P = pd.read_pickle(PATH); T, N = len(P["dates"]), len(P["syms"]); del P
    if not os.path.exists(f"../data/volf_{T}x{N}/ftd.npy"):
        volfeat.build(PATH, "../data/raw", f"../data/volf_{T}x{N}")
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [("Buy only after a pocket pivot (last 10 sessions)", dict(Q, pocket_pivot=True)),
         ("Buy only in a volume dry-up (10d vol <= 0.8x 50d)", dict(Q, vdry_max=0.8)),
         ("Buy only if 10d vol <= 50d vol", dict(Q, vdry_max=1.0)),
         ("Distribution days: pause at 6+ in 25 sessions", dict(Q, pause_rules=("breadth", "hyg", "large", "distday"), dd_n=6)),
         ("Distribution days: pause at 8+ in 25 sessions", dict(Q, pause_rules=("breadth", "hyg", "large", "distday"), dd_n=8)),
         ("Follow-through day restarts buying", dict(Q, unpause_on=Q["unpause_on"] + ["ftd"])),
         ("Climax run + stall day: sell half", dict(Q, climax_exit="half")),
         ("Climax run + stall day: sell all", dict(Q, climax_exit="all"))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a54_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a54_volume.csv", index=False)
