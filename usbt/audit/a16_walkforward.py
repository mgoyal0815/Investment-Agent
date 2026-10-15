"""Validation 1: walk-forward. At each boundary B (2008, 2012, 2016, 2020, 2024) pick, among Systems F-K, the one
with the most years beating SPY and QQQ on monthly starts in [B-8y, B) judged only up to B (no knowledge after B).
Then score every system on monthly starts in [B, B+4y) judged to today. Compare the walk-forward pick with always-K."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
S = {s: json.load(open(f"../system_{s}.json"))["cfg"] for s in "FGHIJK"}
rows = []
for B in (2008, 2012, 2016, 2020, 2024):
    tr = Y.starts(f"{B-8}-01-01", f"{B-1}-12-31", step=2); te = Y.starts(f"{B}-01-01", f"{min(B+3, 2025)}-12-31", step=2)
    for s, c in S.items():
        a = Y.summary(Y.evaluate(pool, c, tr, end=f"{B-1}-12-31"))["yrs_beat_SPY_QQQ"]
        b = Y.summary(Y.evaluate(pool, c, te))["yrs_beat_SPY_QQQ"]
        rows.append(dict(boundary=B, system=s, past_8y=a, next_4y=b)); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a16_walkforward.csv", index=False)
pick = R.loc[R.groupby("boundary").past_8y.idxmax()][["boundary", "system", "past_8y", "next_4y"]].rename(columns={"system": "picked"})
pick["K_next_4y"] = R[R.system == "K"].set_index("boundary").loc[pick.boundary, "next_4y"].values
pick["G_next_4y"] = R[R.system == "G"].set_index("boundary").loc[pick.boundary, "next_4y"].values
pick["best_next_4y"] = R.groupby("boundary").next_4y.max().values
print(R.pivot(index="boundary", columns="system", values="next_4y").round(3).to_string())
print(pick.round(3).to_string(index=False))
print("average next-4y: walk-forward pick %.3f, always K %.3f, always G %.3f" % (pick.next_4y.mean(), pick.K_next_4y.mean(), pick.G_next_4y.mean()))
