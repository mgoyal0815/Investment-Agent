"""A deterministic tie-break that uses information (owner, 17 Oct 2026: "find a deterministic way that maximizes the
returns"). R's score is a percentile of the average of two percentiles, so many stocks share a score. Each candidate
breaks those ties on a real quantity first (alphabetical only after it). Controls: alphabetical and reverse
alphabetical (no information). A key only counts if it beats R's tie-seed range (a69_R_day1.json), not just the
alphabetical draw. Calendar-robust, day-1 starts every 3rd month 2000-2024 (or --day 15), runs to Dec 2025."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
KEYS = ["rev_alpha", "rs126", "dist200", "zsum", "m12", "hi52", "medv", "lowvol"]

if __name__ == "__main__":
    args = sys.argv[1:]; day = 1
    if args[:1] == ["--day"]:
        day, args = int(args[1]), args[2:]
    keys = args or KEYS
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    R = json.load(open("../system_R.json"))["cfg"]
    rows = []
    for k in keys:
        h, G = RB.rscore(pool, dict(R, tie_key=k), day=day)
        G.insert(0, "tie_key", k); G.to_csv(f"a71_grid_day{day}.csv", mode="a", header=not rows, index=False)
        rows.append(dict(tie_key=k, **h))
        print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in rows[-1].items() if a in ("tie_key", "cal", "cal_min", "cal_max", "beat", "smh3", "ddok", "cagr", "worst")}, flush=True)
        pd.DataFrame(rows).to_csv(f"a71_tie_keys_day{day}.csv", index=False)
    json.dump(dict(keys=keys, day=day, step=3, provenance=RB.provenance(seeds=(), panel=PATH, system_file="system_R.json")),
              open(f"a71_tie_keys_day{day}.json", "w"), indent=1)
