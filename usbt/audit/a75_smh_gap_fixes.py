"""Owner, 17 Oct 2026: try (1) a shorter start-up ramp and (2) parking idle money in SMH more often, on System R1.
Calendar-robust (5 review calendars), deterministic R1 tie-break; a change counts only if it beats R1 by more than
R1's tie-luck spread (1.8 pts of calendar years on day-1 starts, a69). --day 15 for the held-back starts."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
R1 = json.load(open("../system_R1.json"))["cfg"]
V = {"R1": {}, "ramp 1 month (no ramp)": dict(ramp=1), "ramp 2 months": dict(ramp=2),
     "park SMH always (outside pauses)": dict(park_smh="always"), "park SMH while above its 200-day": dict(park_smh="trend"),
     "leader lookback 3 months": dict(leader_lb=63), "ramp 1 + park SMH trend": dict(ramp=1, park_smh="trend")}

if __name__ == "__main__":
    args = sys.argv[1:]; day = 1
    if args[:1] == ["--day"]:
        day, args = int(args[1]), args[2:]
    names = args or list(V)
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    rows = []
    for n in names:
        h, _ = RB.rscore(pool, dict(R1, **V[n]), day=day)
        rows.append(dict(test=n, **h))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items() if k in ("test", "cal", "cal_min", "beat", "smh3", "ddok", "cagr", "worst")}, flush=True)
        pd.DataFrame(rows).to_csv(f"../audit/a75_day{day}.csv", index=False)
    json.dump(dict(variants=V, day=day, provenance=RB.provenance(seeds=(), panel=PATH, system_file="system_R1.json")), open(f"../audit/a75_day{day}.json", "w"), indent=1)
