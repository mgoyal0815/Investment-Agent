"""Tie luck (owner, 17 Oct 2026). Score systems over tie seeds 1-8 x the 5 review calendars (robust.tscore), plus the
alphabetical tie-break. Day-1 starts every 3rd month 2000-2024 (day 15 with --day 15), runs to Dec 2025.
Systems A-K are scored on the panel they were built on (panel.pkl, S&P 500), L-R on panel_ndx.pkl.
Writes per system: a69_<X>_day<d>.json (headline + provenance), a69_<X>_day<d>_seeds.csv, a69_<X>_day<d>_grid.csv.

    python3 a69_tie_luck.py [--day 15] R Q P ..."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, os, multiprocessing as mp, yardstick as Y, robust as RB

args = sys.argv[1:]
day = 1
if args[:1] == ["--day"]:
    day, args = int(args[1]), args[2:]
NDX = set("LMNOPQR")
pool, cur = None, None
for x in args:
    path = "../data/panel_ndx.pkl" if x in NDX else "../data/panel.pkl"
    out = f"a69_{x}_day{day}"
    if os.path.exists(out + ".json"):
        print("skip", x, "(done)", flush=True); continue
    if path != cur:
        if pool:
            pool.close(); pool.join()
        Y._init(path); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(path,)); cur = path
    cfg = json.load(open(f"../system_{x}.json"))["cfg"]
    h, S, G = RB.tscore(pool, cfg, day=day)
    S.to_csv(out + "_seeds.csv", index=False); G.to_csv(out + "_grid.csv", index=False)
    json.dump(dict(system=x, day=day, step=3, head=h, provenance=RB.provenance(panel=path, system_file=f"system_{x}.json")),
              open(out + ".json", "w"), indent=1)
    print(x, {k: round(v, 3) for k, v in h.items() if k.startswith(("cal ", "alpha cal", "beat med", "worst", "cagr med"))}, flush=True)
