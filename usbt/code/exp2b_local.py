"""Experiment 2b: local search (training window only) around the best exp2 configurations.

Coordinate ascent: change one setting at a time to every value in the search space, keep the
change if it raises the objective, two passes. Run for two objectives:
  "balanced" - runner.objective (beat all three, and a drawdown smaller than every index);
  "beat_all" - the same but with the drawdown penalty halved, to see what the beat-every-year
               goal alone would pick.
Writes results/exp2b_<objective>.csv (the path taken) and exp2b_best.json.
"""
import json
import sys
import pandas as pd
import runner as R
from exp2_search import SPACE_SLOT, SPACE_V5, materialise


def obj(a, kind):
    if kind == "balanced":
        return R.objective(a)
    return a["beat_all"] + 0.25 * a["roll_win"] + 0.1 * a["beat_SPY"] - 0.5 * (1 - a["dd_ok"]) + 0.5 * a["cagr"]


def climb(rn, base, kind, passes=2):
    cur = dict(base)
    a = rn.eval_many([materialise(cur)], "train")[0]
    best = obj(a, kind)
    path = [dict(step="start", obj=best, cfg=json.dumps(cur, default=list), **a)]
    space = SPACE_SLOT if cur.get("sizing") == "slot" else SPACE_V5
    for p in range(passes):
        improved = False
        for k, vals in space.items():
            cands = [dict(cur, **{k: v}) for v in vals if v != cur.get(k)]
            if not cands:
                continue
            res = rn.eval_many([materialise(c) for c in cands], "train")
            for c, r in zip(cands, res):
                o = obj(r, kind)
                if o > best + 1e-6:
                    best, cur, a, improved = o, c, r, True
            path.append(dict(step=f"pass{p} {k}={cur.get(k)}", obj=best, cfg=json.dumps(cur, default=list), **a))
            print(kind, f"pass{p}", k, "->", cur.get(k), round(best, 3), R.fmt("", a), flush=True)
        if not improved:
            break
    return cur, a, path


if __name__ == "__main__":
    top = pd.read_csv("../results/exp2_all.csv")
    rn = R.Runner()
    out = {}
    for kind in ("balanced", "beat_all"):
        top["o2"] = [obj(r, kind) for _, r in top.iterrows()]
        seeds = top.sort_values("o2", ascending=False).head(3)
        best = None
        rows = []
        for n, (_, row) in enumerate(seeds.iterrows()):
            cfg, a, path = climb(rn, json.loads(row["cfg"]), kind)
            for p in path:
                p["seed"] = n
            rows += path
            o = obj(a, kind)
            if best is None or o > best[0]:
                best = (o, cfg, a)
        pd.DataFrame(rows).to_csv(f"../results/exp2b_{kind}.csv", index=False)
        out[kind] = dict(objective=best[0], cfg=best[1], train=best[2])
        json.dump(out, open("../results/exp2b_best.json", "w"), indent=1, default=float)
