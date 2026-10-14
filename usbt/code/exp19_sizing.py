"""Experiment 19: size the winners. Coordinate ascent around System A on 2012-2019
(runner.objective2), two passes, over: the three v5 sizing budgets separately (base, risk, vol),
adds to extended winners (framework 10.1 add_red_hi), entry zone gate, profit-lock settings, rank
exit width, universe size, route cutoffs, leader look-back, breadth line. Finalist tested on
2020-2026 and 2000-2011, plus one-change-at-a-time from System A on all windows.
Writes results/exp19_*."""
import json
import pandas as pd
import runner as R
from exp2_search import materialise

A = json.load(open("../system_A.json"))["cfg"]
SPACE = dict(base_mult=[1.0, 1.5, 2.0], risk_mult=[1.0, 1.5, 2.0, 3.0], vol_mult=[1.0, 1.5, 2.0, 3.0],
             add_red_hi=[None, 2.0, 3.0, 5.0], zone_gate=[False, True], lock_at=[0.2, 0.3, 0.4],
             lock_trail=[0.2, 0.25, 0.3], N=[20, 30, 40], universe_n=[300, 500, 800],
             cont_cut=[0.85, 0.88, 0.92], rec_cut=[0.83, 0.90, 0.95], leader_lb=[63, 126, 252],
             breadth_level=[0.2, 0.3])

if __name__ == "__main__":
    rn = R.Runner()
    # one change at a time, all windows (transparent view)
    single = [("System A", A)]
    for k, vals in SPACE.items():
        for v in vals:
            if v != A.get(k, {"base_mult": 1.0, "risk_mult": 1.0, "vol_mult": 1.0, "add_red_hi": None}.get(k)):
                single.append((f"{k}={v}", dict(A, **{k: v})))
    res = {p: rn.eval_many([materialise(c) for _, c in single], p) for p in ("train", "test", "early")}
    rows = []
    for n, (lab, c) in enumerate(single):
        r = {"label": lab}
        for p in res:
            a = res[p][n]
            r.update({f"{p}_obj2": R.objective2(a), f"{p}_cagr": a["cagr"], f"{p}_dd": a["maxdd"], f"{p}_beat": a["beat_all"]})
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("train_obj2", ascending=False)
    df.to_csv("../results/exp19_single.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["label"] + [f"{p}_{x}" for p in ("train", "test", "early") for x in ("cagr", "dd", "beat")]
    print("ONE CHANGE AT A TIME\n" + df[cols].round(3).to_string(index=False), flush=True)
    # coordinate ascent on training only
    cur = dict(A)
    best = R.objective2(rn.eval_many([materialise(cur)], "train")[0])
    for p in range(2):
        imp = False
        for k, vals in SPACE.items():
            cands = [dict(cur, **{k: v}) for v in vals if v != cur.get(k)]
            for c, a in zip(cands, rn.eval_many([materialise(c) for c in cands], "train")):
                if R.objective2(a) > best + 1e-6:
                    best, cur, imp = R.objective2(a), c, True
            print(f"pass{p} {k} -> {cur.get(k)} {best:+.3f}", flush=True)
        if not imp:
            break
    json.dump(cur, open("../results/exp19_pick.json", "w"), default=list, indent=1)
    fin = [("System A", A), ("exp19 pick", cur)]
    for p in ("train", "test", "early"):
        for (lab, c), a in zip(fin, rn.eval_many([materialise(c) for _, c in fin], p)):
            print(p, f"{R.objective2(a):+.3f}", R.fmt(lab, a), flush=True)
