"""Experiment 6: Candidate 3, with and without the index ETF for idle cash, crossed with
  option 3  fewer holdings      max_pos None / 12 / 8, size_mult 1 / 1.5 / 2 (bigger positions)
  option 4  faster leader rule  leader_lb 126 / 63 / 21 sessions (only with an index ETF)
  option 5  exit speed          rank exit outside top 1.5 x N with N 20 / 30, hold floor 44 / 88
Index ETF: none, "leader" (always the leading index), "leader_safe" (SHY while the buy pause is on).
Chosen on 2012-2019 (five starts, runner.objective), tested on 2020-2026 and 2000-2011.
Scoring is calendar years beating SPY, QQQ and SMH, plus worst fall versus the indexes.
Writes results/exp6_train.csv, exp6_final.csv, exp6_years.csv."""
import itertools
import json
import pandas as pd
import engine as X, metrics as M, runner as R
from exp2_search import materialise

C3 = json.loads(pd.read_csv("../results/exp2_top.csv").iloc[2]["cfg"])


def grid():
    out = []
    for park, mp, sm, n, mh in itertools.product([None, "leader", "leader_safe"], [None, 12, 8], [1.0, 1.5, 2.0],
                                                 [20, 30], [44, 88]):
        for lb in ([126] if park is None else [126, 63, 21]):
            out.append(dict(C3, park=park, max_pos=mp, size_mult=sm, N=n, min_hold=mh, leader_lb=lb))
    return out


def label(c):
    p = {None: "no index", "leader": "index always", "leader_safe": "index unless paused"}[c["park"]]
    s = f"{p}, {c['max_pos'] or 'any'} stocks, size x{c['size_mult']}, exit top {int(1.5 * c['N'])}, hold {c['min_hold']}"
    return s + (f", leader {c['leader_lb']}d" if c["park"] else "")


if __name__ == "__main__":
    rn = R.Runner()
    cfgs = grid()
    print(len(cfgs), "configurations", flush=True)
    res = []
    for k in range(0, len(cfgs), 36):
        chunk = cfgs[k:k + 36]
        for c, a in zip(chunk, rn.eval_many([materialise(c) for c in chunk], "train")):
            res.append(dict(label=label(c), cfg=json.dumps(c), objective=R.objective(a), **a))
        print(k + len(chunk), "done", flush=True)
    tr = pd.DataFrame(res).sort_values("objective", ascending=False)
    tr.to_csv("../results/exp6_train.csv", index=False)
    # finalists: Candidate 3 itself, Candidate 3 + index always (the earlier variant), and the
    # best 3 by training score for each index setting
    fin = [("Candidate 3", C3), ("C3 + index always (earlier)", dict(C3, park="leader"))]
    for park in ("no index", "index always", "index unless paused"):
        for _, r in tr[tr.label.str.startswith(park + ",")].head(3).iterrows():
            fin.append((r["label"], json.loads(r["cfg"])))
    rows = []
    for per in ("train", "test", "early"):
        for (lab, c), a in zip(fin, rn.eval_many([materialise(c) for _, c in fin], per)):
            print(per, R.fmt(lab[:44], a), flush=True)
            rows.append(dict(period=per, label=lab, cfg=json.dumps(c), **a))
    pd.DataFrame(rows).to_csv("../results/exp6_final.csv", index=False)
    m = X.Market(pd.read_pickle("../data/panel.pkl"))
    yrs = {}
    for lab, c in fin:
        parts = []
        for s, e in (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None)):
            parts.append(M.yearly(X.run(m, materialise(c), start=s, end=e)["curve"]))
        yrs[lab] = pd.concat(parts) * 100
    idx = X.run(m, {}, start="2000-01-03")["curve"].index
    for k, v in M.bench_curves(m, idx).items():
        parts = []
        for s, e in (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None)):
            parts.append(M.yearly(v[s:e]))
        yrs[k] = pd.concat(parts) * 100
    t = pd.DataFrame(yrs).round(1)
    t.index.name = "year"
    t.to_csv("../results/exp6_years.csv")
    best = t[["SPY", "QQQ", "SMH"]].max(axis=1)
    full = t.index < 2026
    for lab, _ in fin:
        w = (t[lab] > best) & full
        cnt = lambda a, b: int(w[(t.index >= a) & (t.index <= b)].sum())
        print(f"{lab[:70]:70s} beat all 3: 2000-11 {cnt(2000, 2011)}/12  2012-19 {cnt(2012, 2019)}/8  "
              f"2020-25 {cnt(2020, 2025)}/6  total {int(w.sum())}/26", flush=True)
