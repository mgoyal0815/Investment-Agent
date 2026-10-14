"""Experiment 7: (1) sector ETFs in the stock picker's universe, (3) faster resumption after a buy
pause. Applied to Candidate 3 and to C3 + index-unless-paused (exp6's best).
  etf: off / on with position size x1 / x2 / x3 (an ETF is less volatile, so it may hold more)
  pause: India v8 (any of breadth, HYG, large-cap trend), two-of-three, HYG + large-cap only,
         large-cap only, breadth + large-cap, breadth threshold 30%, HYG and large-cap both
Chosen on 2012-2019 with runner.objective, tested on 2020-2026 and 2000-2011.
Writes results/exp7_train.csv, exp7_final.csv, exp7_years.csv, exp7_log.txt (stdout)."""
import itertools
import json
import pandas as pd
import engine as X, metrics as M, runner as R
from exp2_search import materialise
from exp6_c3_grid import C3

C3S = dict(C3, park="leader_safe", leader_lb=126)
PAUSES = {
    "pause any of 3": dict(pause_rules=("breadth", "hyg", "large"), pause_mode="any"),
    "pause 2 of 3": dict(pause_rules=("breadth", "hyg", "large"), pause_mode="two"),
    "pause HYG or large": dict(pause_rules=("hyg", "large"), pause_mode="any"),
    "pause large only": dict(pause_rules=("large",), pause_mode="any"),
    "pause breadth or large": dict(pause_rules=("breadth", "large"), pause_mode="any"),
    "pause any, breadth 30%": dict(pause_rules=("breadth", "hyg", "large"), pause_mode="any", breadth_level=0.30),
    "pause HYG and large": dict(pause_rules=("hyg", "large"), pause_mode="two"),
}
ETF = {"no ETFs": dict(etf_in_universe=False), "ETFs x1": dict(etf_in_universe=True, etf_size_mult=1.0),
       "ETFs x2": dict(etf_in_universe=True, etf_size_mult=2.0), "ETFs x3": dict(etf_in_universe=True, etf_size_mult=3.0)}


def grid():
    out = []
    for (bl, b), (el, e), (pl, p) in itertools.product((("C3", C3), ("C3+index unless paused", C3S)), ETF.items(), PAUSES.items()):
        out.append((f"{bl}, {el}, {pl}", dict(b, **e, **p)))
    return out


if __name__ == "__main__":
    rn = R.Runner()
    g = grid()
    res = rn.eval_many([materialise(c) for _, c in g], "train")
    tr = pd.DataFrame([dict(label=l, cfg=json.dumps(c, default=list), objective=R.objective(a), **a) for (l, c), a in zip(g, res)])
    tr = tr.sort_values("objective", ascending=False)
    tr.to_csv("../results/exp7_train.csv", index=False)
    print("top 10 on 2012-2019:")
    for _, r in tr.head(10).iterrows():
        print(f"  {r.objective:.3f} {r.label}", flush=True)
    refs = ["C3, no ETFs, pause any of 3", "C3+index unless paused, no ETFs, pause any of 3"]
    pick = list(tr.head(4).label)
    for fam in ("no ETFs", "ETFs x"):      # best pause-only change and best ETF change, so both levers are seen
        sub = tr[tr.label.str.contains(fam) & ~tr.label.isin(refs + pick)]
        pick += list(sub.head(1).label)
    fin = [(l, json.loads(tr[tr.label == l].iloc[0].cfg)) for l in refs + [p for p in pick if p not in refs]]
    rows = []
    for per in ("train", "test", "early"):
        for (lab, c), a in zip(fin, rn.eval_many([materialise(c) for _, c in fin], per)):
            print(per, R.fmt(lab[:44], a), flush=True)
            rows.append(dict(period=per, label=lab, cfg=json.dumps(c, default=list), **a))
    pd.DataFrame(rows).to_csv("../results/exp7_final.csv", index=False)
    m = X.Market(pd.read_pickle("../data/panel.pkl"))
    spans = (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None))
    yrs = {}
    for lab, c in fin:
        yrs[lab] = pd.concat([M.yearly(X.run(m, materialise(c), start=s, end=e)["curve"]) for s, e in spans]) * 100
    full = X.run(m, {}, start="2000-01-03")["curve"].index
    for k, v in M.bench_curves(m, full).items():
        yrs[k] = pd.concat([M.yearly(v[s:e]) for s, e in spans]) * 100
    t = pd.DataFrame(yrs).round(1)
    t.index.name = "year"
    t.to_csv("../results/exp7_years.csv")
    best = t[["SPY", "QQQ", "SMH"]].max(axis=1)
    for lab, _ in fin:
        w = (t[lab] > best) & (t.index < 2026)
        cnt = lambda a, b: int(w[(t.index >= a) & (t.index <= b)].sum())
        print(f"{lab[:62]:62s} beat all 3: 2000-11 {cnt(2000, 2011)}/12  2012-19 {cnt(2012, 2019)}/8  "
              f"2020-25 {cnt(2020, 2025)}/6  total {int(w.sum())}/26  years {[y for y in t.index if w[y]]}", flush=True)
