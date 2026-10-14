"""Experiment 5: ideas aimed at the index target itself, all three windows.
Index-rotation core: part of the book always in whichever of SPY/QQQ/SMH has the best
126-session momentum (v5 section 6 leader rule), the rest in the stock sleeve.
Writes results/exp5_index_core.csv and per-year counts."""
import json
import pandas as pd
import engine as X, metrics as M, runner as R
from exp2_search import materialise

C3 = json.loads(pd.read_csv("../results/exp2_top.csv").iloc[2]["cfg"])
def w(s, l):
    return dict(FULL_BULL=(s, l), CREDIT_WARN=(s * 0.8, l * 0.8, 1 - s * 0.8 - l * 0.8), MILD_BEAR=(0.30,), DEEP_BEAR=(0.40,))
CFGS = [
    ("US v5 (live)", {}),
    ("Candidate 3", C3),
    ("v5, 30% index core", dict(regime_w=w(0.7, 0.3))),
    ("v5, 50% index core", dict(regime_w=w(0.5, 0.5))),
    ("index rotation only (no stocks)", dict(regime_w=w(0.0, 1.0))),
    ("Candidate 3, idle cash in leading index", dict(C3, park="leader")),
]
if __name__ == "__main__":
    rn = R.Runner()
    rows = []
    for per in ("train", "test", "early"):
        for (lab, c), a in zip(CFGS, rn.eval_many([materialise(c) for _, c in CFGS], per)):
            print(per, R.fmt(lab, a), flush=True)
            rows.append(dict(period=per, label=lab, **a))
    pd.DataFrame(rows).to_csv("../results/exp5_index_core.csv", index=False)
    m = X.Market(pd.read_pickle("../data/panel.pkl"))
    for lab, c in CFGS:
        out = []
        for name, s, e in (("train 2012-19", "2012-01-03", "2019-12-31"), ("test 2020-25", "2020-01-02", "2025-12-31"), ("early 2000-11", "2000-01-03", "2011-12-30")):
            t = M.score(m, X.run(m, materialise(c), start=s, end=e)["curve"])
            out.append(f"{name}: SPY {t['beat_SPY']} QQQ {t['beat_QQQ']} SMH {t['beat_SMH']} ALL {t['beat_all']}/{t['years']}")
        print(lab.ljust(42), " | ".join(out), flush=True)
