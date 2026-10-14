"""Year-by-year returns for every option and the three indexes. Three runs, each a fresh $10,000:
Jan 2000 -> Dec 2011, Jan 2012 -> Dec 2019, Jan 2020 -> 12 Oct 2026. Writes results/yearly_table.csv."""
import json
import pandas as pd
import engine as X, metrics as M
from exp2_search import materialise

top = pd.read_csv("../results/exp2_top.csv")
best = json.load(open("../results/exp2b_best.json"))["balanced"]["cfg"]
C3 = json.loads(top.iloc[2]["cfg"])
OPTS = [("v5 live", {}), ("Candidate 3", C3), ("C3 + idle cash in index", dict(C3, park="leader")),
        ("Hybrid A", best), ("Candidate 2", json.loads(top.iloc[1]["cfg"])), ("India v8", None)]
import runner as R
OPTS[-1] = ("India v8", R.INDIA)
m = X.Market(pd.read_pickle("../data/panel.pkl"))
rows = {}
for s, e in (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None)):
    cols = {}
    for lab, c in OPTS:
        cols[lab] = M.yearly(X.run(m, materialise(c), start=s, end=e)["curve"])
    idx = X.run(m, {}, start=s, end=e)["curve"].index
    for k, v in M.bench_curves(m, idx).items():
        cols[k] = M.yearly(v)
    rows[s] = pd.DataFrame(cols)
t = pd.concat(rows.values()) * 100
t.index.name = "year"
t.round(1).to_csv("../results/yearly_table.csv")
print(t.round(1).to_string())
