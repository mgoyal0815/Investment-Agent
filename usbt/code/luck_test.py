"""Luck test: rerun each system with its biggest winners removed from the universe (the stocks that
made the most money in that window), on the bias-free S&P 500 universe. Writes results/luck_test.csv."""
import json
import pandas as pd
import engine as X, metrics as M
from exp2_search import materialise

def mat(c):
    c = materialise(c)
    if c.get("lock_at") is None:
        c.pop("lock_at", None)
    return c
SYS = [("v5 live, S&P 500", dict(universe="sp500")),
       ("System A, S&P 500", dict(json.load(open("../system_A.json"))["cfg"], universe="sp500")),
       ("System B", json.load(open("../system_B.json"))["cfg"])]
m = X.Market(pd.read_pickle("../data/panel.pkl"))
rows = []
for lab, c in SYS:
    for w, s, e in (("2012-19", "2012-01-03", "2019-12-31"), ("2020-26", "2020-01-02", None), ("2000-11", "2000-01-03", "2011-12-30")):
        r = X.run(m, mat(c), start=s, end=e, record=True)
        top = r["lots"].groupby("sym").pnl.sum().sort_values(ascending=False)
        out = {"system": lab, "window": w, "all": M.cagr(r["curve"])}
        for k in (1, 3, 5):
            cc = dict(c, exclude=list(top.index[:k]))
            out[f"without top {k}"] = M.cagr(X.run(m, mat(cc), start=s, end=e)["curve"])
        out["top names"] = ", ".join(top.index[:5])
        rows.append(out)
        print(out, flush=True)
pd.DataFrame(rows).to_csv("../results/luck_test.csv", index=False)
spy = {w: M.cagr(pd.Series(m.E["SPY"]["c"], index=m.dates)[s:e]) for w, s, e in (("2012-19", "2012-01-03", "2019-12-31"), ("2020-26", "2020-01-02", None), ("2000-11", "2000-01-03", "2011-12-30"))}
print("SPY", {k: round(v, 3) for k, v in spy.items()})
