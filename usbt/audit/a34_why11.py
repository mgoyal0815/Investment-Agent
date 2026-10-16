"""Root cause of the 11th-session advantage (owner, 16 Oct 2026). One run 2000-2025 for second review on sessions
9-13 (stagger [-1, d]); yearly returns vs the better of SPY/QQQ, and the trades that make the difference."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, pickle, numpy as np, pandas as pd, engine as X, yardstick as Y
from exp24_timing import mat
Y._init("../data/panel_ndx.pkl"); m = Y.MM
O = json.load(open("../system_O.json"))["cfg"]
B = {t: pd.Series(m.E[t]["c"], index=m.dates).ffill()["1999-12-31":"2025-12-31"] for t in ("SPY", "QQQ")}
yr = lambda s: s.groupby(s.index.year).last().pct_change().dropna()
best = pd.concat([yr(B["SPY"]), yr(B["QQQ"])], axis=1).max(axis=1)
res, trades = {}, {}
for d in (8, 9, 10, 11, 12):
    r = X.run(m, mat(dict(O, stagger=[-1, d])), start="2000-01-03", end="2025-12-31", record=True)
    c = r["curve"]; y = c.groupby(c.index.year).last() / c.groupby(c.index.year).last().shift(1).fillna(c.iloc[0]) - 1
    res[f"session {d+1}"] = y; trades[d] = r["trades"]
    print(d + 1, "end $%.2fM" % (c.iloc[-1] / 1e6), "years beating both:", int((y.reindex(best.index) > best).sum()), flush=True)
Y_ = pd.DataFrame(res).loc[2000:2025]; Y_["best index"] = best
W = pd.DataFrame({k: Y_[k] > Y_["best index"] for k in res})
pd.set_option("display.width", 250)
print((Y_ * 100).round(1).to_string()); print(W.astype(int).to_string())
pickle.dump(dict(Y=Y_, W=W, trades=trades), open("/tmp/claude-0/-home-user/611e8ebf-cd90-5f23-b4dc-f40b7c4b7cdf/scratchpad/why11.pkl", "wb"))
