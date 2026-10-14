"""Trade journal (owner, 15 Oct 2026): every buy, add and sell of one System D run from 2000, with the
reason, what each indicator said at the decision (the close before the trade), and what happened next.
Writes results/journal_D.csv; prints a sample and an indicator-vs-outcome analysis."""
import json, sys
import numpy as np
import pandas as pd
import engine as X, metrics as M
from exp24_timing import mat

SYS = sys.argv[1] if len(sys.argv) > 1 else "../system_D.json"
m = X.Market(pd.read_pickle("../data/panel.pkl"))
P, D = m.P, m.dates
cfg = json.load(open(SYS))["cfg"]
r = X.run(m, mat(cfg), start="2000-01-03", record=True)
T = r["trades"] if isinstance(r["trades"], pd.DataFrame) else pd.DataFrame(r["trades"], columns=["date", "sym", "side", "value", "why", "ret"])
T.columns = ["date", "sym", "side", "value", "why", "ret_vs_cost"]
daily = r["daily"]
J = {s: j for j, s in enumerate(m.syms)}
di = {d: k for k, d in enumerate(D)}
Cf = P["Cf"]
spy = m.E["SPY"]["c"]


def fwd(i, j, n):
    k = min(i + n, len(D) - 1)
    a, b = Cf[i, j], Cf[k, j]
    return b / a - 1 if a == a and b == b and a > 0 else np.nan


rows = []
for t in T.itertuples():
    if t.sym not in J:
        continue
    j, i = J[t.sym], di[pd.Timestamp(t.date)] - 1          # decision at the previous close
    e50 = P["EMA50"]
    rows.append(dict(
        date=pd.Timestamp(t.date).date(), sym=t.sym, action=t.side, why=t.why, value=round(t.value), ret_vs_cost=round(t.ret_vs_cost, 3),
        # stock indicators at the decision
        ret6m=round(Cf[i, j] / Cf[i - 126, j] - 1, 3) if i >= 126 else np.nan,
        rs126=round(float(P["rs126"][i, j]), 3), above200=round(float(P["dist200"][i, j]), 3),
        trend_ok=bool(P["cont"][i, j]), ema50_rising=bool(Cf[i, j] > e50[i, j] and e50[i, j] > e50[i - 10, j]),
        rsi=round(float(P["RSI"][i, j]), 1), adx=round(float(P["ADX"][i, j]), 1), macd_pos=bool(P["MACDH"][i, j] > 0),
        near_high=round(float(P["hi52"][i, j]), 3), vol30=round(float(P["RV30"][i, j]) * np.sqrt(252), 3),
        # market at the decision
        paused=bool(daily["paused"].iloc[daily.index.get_indexer([D[i]], method="ffill")[0]]) if D[i] >= daily.index[0] else False,
        breadth200=round(float(m.breadth[i]), 2), breadth50=round(float(m.b50[i]), 2),
        vix=round(float(P["VIX"][i]), 1), spy_above200=bool(spy[i] > np.nanmean(spy[i - 199:i + 1])),
        narrow=bool(m.narrow[i]), credit_stress=bool(m.credit_bad[i]),
        # what happened next (stock price, from the decision close)
        next21=round(fwd(i, j, 21), 3), next63=round(fwd(i, j, 63), 3), next126=round(fwd(i, j, 126), 3),
        spy_next63=round(spy[min(i + 63, len(D) - 1)] / spy[i] - 1, 3)))
Jn = pd.DataFrame(rows)
Jn.to_csv("../results/journal_D.csv", index=False)
pd.set_option("display.width", 320, "display.max_columns", 40, "display.max_rows", 200)
show = ["date", "sym", "action", "why", "ret_vs_cost", "rsi", "adx", "macd_pos", "ema50_rising", "near_high", "vol30", "paused", "vix",
        "narrow", "credit_stress", "next63", "next126"]
print("SAMPLE: every trade, Jan 2023 - Dec 2024")
print(Jn[(pd.to_datetime(Jn.date) >= "2023-01-01") & (pd.to_datetime(Jn.date) <= "2024-12-31")][show].to_string(index=False))
Jn.to_pickle("/tmp/claude-0/-home-user/611e8ebf-cd90-5f23-b4dc-f40b7c4b7cdf/scratchpad/journal.pkl")
