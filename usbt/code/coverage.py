"""How survivorship-free is the panel? For the first session of each year, the share of that day's
S&P 500 members (fja05680/sp500 point-in-time lists) that have a price in our panel on that day.
Members we cannot price are mostly companies that later failed or were bought, so a low share
means the backtest is missing exactly the names that would have hurt it.
    python coverage.py   -> ../results/coverage.csv
"""
import numpy as np
import pandas as pd

P = pd.read_pickle("../data/panel.pkl")
dates, syms, C = P["dates"], list(P["syms"]), P["C"]
J = {s: j for j, s in enumerate(syms)}
h = pd.read_csv("../data/sp500_hist.csv", parse_dates=["date"]).set_index("date").sort_index()
rows = []
for y in range(2000, 2027):
    d = dates[dates.searchsorted(pd.Timestamp(f"{y}-01-01"))]
    mem = h.loc[:d].iloc[-1]["tickers"].split(",")
    mem = [t.replace(".", "-") for t in mem]
    i = dates.get_loc(d)
    have = [t for t in mem if t in J and C[i, J[t]] == C[i, J[t]]]
    # members that left the index later (the ones survivorship bias drops)
    rows.append(dict(year=y, members=len(mem), priced=len(have), share=len(have) / len(mem)))
df = pd.DataFrame(rows)
df.to_csv("../results/coverage.csv", index=False)
print(df.to_string(index=False))
