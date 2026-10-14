"""What the indicators said vs what happened, over every trade of the System D journal (2000-2026)."""
import numpy as np
import pandas as pd

J = pd.read_csv("../results/journal_D.csv", parse_dates=["date"])
pd.set_option("display.width", 250)
out = []


def tab(df, col, bins=None, labels=None, what="next63"):
    x = pd.cut(df[col], bins, labels=labels) if bins is not None else df[col]
    g = df.groupby(x, observed=True)[what]
    t = pd.DataFrame({"trades": g.size(), "median next 3m %": (g.median() * 100).round(1),
                      "share up %": (g.apply(lambda s: (s > 0).mean()) * 100).round(0),
                      "vs SPY next 3m %": ((df.groupby(x, observed=True)[what].mean() - df.groupby(x, observed=True)["spy_next63"].mean()) * 100).round(1)})
    return t


B = J[J.action == "BUY"]
S = J[J.action == "SELL"]
txt = []
txt.append(("BUYS by RSI at entry (14-day)", tab(B, "rsi", [0, 60, 70, 75, 80, 85, 101])))
txt.append(("BUYS by ADX at entry", tab(B, "adx", [0, 15, 20, 25, 35, 100])))
txt.append(("BUYS by volatility (annualised, 30-day)", tab(B, "vol30", [0, 0.25, 0.35, 0.5, 0.7, 5])))
txt.append(("BUYS by distance from 52-week high", tab(B, "near_high", [0, 0.85, 0.95, 0.99, 1.01])))
txt.append(("BUYS by MACD above signal", tab(B, "macd_pos")))
txt.append(("BUYS by market: narrow (mega caps leading)", tab(B, "narrow")))
txt.append(("BUYS by market: credit stress", tab(B, "credit_stress")))
txt.append(("BUYS by VIX", tab(B, "vix", [0, 15, 20, 25, 35, 100])))
txt.append(("BUYS by stock 6-month return at entry", tab(B, "ret6m", [-1, 0.2, 0.4, 0.7, 1.0, 10])))
# sells: a GOOD sell is one where the stock then fell or lagged; next63 > 0 means we sold something that kept rising
txt.append(("SELLS by reason (next 3m = what the stock did AFTER we sold)", tab(S, "why")))
txt.append(("SELLS by trend still intact (above a rising 50-day EMA)", tab(S, "ema50_rising")))
txt.append(("SELLS by profit at sale", tab(S, "ret_vs_cost", [-1, -0.15, -0.05, 0.05, 0.2, 0.5, 100])))
txt.append(("SELLS by RSI at sale", tab(S, "rsi", [0, 35, 45, 55, 100])))
txt.append(("SELLS by ADX at sale", tab(S, "adx", [0, 15, 20, 30, 100])))
for lab, t in txt:
    print("\n" + lab); print(t.to_string())
S2 = S[(S.why == "rank")]
w = S2[S2.ema50_rising & (S2.ret_vs_cost > 0)]
print(f"\nRank sells of PROFITABLE stocks whose trend was still intact: {len(w)}; median next 3m {w.next63.median()*100:.1f}%, next 6m {w.next126.median()*100:.1f}%")
w = S2[~S2.ema50_rising & (S2.ret_vs_cost < 0)]
print(f"Rank sells of LOSING stocks whose trend had broken: {len(w)}; median next 3m {w.next63.median()*100:.1f}%, next 6m {w.next126.median()*100:.1f}%")
