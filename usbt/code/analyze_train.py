"""Where does the current best make and lose money on the TRAINING window (2012-2019)?
Trade-level and day-level attribution. Writes results/root_*.csv and prints the tables."""
import sys
import numpy as np
import pandas as pd
import engine as X, metrics as M
from exp2_search import materialise
from exp8_more import NEW

S, E = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2012-01-03", "2019-12-31")
m = X.Market(pd.read_pickle("../data/panel.pkl"))
r = X.run(m, materialise(NEW), start=S, end=E, record=True)
L, D = r["lots"], r["daily"]
Cf = m.P["Cf"]
pd.set_option("display.width", 220, "display.max_columns", 30)

# what each closed position did after we sold it (63 sessions on)
def after(row, n=63):
    i, j = row.exit_i, row.j
    k = min(i + n, len(m.dates) - 1)
    a, b = Cf[i, j], Cf[k, j]
    return b / a - 1 if a == a and b == b and a > 0 else np.nan
L["after63"] = L.apply(after, axis=1)
L["ret"] = L.pnl / L.invested
L["win"] = L.pnl > 0
print(f"closed positions {len(L)}, total P&L ${L.pnl.sum():,.0f}, win rate {L.win.mean():.0%}, "
      f"median hold {L.held.median():.0f} sessions, costs ${r['costs']:,.0f}")

def tab(by):
    g = L.groupby(by)
    return pd.DataFrame({"n": g.size(), "pnl$": g.pnl.sum().round(0), "win%": (g.win.mean() * 100).round(0),
                         "avg ret%": (g.ret.mean() * 100).round(1), "median hold": g.held.median(),
                         "after-exit 63d%": (g.after63.median() * 100).round(1)})
print("\nBY EXIT REASON\n", tab("why"))
print("\nBY ROUTE (1 continuation, 2 recovery)\n", tab("route"))
print("\nBY ZONE AT ENTRY\n", tab("zone"))
L["hold_b"] = pd.cut(L.held, [-1, 10, 21, 63, 126, 252, 9999], labels=["<=10", "11-21", "22-63", "64-126", "127-252", ">252"])
print("\nBY HOLDING PERIOD (sessions)\n", tab("hold_b"))
L["size_b"] = pd.cut(L.invested / 100, [0, 1, 2, 5, 10, 1e9], labels=["<$100", "$100-200", "$200-500", "$500-1000", ">$1000"])
print("\nBY AMOUNT INVESTED\n", tab("size_b"))
L["score_b"] = pd.cut(L.score, [0, 0.9, 0.95, 0.99, 1.01])
print("\nBY RANK SCORE AT ENTRY\n", tab("score_b"))
print("\nTOP 10 WINNERS\n", L.nlargest(10, "pnl")[["sym", "entry", "exit", "held", "invested", "pnl", "peak_ret", "exit_ret", "why"]].round(2))
print("\nTOP 10 LOSERS\n", L.nsmallest(10, "pnl")[["sym", "entry", "exit", "held", "invested", "pnl", "peak_ret", "exit_ret", "why"]].round(2))
share_top = L.nlargest(int(len(L) * 0.05), "pnl").pnl.sum() / L.pnl.sum()
print(f"\ntop 5% of positions make {share_top:.0%} of total P&L")
gave_back = L[(L.peak_ret > 0.3)]
print(f"positions that were up 30%+ at some point: {len(gave_back)}, gave back on average "
      f"{(gave_back.peak_ret - gave_back.exit_ret).mean():.0%} of entry price before exit")

# day-level attribution by year
D = D.copy()
D["cash_share"] = D.cash / D.nav
D["stock_share"] = D.stocks / D.nav
D["park_share"] = D.parked / D.nav
spy = pd.Series(m.E["SPY"]["c"], index=m.dates).reindex(D.index)
yr = D.groupby(D.index.year)
att = pd.DataFrame({"return%": ((yr.nav.last() / yr.nav.first() - 1) * 100).round(1),
                    "SPY%": ((spy.groupby(spy.index.year).last() / spy.groupby(spy.index.year).first() - 1) * 100).round(1),
                    "in stocks%": (yr.stock_share.mean() * 100).round(0), "parked%": (yr.park_share.mean() * 100).round(0),
                    "idle cash%": (yr.cash_share.mean() * 100).round(0), "paused days%": (yr.paused.mean() * 100).round(0),
                    "positions": yr.npos.mean().round(0)})
L["exit_year"] = pd.to_datetime(L.exit).dt.year
att["stock P&L$"] = L.groupby("exit_year").pnl.sum().round(0)
print("\nBY YEAR\n", att)
L.to_csv(f"../results/root_lots_{S[:4]}_{(E or 'end')[:4]}.csv", index=False)
att.to_csv(f"../results/root_years_{S[:4]}_{(E or 'end')[:4]}.csv")
