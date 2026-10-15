"""Audit 3: price glitches in what System H actually held. For every holding period, flag one-day moves
over 35% (up or down) and moves that reverse by more than half the next day (classic bad tick or split error)."""
import pickle, numpy as np, pandas as pd
P = pd.read_pickle("../data/panel.pkl"); C = P["Cf"].astype(float); D = P["dates"]; S = P["syms"]
R = pickle.load(open("/tmp/claude-0/-home-user/611e8ebf-cd90-5f23-b4dc-f40b7c4b7cdf/scratchpad/runH.pkl", "rb"))
L = R["lots"]; di = {d: k for k, d in enumerate(D)}
rows = []
for t in L.itertuples():
    a = di.get(pd.Timestamp(t.entry)); b = di.get(pd.Timestamp(t.exit)) if pd.notna(t.exit) else len(D) - 1
    if a is None or b is None: continue
    c = C[a:b + 1, t.j]; r = c[1:] / c[:-1] - 1
    for k in np.flatnonzero(np.abs(np.nan_to_num(r)) > 0.35):
        nxt = r[k + 1] if k + 1 < len(r) else np.nan
        rows.append(dict(sym=t.sym, date=D[a + k + 1].date(), move=round(float(r[k]), 3), next_day=round(float(nxt), 3) if nxt == nxt else None,
                         reverses=bool(nxt == nxt and np.sign(nxt) != np.sign(r[k]) and abs(nxt) > abs(r[k]) / 2), lot_pnl=round(t.pnl)))
G = pd.DataFrame(rows)
print(f"lots checked: {len(L)}; one-day moves over 35% inside holdings: {len(G)}; of which reverse next day: {int(G.reverses.sum()) if len(G) else 0}")
if len(G): print(G.to_string(index=False))
top = L.sort_values("pnl", ascending=False).head(15)[["sym", "entry", "exit", "exit_ret", "pnl"]]
print("\nTop 15 lots by profit (largest contributors):"); print(top.to_string(index=False))
print(f"\nTop 15 lots = {top.pnl.sum() / L.pnl.sum():.0%} of all lot profit")
