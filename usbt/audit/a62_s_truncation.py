"""Audit 1: look-ahead test. Rebuild the market from the panel with every row after a cut-off date deleted
(prices, indicators, ETFs, VIX, volume indicators, sector map) and rerun System S (R + 2x parking). If any rule used data
after the cut-off, the account value up to the cut-off would differ from the full-data run."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, numpy as np, pandas as pd, engine as X
from exp24_timing import mat
P = pd.read_pickle("../data/panel_ndx.pkl")
cfg = mat(dict(json.load(open("../system_R.json"))["cfg"], lev_park=200, lev_after=126, lev_real=True, lev_only=["SPY", "QQQ"]))
_load = np.load
def cut_market(T):
    n = int((P["dates"] <= pd.Timestamp(T)).sum())
    Q = {}
    for k, v in P.items():
        if k == "dates": Q[k] = v[:n]
        elif k == "etf": Q[k] = {e: (o[:n], c[:n]) for e, (o, c) in v.items()}
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(P["dates"]): Q[k] = v[:n]
        else: Q[k] = v
    class Z(dict):
        pass
    def fake_load(fn, *a, **k):
        z = _load(fn, *a, **k)
        if hasattr(z, "files"):
            d = Z({f: z[f][:n] for f in z.files}); d.files = list(d.keys()); return d
        return z[:n] if getattr(z, "ndim", 0) >= 1 and z.shape[0] == len(P["dates"]) else z
    X.np.load = fake_load
    try:
        return X.Market(Q)
    finally:
        X.np.load = _load
full = X.run(X.Market(P), cfg, start="2000-01-03")["curve"]
out = []
for T in ("2003-06-30", "2008-10-15", "2011-08-31", "2016-02-29", "2020-03-31", "2022-10-14", "2025-04-30"):
    m = cut_market(T)
    c = X.run(m, cfg, start="2000-01-03")["curve"]
    a = full.loc[:T].iloc[:-1]; b = c.loc[:T].iloc[:-1]     # the last session's open-to-close is the same either way
    diff = float((a / b - 1).abs().max())
    out.append((T, len(b), f"{diff:.2e}", "IDENTICAL" if diff < 1e-9 else "DIFFERS"))
    print(out[-1], flush=True)
