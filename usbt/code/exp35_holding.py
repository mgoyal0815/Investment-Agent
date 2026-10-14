"""Experiment 35: share of monthly starts where System C beats BOTH SPY and QQQ after 1, 3, 5 years and to the end date (argv[1]; default today)."""
import sys; sys.path.insert(0,'.')
import json, multiprocessing as mp, pandas as pd, numpy as np, engine as X, yardstick as Y
from exp24_timing import mat
END=sys.argv[1] if len(sys.argv)>1 else None
cfg=json.load(open('../system_C.json'))['cfg']
def job(s):
    r=X.run(Y.MM,mat(cfg),start=str(s.date()),end=END)["curve"]; idx=r.index
    B={k:pd.Series(Y.MM.E[k]["c"],index=Y.MM.dates).reindex(idx) for k in ("SPY","QQQ")}
    out={"start":s}
    for h in (1,3,5,None):
        t=idx[-1] if h is None else s+pd.DateOffset(years=h)
        if t>idx[-1]: out[h]=np.nan; continue
        e=idx[min(idx.searchsorted(t),len(idx)-1)]
        rs=r[e]/r.iloc[0]; out[h]=float(rs>B["SPY"][e]/B["SPY"].iloc[0] and rs>B["QQQ"][e]/B["QQQ"].iloc[0])
    return out
Y._init("../data/panel.pkl")
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=("../data/panel.pkl",))
T=pd.DataFrame(pool.map(job,Y.starts()))
print("System C beats BOTH SPY and QQQ, by holding period, share of monthly starts 2000-2025:")
for h in (1,3,5,None): print(" held", h or "to today", f"{T[h].mean():.0%}  (n={T[h].notna().sum()})")


