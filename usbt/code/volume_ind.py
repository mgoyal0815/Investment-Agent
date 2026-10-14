"""Volume indicators (exp52), computed stock by stock so memory stays small. For every panel column:
UDVR  50-day up-volume / down-volume (IBD-style accumulation; >1 = more volume on up days)
OBVT  On-Balance Volume vs its 50-day average, in days of average volume (rising OBV > 0)
CMF   20-day Chaikin Money Flow (-1..1; > 0 = closes near the day's high on volume)
RVOL  20-day average volume / 100-day average volume
Writes ../data/vol_<panel>.npz with float32 arrays aligned to that panel (T x N)."""
import sys
import numpy as np
import pandas as pd

name = sys.argv[1] if len(sys.argv) > 1 else "panel"
P = pd.read_pickle(f"../data/{name}.pkl")
idx, syms = P["dates"], list(P["syms"])
T, N = len(idx), len(syms)
out = {k: np.full((T, N), np.nan, np.float32) for k in ("UDVR", "OBVT", "CMF", "RVOL")}
for j, s in enumerate(syms):
    try:
        df = pd.read_parquet(f"../data/raw/{s}.parquet").set_index("date").sort_index()
    except Exception:
        continue
    df = df[~df.index.duplicated()].reindex(idx)
    traded = df["close"].notna()
    c = df["close"].ffill(); h = df["high"].ffill(); l = df["low"].ffill()
    v = df["volume"].where(traded, 0.0).fillna(0.0)
    d = c.diff()
    up = v.where(d > 0, 0.0); dn = v.where(d < 0, 0.0)
    out["UDVR"][:, j] = (up.rolling(50, min_periods=40).sum() / dn.rolling(50, min_periods=40).sum().replace(0, np.nan)).to_numpy()
    obv = (np.sign(d).fillna(0) * v).cumsum()
    va = v.rolling(50, min_periods=40).mean().replace(0, np.nan)
    out["OBVT"][:, j] = ((obv - obv.rolling(50, min_periods=40).mean()) / va).to_numpy()
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    out["CMF"][:, j] = ((mfm * v).rolling(20, min_periods=15).sum() / v.rolling(20, min_periods=15).sum().replace(0, np.nan)).to_numpy()
    out["RVOL"][:, j] = (v.rolling(20, min_periods=15).mean() / v.rolling(100, min_periods=60).mean().replace(0, np.nan)).to_numpy()
    for k in out:
        out[k][~np.asarray(P["Cf"][:, j] == P["Cf"][:, j]), j] = np.nan
np.savez_compressed(f"../data/vol_{name}.npz", **out)
j = syms.index("NVDA"); i = list(idx).index(pd.Timestamp("2023-05-30"))
print({k: round(float(out[k][i, j]), 3) for k in out})
for k in out:
    x = out[k][3000:]; x = x[np.isfinite(x)]; print(k, "10/50/90 pct", np.percentile(x, [10, 50, 90]).round(3))
