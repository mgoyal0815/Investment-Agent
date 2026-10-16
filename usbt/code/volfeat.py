"""Volume features (owner, 16 Oct 2026), from the daily volume already in data/raw (FMP, dividend-adjusted OHLCV).
Builds data/volf_<T>x<N>/*.npy once; the engine reads them as memory maps (Market.volf). All use data up to close i.
  pp10     pocket pivot in the last 10 sessions (Kacher & Morales 2010): an up day whose volume beats every down day's
           volume of the prior 10 sessions, closing in the top half of its range, within 5% of its 10-day average
  vdry     10-day / 50-day average volume (volume dry-up below 1 = quiet base, Minervini)
  climax   O'Neil climax run: up 25%+ in 15 sessions or up on 8 of 10 days, 70%+ above the 200-day average, and a stall
           day (close in the bottom third of the range on 1.5x the 50-day volume) in the last 5 sessions
  dd25     distribution days in 25 sessions (O'Neil): SPY or QQQ down 0.2%+ on higher volume than the day before (max)
  ftd      follow-through day (O'Neil): SPY or QQQ up 1.25%+ on higher volume, 4-30 sessions after its 30-day closing low"""
import os, sys
import numpy as np, pandas as pd


def build(panel_path, raw_dir, out_dir):
    P = pd.read_pickle(panel_path); D = pd.DatetimeIndex(P["dates"]); syms = list(P["syms"]); T, N = len(D), len(syms)
    V = np.full((T, N), np.nan, np.float32); H = V.copy(); L = V.copy(); C = V.copy()
    for k, s in enumerate(syms):
        f = os.path.join(raw_dir, f"{s}.parquet")
        if not os.path.exists(f): continue
        d = pd.read_parquet(f).set_index("date").sort_index(); d = d[~d.index.duplicated()].reindex(D)
        V[:, k], H[:, k], L[:, k], C[:, k] = d.volume, d.high, d.low, d.close
    del P
    Cd, Vd = pd.DataFrame(C), pd.DataFrame(V)
    up = (Cd > Cd.shift(1)).to_numpy(); dn = (Cd < Cd.shift(1)).to_numpy()
    rng = np.where(H - L > 0, H - L, np.nan); clv = (C - L) / rng
    downvol = pd.DataFrame(np.where(dn, V, 0.0)).shift(1).rolling(10, min_periods=5).max().to_numpy()
    s10 = Cd.rolling(10, min_periods=8).mean().to_numpy()
    ppd = up & (V > downvol) & (clv >= 0.5) & (np.abs(C / s10 - 1) <= 0.05)
    pp10 = pd.DataFrame(ppd.astype(np.float32)).rolling(10, min_periods=1).max().to_numpy() > 0
    v10 = Vd.rolling(10, min_periods=8).mean(); v50 = Vd.rolling(50, min_periods=40).mean()
    vdry = (v10 / v50).to_numpy(np.float32)
    r15 = (Cd / Cd.shift(15) - 1).to_numpy(); ups10 = pd.DataFrame(up.astype(np.float32)).rolling(10).sum().to_numpy()
    s200 = Cd.rolling(200, min_periods=150).mean().to_numpy()
    stall = (clv <= 1 / 3) & (V > 1.5 * v50.to_numpy())
    stall5 = pd.DataFrame(stall.astype(np.float32)).rolling(5, min_periods=1).max().to_numpy() > 0
    climax = ((r15 >= 0.25) | (ups10 >= 8)) & (C >= 1.7 * s200) & stall5
    dd, ftd = np.zeros(T), np.zeros(T, bool)
    for t in ("SPY", "QQQ"):
        e = pd.read_parquet(os.path.join(raw_dir, f"{t}.parquet")).set_index("date").sort_index().reindex(D)
        c, v = e.close, e.volume; r = c.pct_change()
        dist = ((r <= -0.002) & (v > v.shift(1))).astype(float).rolling(25, min_periods=1).sum().to_numpy()
        dd = np.maximum(dd, np.nan_to_num(dist))
        low_pos = c.rolling(30, min_periods=10).apply(lambda x: len(x) - 1 - int(np.nanargmin(x)), raw=True)
        f_ = (r >= 0.0125) & (v > v.shift(1)) & (low_pos >= 4) & (low_pos <= 29)
        ftd |= f_.fillna(False).to_numpy()
    os.makedirs(out_dir, exist_ok=True)
    for k, a in dict(pp10=pp10, vdry=vdry, climax=climax, dd25=dd, ftd=ftd).items():
        np.save(os.path.join(out_dir, k + ".npy"), a)
    print("volume features written", out_dir, {k: float(np.nanmean(a)) for k, a in dict(pp10=pp10, climax=climax, ftd=ftd).items()}, "dd25 mean", float(dd.mean()))


if __name__ == "__main__":
    pp = sys.argv[1] if len(sys.argv) > 1 else "../data/panel_ndx.pkl"
    P = pd.read_pickle(pp); T, N = len(P["dates"]), len(P["syms"]); del P
    build(pp, "../data/raw", f"../data/volf_{T}x{N}")
