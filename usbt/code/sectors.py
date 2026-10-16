"""Point-in-time industry assignment (exp49): every 21 sessions, each stock is assigned to the sector/industry
ETF its daily returns moved with most over the previous 126 sessions (correlation). Uses only past prices.
Writes ../data/sector_idx.npy (T x N int8, -1 = none) and the ETF list."""
import numpy as np, pandas as pd, json, sys
P = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else "../data/panel_idx.pkl")
SECT = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "SMH", "IGV", "XLC"]
syms = list(P["syms"]); R = P["ret1"].astype(np.float64)
cols = [syms.index(s) for s in SECT]
T, N = R.shape
out = np.full((T, N), -1, np.int8)
cur = np.full(N, -1, np.int8)
W = int(sys.argv[2]) if len(sys.argv) > 2 else 126      # look-back window in sessions (audit: 63 / 126 / 252)
for i in range(W, T):
    if i % 21 == 0:
        w = R[i - W + 1:i + 1]
        E = w[:, cols]
        ok_e = np.isfinite(E).sum(0) >= int(0.8 * W)
        Wm = np.where(np.isfinite(w), w, np.nan)
        mu = np.nanmean(Wm, 0); sd = np.nanstd(Wm, 0)
        Z = (Wm - mu) / np.where(sd > 0, sd, np.nan)
        Ze = (E - np.nanmean(E, 0)) / np.where(np.nanstd(E, 0) > 0, np.nanstd(E, 0), np.nan)
        C = np.nanmean(Z[:, :, None] * Ze[:, None, :], 0)          # N x S
        C[:, ~ok_e] = -np.inf
        C = np.where(np.isfinite(C), C, -np.inf)
        best = C.argmax(1).astype(np.int8)
        cur = np.where(np.isfinite(C.max(1)) & (np.isfinite(Wm).sum(0) >= int(0.8 * W)), best, -1).astype(np.int8)
    out[i] = cur
np.save("../data/sector_idx.npy" if W == 126 else f"../data/sector_idx_{W}.npy", out); json.dump(SECT, open("../data/sector_list.json", "w"))
j = syms.index("NVDA"); i = list(P["dates"]).index(pd.Timestamp("2023-06-01"))
print("NVDA 2023-06 ->", SECT[out[i, j]], "| AAPL ->", SECT[out[i, syms.index("AAPL")]], "| XOM ->", SECT[out[i, syms.index("XOM")]], "| JPM ->", SECT[out[i, syms.index("JPM")]])
