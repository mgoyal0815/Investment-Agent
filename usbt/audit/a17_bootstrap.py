"""Validation 2: bootstrap. K's 26 calendar years (2000-2025, one run from 2000) and K's every-start results.
(a) Resample the 26 years with replacement 10,000 times: share of years beating both SPY and QQQ.
(b) Block bootstrap with 3-year blocks (keeps bull/bear runs together).
(c) Per-start years-beating rates from the every-start table: spread across start dates."""
import numpy as np, pandas as pd, json
y = pd.read_csv("../results/final_K_years.csv", index_col=0).loc[2000:2025]
win = (y["System K"] > y[["SPY", "QQQ"]].max(axis=1)).to_numpy().astype(float)
rng = np.random.default_rng(0)
a = np.array([rng.choice(win, len(win)).mean() for _ in range(10000)])
blocks = [win[i:i + 3] for i in range(len(win) - 2)]
b = np.array([np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), 9)])[:26].mean() for _ in range(10000)])
out = dict(calendar_wins=f"{int(win.sum())} of {len(win)}", share=float(win.mean()),
           iid_5_50_95=[float(np.percentile(a, p)) for p in (5, 50, 95)], block_5_50_95=[float(np.percentile(b, p)) for p in (5, 50, 95)],
           chance_75pct_or_more_iid=float((a >= 0.75).mean()), chance_75pct_or_more_block=float((b >= 0.75).mean()),
           chance_below_50pct_block=float((b < 0.5).mean()))
print(json.dumps(out, indent=1)); json.dump(out, open("../audit/a17_bootstrap.json", "w"), indent=1)
