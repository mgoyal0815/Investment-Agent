"""Check a rebuilt panel against a stored result (REBUILD.md step 6).

    python3 check_rebuild.py [system_R] [../data/panel_ndx.pkl]

Runs the system from 2000-01-03 to 2025-12-31, rescales like final_*.py (first close = $10,000) and compares the
daily curve with results/final_<X>_curves.csv. Expect the end within ~0.1% (FMP revises dividend-adjusted history
every time a stock pays a dividend, so cent-level drift is normal). A first divergence in 2000-2001 or an end more
than 1% off means the build or the environment is wrong: check the BUILD.json stock count (1556 for panel_ndx) and
the numpy / pandas versions (envinfo.py)."""
import json, sys
import pandas as pd
import envinfo
envinfo.check()
import engine as X
from exp24_timing import mat

name = sys.argv[1] if len(sys.argv) > 1 else "system_R"
path = sys.argv[2] if len(sys.argv) > 2 else "../data/panel_ndx.pkl"
letter = name.split("_")[-1]
m = X.Market(pd.read_pickle(path))
cfg = json.load(open(f"../{name}.json"))["cfg"]
c = X.run(m, mat(cfg), start="2000-01-03", end="2025-12-31")["curve"]
mine = c / c.iloc[0] * 10000
ref = pd.read_csv(f"../results/final_{letter}_curves.csv", index_col=0, parse_dates=True)[f"System {letter}"]
d = pd.concat([mine.rename("mine"), ref.rename("ref")], axis=1).dropna()
d["rel"] = d.mine / d.ref - 1
first = {th: (d.index[d.rel.abs() > th][0].date() if (d.rel.abs() > th).any() else None) for th in (1e-6, 1e-3, 1e-2)}
end = d.rel.iloc[-1]
print(json.dumps(dict(system=name, env=envinfo.stamp(path), account_end=round(float(c.iloc[-1]), 2), rescaled_end=round(float(d.mine.iloc[-1]), 2),
                      stored_end=round(float(d.ref.iloc[-1]), 2), end_gap=f"{end:+.4%}", first_divergence={str(k): str(v) for k, v in first.items()}), indent=1))
ok = abs(end) < 0.01 and (first[1e-2] is None or first[1e-2].year >= 2002)
print("REBUILD OK" if ok else "REBUILD MISMATCH: see REBUILD.md")
sys.exit(0 if ok else 1)
