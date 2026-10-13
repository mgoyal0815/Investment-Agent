"""Reference: the LIVE v5 code (live_harness) vs the fast engine's v5 default, continuous runs."""
import pandas as pd, engine as X, live_harness as LH, metrics as M
P = pd.read_pickle("../data/panel.pkl"); m = X.Market(P)
rows = []
for s, e in (("2012-01-03", "2018-12-31"), ("2019-01-02", None), ("2012-01-03", None), ("2000-01-03", "2011-12-30")):
    for lab, f in (("live code", lambda: LH.run(m, start=s, end=e)), ("engine v5", lambda: X.run(m, {}, start=s, end=e))):
        r = f(); sc = M.score(m, r["curve"]); t = sc.pop("table")
        print(f"{s[:4]}-{(e or '2026')[:4]} " + M.line(lab, sc), flush=True)
        rows.append(dict(window=f"{s}..{e or 'end'}", label=lab, **sc))
pd.DataFrame(rows).to_csv("../results/ref_live_vs_engine.csv", index=False)
