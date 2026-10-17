"""System R 2000-01-03 to 2025-12-31, one account, over tie seeds 1-40 plus the alphabetical tie-break (owner, 17 Oct
2026: restate growth and worst fall as a median with a range, not one lucky path). Account values (not rescaled)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, numpy as np, pandas as pd, yardstick as Y, robust as RB, engine as X
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"
SEEDS = tuple(range(1, 41))


KEYS = ("rev_alpha", "rs126", "dist200", "zsum", "m12", "hi52", "medv", "lowvol")


def one(a):
    cfg, sd = a
    c_ = dict(cfg, tie_key=sd) if isinstance(sd, str) else dict(cfg, tie_seed=sd)
    c = X.run(Y.MM, mat(c_), start="2000-01-03", end="2025-12-31")["curve"]
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    return dict(tie_seed="alpha" if sd is None else sd, end=float(c.iloc[-1]), cagr=float((c.iloc[-1] / 1e4) ** (1 / yrs) - 1),
                worst=float((c / c.cummax() - 1).min()))


if __name__ == "__main__":
    x = sys.argv[1] if len(sys.argv) > 1 else "R"
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    cfg = json.load(open(f"../system_{x}.json"))["cfg"]
    T = pd.DataFrame(pool.map(one, [(cfg, None)] + [(cfg, s) for s in SEEDS] + [(cfg, k) for k in KEYS], chunksize=1))
    T.to_csv(f"a70_{x}_paths.csv", index=False)
    s = T[T.tie_seed.map(lambda v: isinstance(v, int))]
    head = {k: dict(alpha=float(T[k].iloc[0]), median=float(s[k].median()), lo=float(s[k].min()), hi=float(s[k].max()),
                    p10=float(s[k].quantile(.1)), p90=float(s[k].quantile(.9))) for k in ("end", "cagr", "worst")}
    json.dump(dict(system=x, head=head, provenance=RB.provenance(seeds=SEEDS, panel=PATH, period="2000-01-03..2025-12-31")),
              open(f"a70_{x}_paths.json", "w"), indent=1)
    print(json.dumps(head, indent=1)); print(T[T.tie_seed.map(lambda v: isinstance(v, str))].to_string())
