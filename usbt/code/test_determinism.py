"""Determinism test (owner, 17 Oct 2026): the same system on the same panel must give the same account, run after run
and on every numpy / pandas version. Prints a JSON line: environment, end value, and SHA-256 hashes of the daily
account curve (exact float64 bytes, and rounded to the cent), of every trade decision (date, symbol, side, reason)
and of the final book. The decisions hash must match on every environment; the curve may differ by float rounding
(about a cent: pandas 2 and 3 handle float32 columns differently), never by a trade.

    python3 test_determinism.py <panel.pkl> <system json> [start] [end] [key=value ...]

Run it twice on one environment (must match exactly) and on two environments, e.g. the container's python3
(numpy 2.5.3 / pandas 3.0.5) and the live venv /home/claude/v5/.venv/bin/python (numpy 1.26.4 / pandas 2.2.2); the
cent-rounded hashes must match. The engine is imported from the working directory (engine.py)."""
import hashlib, json, sys
import numpy as np, pandas as pd
import engine as X


def materialise(c):
    c = dict(c)
    if c.get("sizing") == "slot":
        c["min_hold"] = c.get("rank_min_hold", 0)
    if c.get("lock_at") is None:
        c.pop("lock_at", None)
    return c


panel, sysf = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
pos = [a for a in rest if "=" not in a]
start = pos[0] if pos else "2000-01-03"
end = pos[1] if len(pos) > 1 else "2025-12-31"
cfg = json.load(open(sysf))
cfg = cfg.get("cfg", cfg)
for a in rest:
    if "=" in a and not a.startswith("dump="):
        k, v = a.split("=", 1)
        cfg[k] = json.loads(v) if v[:1] in "[{0123456789-tfn" else v
m = X.Market(pd.read_pickle(panel))
out = X.run(m, materialise(cfg), start=start, end=end, record=True)
c = out["curve"].to_numpy(np.float64)
book = json.dumps(out.get("book"), sort_keys=True, default=str)
dump = [a.split("=", 1)[1] for a in rest if a.startswith("dump=")]
if dump:                                   # dump=<prefix>: write the curve and the book for a side-by-side comparison
    out["curve"].to_csv(dump[0] + "_curve.csv"); open(dump[0] + "_book.json", "w").write(book)
print(json.dumps(dict(numpy=np.__version__, pandas=pd.__version__, start=start, end=end, overrides=[a for a in rest if "=" in a],
                      end_value=round(float(c[-1]), 2), sessions=len(c),
                      curve_sha_exact=hashlib.sha256(c.tobytes()).hexdigest()[:16],
                      curve_sha_cents=hashlib.sha256(np.round(c, 2).tobytes()).hexdigest()[:16],
                      trades=len(out["trades"]),
                      decisions_sha=hashlib.sha256(out["trades"][["date", "sym", "side", "why"]].astype(str).to_csv(index=False).encode()).hexdigest()[:16],
                      book_sha=hashlib.sha256(book.encode()).hexdigest()[:16])))
