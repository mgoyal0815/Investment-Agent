"""Tie-break check (owner, 17 Oct 2026): engine.rank_desc must order tied scores identically on every numpy version.
Run on two versions and compare the hashes, e.g. container python3 (numpy 2.5.3) and /home/claude/v5/.venv (1.26.4).
17 Oct 2026: rank_desc 9264628e7fa85137 on both; the old np.argsort gave 293865930d79a5c1 (2.5.3) vs 8fd624af0becdd2f (1.26.4)."""
import numpy as np, hashlib
def rank_desc(x, key):
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=-np.inf)
    return np.lexsort((key, -x))
rng = np.random.default_rng(0); h = hashlib.sha256(); h2 = hashlib.sha256()
for t in range(2000):
    n = 1556
    x = np.round(rng.random(n), 2).astype(np.float32); x[rng.random(n) < 0.2] = np.nan   # heavy ties + NaN
    key = rng.permutation(n)
    h.update(rank_desc(x, key).astype(np.int64).tobytes())
    h2.update(np.argsort(-np.nan_to_num(x, nan=-np.inf)).astype(np.int64).tobytes())
print(np.__version__, "rank_desc", h.hexdigest()[:16], "old argsort", h2.hexdigest()[:16])
