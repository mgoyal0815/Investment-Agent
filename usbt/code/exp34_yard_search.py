"""Experiment 34: search on the owner's yardstick (yardstick.py). Selection uses ONLY runs that start
2012-01..2016-12 and stop on 2019-12-31 (nothing after 2019 is seen). Score = share of years beating
both SPY and QQQ + 0.5 x share of 3-year windows beating SMH + 0.5 x share of starts with the smaller
drawdown + share of starts passing everything. Finalists are then judged on starts 2000-2008 (stopping
2011-12-30) and starts 2020-2025 (to today), and on every monthly start 2000-2025 to today."""
import json, multiprocessing as mp, random, sys
import pandas as pd
import yardstick as Y
from exp22_clean_search import SV5, SSL

SC = json.load(open("../system_C.json"))["cfg"]
SSL2 = dict(SSL, stagger=[None, [-1, 10], [-1, 5, 10, 15]])


def sample(rng):
    c, sp = (dict(sizing="v5"), SV5) if rng.random() < 0.35 else (dict(sizing="slot"), SSL2)
    for k, v in sp.items():
        c[k] = rng.choice(v)
    c["universe"] = "sp500"
    if c.get("stagger") is None:
        c.pop("stagger", None)
    return c


def score(s):
    return s["yrs_beat_SPY_QQQ"] + 0.5 * s["r3_beat_SMH"] + 0.5 * s["dd_ok"] + s["PASS"]


def ev(pool, c, st, end):
    T = Y.evaluate(pool, c, st, end)
    return Y.summary(T), T


def climb(pool, cur, st):
    sp = SSL2 if cur.get("sizing") == "slot" else SV5
    best = score(ev(pool, cur, st, "2019-12-31")[0])
    for p in range(2):
        imp = False
        for k, vals in sp.items():
            for v in vals:
                if v == cur.get(k):
                    continue
                c = dict(cur, **{k: v})
                if c.get("stagger") is None:
                    c.pop("stagger", None)
                s = score(ev(pool, c, st, "2019-12-31")[0])
                if s > best + 1e-6:
                    best, cur, imp = s, c, True
        print(f"  pass{p} {best:.3f}", flush=True)
        if not imp:
            break
    return cur, best


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    Y._init("../data/panel.pkl")
    tr = Y.starts("2012-01-01", "2016-12-31", 2)
    rng = random.Random(34)
    raw = [SC] + [sample(rng) for _ in range(n)]
    rows = []
    for k, c in enumerate(raw):
        s, _ = ev(pool, c, tr, "2019-12-31")
        rows.append(dict(cfg=json.dumps(c, default=list), score=score(s), **{x: s[x] for x in s if x != "label"}))
        if (k + 1) % 25 == 0:
            df = pd.DataFrame(rows).sort_values("score", ascending=False)
            df.to_csv("../results/exp34_all.csv", index=False)
            print(k + 1, df.iloc[0][["score", "yrs_beat_SPY_QQQ", "r3_beat_SMH", "dd_ok", "PASS", "med_cagr"]].round(3).to_dict(), flush=True)
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv("../results/exp34_all.csv", index=False)
    fin = [("System C", SC)]
    for i, c in enumerate(json.loads(x) for x in df.head(3).cfg):
        cc, o = climb(pool, c, tr)
        fin.append((f"pick {i + 1} (train {o:.3f})", cc))
    json.dump([{"label": l, "cfg": c} for l, c in fin], open("../results/exp34_finalists.json", "w"), default=list, indent=1)
    out = []
    for lab, c in fin:
        for per, (st, end) in {"train 2012-16 starts": (tr, "2019-12-31"),
                               "test 2000-08 starts": (Y.starts("2000-01-01", "2008-12-31", 2), "2011-12-30"),
                               "test 2020-25 starts": (Y.starts("2020-01-01", "2025-10-31", 1), None),
                               "every start 2000-25, to today": (Y.starts(), None)}.items():
            s, _ = ev(pool, c, st, end)
            s["label"], s["period"], s["score"] = lab, per, score(s)
            out.append(s)
            print(per, lab, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(out).to_csv("../results/exp34_finalists.csv", index=False)
