"""Experiment 38: the weak spot (years big tech runs away). System C plus an index share inside the same
account: fixed QQQ/SMH/SPY share (with or without a 200-day trend filter), or the leader-index core
(hold the leading of SMH/QQQ while it is strong). Chosen on starts 2012-2016 stopping at 2019 with the
exp34 yardstick score; judged on starts 2000-2008 and 2020-2025 and every start to today."""
import itertools, json, multiprocessing as mp
import pandas as pd
import yardstick as Y
from exp34_yard_search import score

SC = json.load(open("../system_C.json"))["cfg"]


def grid():
    g = [("System C", SC)]
    for e, w, t in itertools.product(("QQQ", "SMH", "SPY"), (0.2, 0.3, 0.4, 0.5), (False, True)):
        g.append((f"+{int(w*100)}% {e}{' trend' if t else ''}", dict(SC, core_w=w, core_mode="fixed", core_etf=e, core_trend=t)))
    for w, mn, lb in itertools.product((0.3, 0.5), (0.0, 0.1, 0.2), (63, 126)):
        g.append((f"+{int(w*100)}% leader core min{mn} lb{lb}", dict(SC, core_w=w, core_min=mn, core_lb=lb)))
    return g


if __name__ == "__main__":
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 2), "2019-12-31"),
           "test 2000-08": (Y.starts("2000-01-01", "2008-12-31", 2), "2011-12-30"),
           "test 2020-25": (Y.starts("2020-01-01", "2025-10-31", 1), None),
           "all starts": (Y.starts(), None)}
    rows = []
    for lab, c in grid():
        r = {"label": lab}
        for p, (st, end) in PER.items():
            s = Y.summary(Y.evaluate(pool, c, st, end))
            r.update({f"{p}|score": score(s), f"{p}|beat": s["yrs_beat_SPY_QQQ"], f"{p}|smh3": s["r3_beat_SMH"],
                      f"{p}|dd_ok": s["dd_ok"], f"{p}|cagr": s["med_cagr"], f"{p}|worst": s["worst_dd"], f"{p}|pass": s["PASS"]})
        rows.append(r)
        print(lab, {k: round(v, 3) for k, v in r.items() if k.endswith(("|beat", "|cagr"))}, flush=True)
    df = pd.DataFrame(rows).sort_values("train|score", ascending=False)
    df.to_csv("../results/exp38_core.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    cols = ["label"] + [f"{p}|{k}" for p in PER for k in ("beat", "smh3", "dd_ok", "cagr")]
    print(df[cols].round(3).head(15).to_string(index=False))
