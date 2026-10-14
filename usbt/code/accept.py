"""Acceptance test for the iterative Bayesian protocol (owner, 15 Oct 2026). A candidate replaces the incumbent
only if, from every monthly start 2000-2025 (first session of the month):
  1. years beating SPY and QQQ improve by >= 2 points overall (beyond the ~2-point tie-break noise),
  2. no start era (2000-11, 2012-19, 2020-25) is more than 1 point worse,
  3. the drawdown record is no worse: drawdown-ok share not lower, worst fall not more than 3 points deeper,
  4. it also improves (>= 1 point, drawdown-ok not lower) on the held-back starts (15th of each month).
usage: python3 accept.py incumbent.json candidate.json [panel]"""
import json, multiprocessing as mp, sys
import pandas as pd
import yardstick as Y


def card(pool, c, day):
    T = Y.evaluate(pool, c, Y.starts(day=day))
    out = {}
    for era, (a, b) in {"2000-11": (2000, 2011), "2012-19": (2012, 2019), "2020-25": (2020, 2025), "all": (2000, 2025)}.items():
        s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
        out.update({f"{era}|beat": s["yrs_beat_SPY_QQQ"], f"{era}|ddok": s["dd_ok"], f"{era}|cagr": s["med_cagr"], f"{era}|worst": s["worst_dd"]})
    return out


def decide(pool, inc, cand):
    I1, C1 = card(pool, inc, 1), card(pool, cand, 1)
    I15, C15 = card(pool, inc, 15), card(pool, cand, 15)
    checks = {
        "overall +2 pts": C1["all|beat"] - I1["all|beat"] >= 0.02,
        "no era -1 pt": all(C1[f"{e}|beat"] - I1[f"{e}|beat"] >= -0.01 for e in ("2000-11", "2012-19", "2020-25")),
        "drawdown-ok not lower": C1["all|ddok"] >= I1["all|ddok"] - 1e-9,
        "worst fall within 3 pts": C1["all|worst"] >= I1["all|worst"] - 0.03,
        "held-back starts +1 pt": C15["all|beat"] - I15["all|beat"] >= 0.01 and C15["all|ddok"] >= I15["all|ddok"] - 1e-9}
    tab = pd.DataFrame({"incumbent": I1, "candidate": C1, "incumbent (15th)": I15, "candidate (15th)": C15})
    return all(checks.values()), checks, tab


if __name__ == "__main__":
    path = sys.argv[3] if len(sys.argv) > 3 else "../data/panel_idx.pkl"
    Y._init(path)
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(path,))
    inc, cand = (json.load(open(f))["cfg"] for f in sys.argv[1:3])
    ok, checks, tab = decide(pool, inc, cand)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string()); print(checks); print("ACCEPT" if ok else "REJECT", flush=True)
