"""Experiment 43 (owner, 15 Oct 2026): which enhancements matter, alone and together, judged by a Bayesian
model instead of one-at-a-time tests.

Phase 1  random factorial design: N configurations, every enhancement switched on/off (or to one of a few
         levels) at random on top of System D. Each one is scored, fresh $10,000 from many start dates, on
         three separate start periods: 2012-2016 starts (stop 2019) = TRAIN, 2000-2008 starts (stop 2011)
         and 2020-2025 starts (to today) = CHECK. Target: share of years (from each start) beating BOTH
         SPY and QQQ; also drawdown-ok share, growth, SMH 3-year windows.
Phase 2  Bayesian ridge regression (ARD-style shrinkage) of the target on every enhancement and every
         pairwise interaction, fitted separately on each period. An enhancement "matters" only if its
         effect is credibly non-zero on TRAIN and has the same sign on both CHECK periods.
Phase 3  Bayesian optimisation (Optuna TPE) on TRAIN only, starting from the best factorial points; the
         final picks are then judged on the CHECK periods and on every monthly start 2000-2025.
"""
import itertools, json, multiprocessing as mp, random, sys
import numpy as np
import pandas as pd
import yardstick as Y

SD = json.load(open("../system_D.json"))["cfg"]
BASE_PAUSE = ("breadth", "hyg", "large")
F = dict(trend_hold=[None, True], lock_trend=[None, True], rsi_max=[None, 75, 80], adx_min=[None, 20], macd_pos=[None, True],
         hi52_min=[None, 0.9], vol_max_pct=[None, 0.8], bo_recent=[None, 60], score_mix=[None, "resmom", "hi52"],
         rank_exempt_gain=[1.0, None], rsi_sell_floor=[None, 40], narrow_size_w=[None, 0.5], thrust_days=[None, 63],
         chand_k=[None, 4], N=[10, 12, 15], buffer=[1.5, 2.0], lock_at=[0.3, 0.5, None], rank_ladder=[None, 2],
         lock_half=[None, True], weight=[None, "invvol"], add_cap=[3.0, 4.0], pause_needs_bear=[None, True],
         p_credit=[False, True], p_vix30=[False, True], p_vixrel=[False, True], p_momcrash=[False, True], p_spymacd=[False, True])
TOK = dict(p_credit="credit", p_vix30="vix30", p_vixrel="vixrel1.3", p_momcrash="momcrash", p_spymacd="spymacd")


def build(x):
    """x: dict factor -> level. Returns an engine config."""
    c = dict(SD)
    for k, v in x.items():
        if k in TOK:
            continue
        if k == "score_mix":
            c[k] = None if v is None else {v: 0.25}
        elif k == "lock_half":
            c[k] = v
            if v:
                c["lock_trail2"] = 0.4
        else:
            c[k] = v
    extra = tuple(TOK[k] for k in TOK if x.get(k))
    c["pause_rules"] = BASE_PAUSE + extra
    return c


PER = None


def evaluate(pool, c):
    out = {}
    for p, (st, e) in PER.items():
        s = Y.summary(Y.evaluate(pool, c, st, e))
        out.update({f"{p}|beat": s["yrs_beat_SPY_QQQ"], f"{p}|ddok": s["dd_ok"], f"{p}|cagr": s["med_cagr"],
                    f"{p}|smh3": s["r3_beat_SMH"], f"{p}|worst": s["worst_dd"]})
    return out


def obj(r, p="train"):
    """Owner's target: years beating SPY and QQQ, with the drawdown rule as a hard-ish constraint."""
    return r[f"{p}|beat"] - 0.5 * (1 - r[f"{p}|ddok"])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 2), "2019-12-31"),
           "early": (Y.starts("2000-01-01", "2008-12-31", 3), "2011-12-30"),
           "late": (Y.starts("2020-01-01", "2025-10-31", 2), None)}
    rng = random.Random(43)
    rows = []
    design = [{k: v[0] for k, v in F.items()}] + [{k: rng.choice(v) for k, v in F.items()} for _ in range(n)]
    for k, x in enumerate(design):
        r = evaluate(pool, build(x))
        rows.append({**{f"f:{a}": json.dumps(b) for a, b in x.items()}, **r})
        if (k + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv("../results/exp43_design.csv", index=False)
            d = pd.DataFrame(rows)
            print(k + 1, "best train beat", round(d["train|beat"].max(), 3), flush=True)
    pd.DataFrame(rows).to_csv("../results/exp43_design.csv", index=False)
    print("PHASE1_DONE", flush=True)
