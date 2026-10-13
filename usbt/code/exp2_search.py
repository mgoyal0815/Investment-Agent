"""Experiment 2: combination search on the TRAINING window only (2012-2019, five start dates).

Random sample of the combined v5 x India design space, plus a few levers aimed at the SMH problem
(a bigger leader-ETF sleeve, idle cash in the leader). Every configuration is scored with
runner.objective. The top 25 by objective go to results/exp2_top.csv; all to exp2_all.csv.
Nothing from 2020 on or before 2012 is looked at here.
"""
import json
import random
import sys
import pandas as pd
import runner as R

LEADER_HEAVY = dict(FULL_BULL=(0.60, 0.40), CREDIT_WARN=(0.50, 0.30, 0.20), MILD_BEAR=(0.30,), DEEP_BEAR=(0.40,))
LEADER_MID = dict(FULL_BULL=(0.75, 0.25), CREDIT_WARN=(0.60, 0.20, 0.20), MILD_BEAR=(0.30,), DEEP_BEAR=(0.40,))
FULL_STOCK = dict(FULL_BULL=(1.00, 0.0), CREDIT_WARN=(0.80, 0.0, 0.20), MILD_BEAR=(0.30,), DEEP_BEAR=(0.40,))
SPACE_SLOT = dict(
    N=[5, 6, 8, 10, 12, 15, 20], score=["nifty", "robust"], trend=[True, False], exit=["rank", "stop+rank"],
    buffer=[1.0, 1.5, 2.0, 3.0], adds=["india", None], cadence=["twice", "monthly", "daily"],
    overlay=["pause", "v5", "v5_soft", "none"], pause_rules=[("breadth", "hyg", "large"), ("hyg",), ("large",), ("breadth", "large")],
    intramonth=[True, False], park=[None, "GLD", "SHY", "leader"], gold_slots=[0, 2, 4],
    universe_n=[100, 200, 300, 500, 1000], ramp=[0, 3], zone_gate=[False, True],
    regime_w=[None, "LEADER_MID", "LEADER_HEAVY", "FULL_STOCK"], rank_min_hold=[0, 21, 63],
)
SPACE_V5 = dict(
    universe=["adv50", "topN"], universe_n=[200, 300, 500, 1000], overlay=["v5", "pause", "none", "v5_soft"],
    intramonth=[True, False], zone_gate=[True, False], cadence=["daily", "twice"], park=[None, "GLD", "SHY", "leader"],
    adds=["v5", None], exit=["stop", "stop+rank"], min_hold=[88, 44, 21], cont_cut=[0.88, 0.95], rec_cut=[0.83, 0.95],
    regime_w=[None, "LEADER_MID", "LEADER_HEAVY", "FULL_STOCK"], ramp=[0, 3],
)
W = dict(LEADER_HEAVY=LEADER_HEAVY, LEADER_MID=LEADER_MID, FULL_STOCK=FULL_STOCK)


def sample(rng):
    if rng.random() < 0.6:
        c = dict(sizing="slot", universe="topN")
        for k, v in SPACE_SLOT.items():
            c[k] = rng.choice(v)
    else:
        c = dict(sizing="v5")
        for k, v in SPACE_V5.items():
            c[k] = rng.choice(v)
        if c["exit"] == "stop+rank":
            c["N"] = rng.choice([15, 20, 30])
    return c


def materialise(c):
    c = dict(c)
    if isinstance(c.get("regime_w"), str):
        c["regime_w"] = W[c["regime_w"]]
    if c.get("sizing") == "slot":
        c["min_hold"] = c.get("rank_min_hold", 0)
    return c


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rng = random.Random(7)
    raw = [sample(rng) for _ in range(n)]
    seed = [dict(R.V5), dict(R.INDIA)]
    raw = seed + raw
    rn = R.Runner()
    res = []
    B = 40
    for k in range(0, len(raw), B):
        chunk = raw[k:k + B]
        out = rn.eval_many([materialise(c) for c in chunk], "train")
        for c, a in zip(chunk, out):
            res.append(dict(cfg=json.dumps(c, default=list), objective=R.objective(a), **a))
        df = pd.DataFrame(res).sort_values("objective", ascending=False)
        df.to_csv("../results/exp2_all.csv", index=False)
        print(k + B, "done; best:", flush=True)
        for _, r in df.head(3).iterrows():
            print("  ", round(r.objective, 3), R.fmt("", r), flush=True)
    df.head(25).to_csv("../results/exp2_top.csv", index=False)
