"""Experiment 1: baselines and every difference one at a time, in both directions, on the
training window (2012-2018, five start dates). Writes results/exp1_switches.csv."""
import pandas as pd
import runner as R

if __name__ == "__main__":
    rn = R.Runner()
    rows = []
    cfgs = [("US v5 (live)", R.V5), ("India v8 on US", R.INDIA)]
    for lab, ch in R.SWITCHES:
        cfgs.append(("v5 + " + lab, dict(R.V5, **ch)))
    # reverse: India with one thing put back to v5
    back = [
        ("India + v5 regime overlay", dict(overlay="v5", intramonth=True)),
        ("India + v5 stops", dict(exit="stop+rank", min_hold=88)),
        ("India + v5 ranking (RobustPair)", dict(score="robust")),
        ("India + pullback-band entry", dict(zone_gate=True)),
        ("India decide daily", dict(cadence="daily")),
        ("India, no gold", dict(park=None, gold_slots=0)),
        ("India, idle cash no gold slots", dict(gold_slots=0)),
        ("India + $50m-ADV universe", dict(universe="adv50")),
        ("India + 3-month ramp", dict(ramp=3)),
        ("India no adds", dict(adds=None)),
        ("India no pause", dict(overlay="none")),
        ("India 29 slots", dict(N=29)),
    ]
    for lab, ch in back:
        cfgs.append((lab, dict(R.INDIA, **ch)))
    for period in ("train",):
        res = rn.eval_many([c for _, c in cfgs], period)
        for (lab, c), a in zip(cfgs, res):
            print(R.fmt(lab, a), flush=True)
            rows.append(dict(period=period, label=lab, objective=R.objective(a), **a))
    pd.DataFrame(rows).to_csv("../results/exp1_switches.csv", index=False)
