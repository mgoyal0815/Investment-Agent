"""Experiment helpers: named configurations, periods, multi-start evaluation in parallel."""
import multiprocessing as mp
import numpy as np
import pandas as pd

import engine as X
import metrics as M

M_ = None

V5 = {}
INDIA = dict(universe="topN", universe_n=500, score="nifty", trend=True, zone_gate=False, sizing="slot", N=10,
             exit="rank", buffer=1.5, min_hold=0, adds="india", cadence="twice", overlay="pause", intramonth=False,
             park="GLD", gold_slots=4, ramp=0, adv_cap=0.01)

# the 15 differences, as (label, v5 -> India change). 15 (tax-aware sell delay) has no US-tax meaning: not modelled.
SWITCHES = [
    ("1-2 hold 10, equal slots", dict(sizing="slot", N=10, adds=None)),
    ("3 fully invested (no overlay)", dict(overlay="none", intramonth=False)),
    ("4 India buy pause instead of regimes", dict(overlay="pause", intramonth=False)),
    ("5 no MTD exit / daily re-entry", dict(intramonth=False)),
    ("6 sell on rank, no stops/hold", dict(exit="rank", min_hold=0, sizing="slot", N=10, adds=None)),
    ("7 no adds (v5) ", dict(adds=None)),
    ("8 India ranking + trend", dict(score="nifty", sizing="slot", N=10, adds=None)),
    ("9 no pullback-band entry", dict(zone_gate=False)),
    ("10 decide twice a month", dict(cadence="twice")),
    ("11 idle cash in GLD", dict(park="GLD")),
    ("12 top-500 universe", dict(universe="topN", universe_n=500)),
    ("13 no ramp-in", dict(ramp=0)),
    ("14 1% ADV order cap", dict(adv_cap=0.01)),
]

PERIODS = {
    "train": ("2012-01-03", "2019-12-31", ["2012-01-03", "2012-04-02", "2012-07-02", "2012-10-01", "2013-01-02"]),
    "test": ("2020-01-02", None, ["2020-01-02", "2020-04-01", "2020-07-01", "2020-10-01", "2021-01-04"]),
    "early": ("2000-01-03", "2011-12-30", ["2000-01-03", "2000-04-03", "2000-07-03", "2000-10-02", "2001-01-02"]),
}


def _init(path):
    global M_
    M_ = X.Market(pd.read_pickle(path))


def _one(args):
    cfg, start, end = args
    r = X.run(M_, cfg, start=start, end=end)
    s = M.score(M_, r["curve"])
    s.pop("table")
    s["expo"] = float(r["expo"].mean())
    s["costs"] = r["costs"]
    return s


class Runner:
    def __init__(self, path="../data/panel.pkl", procs=4):
        self.pool = mp.get_context("fork").Pool(procs, initializer=_init, initargs=(path,))

    def eval_many(self, cfgs, period):
        a, b, starts = PERIODS[period]
        jobs = [(c, s, b) for c in cfgs for s in starts]
        res = self.pool.map(_one, jobs, chunksize=1)
        out = []
        k = len(starts)
        for n, c in enumerate(cfgs):
            rs = res[n * k:(n + 1) * k]
            out.append(agg(rs))
        return out


def agg(rs):
    f = lambda key: [r[key] for r in rs]
    yrs = np.array(f("years"))
    ba = np.array(f("beat_all"))
    return dict(cagr=np.median(f("cagr")), cagr_min=min(f("cagr")), maxdd=min(f("maxdd")),
                dd_idx=min(max(r["dd_SPY"], r["dd_QQQ"], r["dd_SMH"]) for r in rs),
                dd_ok=float(np.mean(f("dd_ok"))), beat_all=float(np.mean(ba / yrs)), beat_all_min=int(ba.min()),
                years=int(yrs.max()), beat_SPY=float(np.mean(np.array(f("beat_SPY")) / yrs)),
                beat_QQQ=float(np.mean(np.array(f("beat_QQQ")) / yrs)), beat_SMH=float(np.mean(np.array(f("beat_SMH")) / yrs)),
                roll_win=float(np.mean(f("roll_win"))), min_year_gap=min(f("min_year_gap")), expo=float(np.mean(f("expo"))))


def objective(a):
    """Train-window selection score: the owner's goal is beating all three every year, with a max
    drawdown smaller than every index's. Drawdown failures are penalised hard."""
    return a["beat_all"] + 0.25 * a["roll_win"] + 0.1 * a["beat_SPY"] - 1.0 * (1 - a["dd_ok"]) + 0.5 * a["cagr"]


def objective2(a):
    """Owner's rule from 14 Oct 2026: growth high, drawdown low, and beating the indexes.
    1 point less drawdown is worth half a point of return."""
    return a["cagr"] + 0.5 * a["maxdd"] + 0.1 * a["beat_all"]


def fmt(lab, a):
    return (f"{lab:44s} CAGR {a['cagr']:6.1%} (min {a['cagr_min']:6.1%})  DD {a['maxdd']:6.1%} vs idx {a['dd_idx']:6.1%} ok {a['dd_ok']:.0%}"
            f"  beatAll {a['beat_all']:.0%} (min {a['beat_all_min']}/{a['years']})  SPY {a['beat_SPY']:.0%} QQQ {a['beat_QQQ']:.0%}"
            f" SMH {a['beat_SMH']:.0%}  roll12 {a['roll_win']:.0%}  expo {a['expo']:.0%}")
