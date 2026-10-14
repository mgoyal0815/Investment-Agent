"""Experiment 51 (owner, 15 Oct 2026: 'check multiple ranges, not one value'): full-range sweeps of every numeric
setting, one at a time, on top of the current best system (System F if round 1 was accepted, else System E).
Each value is scored on the same three start periods as the Bayesian rounds (2012-16 choosing, 2000-08 and
2020-25 checking), so every setting gets a response curve rather than one or two points."""
import json, multiprocessing as mp, os, sys
import numpy as np
import pandas as pd
import yardstick as Y
import exp50_bayes_all as B

PATH = "../data/panel_idx.pkl"
base_file = "../system_F.json" if os.path.exists("../system_F.json") else "../system_E.json"
BASE = json.load(open(base_file))["cfg"]
SW = {"rsi_max (skip buys above this RSI)": ("rsi_max", [None, 95, 90, 85, 82, 80, 78, 75, 72, 70, 65, 60]),
      "rsi_sell_floor (no rank sale below this RSI)": ("rsi_sell_floor", [None, 25, 30, 35, 40, 45, 50]),
      "rsi_take (sell HALF when RSI above, in profit)": ("rsi_take", [None, 95, 92, 90, 88, 85, 82, 80, 75]),
      "rsi_take_all (sell ALL when RSI above, in profit)": ("rsi_take", [None, 95, 92, 90, 88, 85, 82, 80]),
      "adx_min (buy only above this ADX)": ("adx_min", [None, 10, 15, 18, 20, 22, 25, 30, 35]),
      "hi52_min (buy only within x of 52-week high)": ("hi52_min", [None, 0.7, 0.8, 0.85, 0.9, 0.95]),
      "lock_at (profit lock starts at +x)": ("lock_at", [None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]),
      "lock_trail (give-back allowed)": ("lock_trail", [0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.35, 0.4]),
      "rank_exempt_gain (winners above +x ignore rank)": ("rank_exempt_gain", [None, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
      "buffer (sell when outside top N x buffer)": ("buffer", [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]),
      "N (number of slots)": ("N", [6, 8, 9, 10, 11, 12, 14]),
      "add_gain (add after +x)": ("add_gain", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]),
      "add_cap (max slots per stock)": ("add_cap", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
      "add_size (each add, in slots)": ("add_size", [0.25, 0.5, 0.75, 1.0]),
      "breadth_level (pause when breadth below)": ("breadth_level", [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]),
      "gold_slots": ("gold_slots", [0, 1, 2, 3, 4, 5]),
      "vol_target (own-book volatility cap, hard)": ("vol_target", [None, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]),
      "rank_min_hold (sessions)": ("rank_min_hold", [0, 10, 21, 42, 63]),
      "leader_lb (parking lookback)": ("leader_lb", [21, 63, 126, 252])}

if __name__ == "__main__":
    Y._init(PATH)
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 3), "2019-12-31"),
           "early": (Y.starts("2000-01-01", "2008-12-31", 4), "2011-12-30"),
           "late": (Y.starts("2020-01-01", "2025-10-31", 3), None)}
    rows = []
    base_r = B.evaluate(pool, BASE, PER)
    rows.append(dict(setting="BASE", value=json.dumps(None), **base_r))
    for lab, (key, vals) in SW.items():
        for v in vals:
            c = dict(BASE, **{key: v})
            if key == "vol_target" and v:
                c["vol_hard"] = True
            if lab.startswith("rsi_take_all") and v:
                c["rsi_take_frac"] = 1.0
            r = B.evaluate(pool, c, PER)
            rows.append(dict(setting=lab, value=json.dumps(v), is_base=json.dumps(v) == json.dumps(BASE.get(key)), **r))
        pd.DataFrame(rows).to_csv("../results/exp51_sweeps.csv", index=False)
        print("swept", lab, flush=True)
    print("DONE", base_file, flush=True)
