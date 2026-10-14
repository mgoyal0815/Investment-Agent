"""Experiment 42: technical indicators on top of System D (owner, 15 Oct 2026): more stocks (N), entry filters
(ADX trend strength, volatility cap, near 52-week high, MACD, RSI not overbought, recent 55-day breakout), score blends
(52-week high, residual momentum, ADX, RSI), exits (keep while above a rising 50-day EMA; chandelier ATR
stop), market pause votes with VIX, VIX spike, SPY/QQQ MACD, credit (HYG/IEF), 2-year momentum-crash risk;
breadth-thrust restart; narrow-market tilt to the biggest companies. One change at a time, then coordinate ascent,
chosen on starts 2012-2016 stopping 2019 (yardstick score); judged on starts 2000-2008, 2020-2025, all."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat
from exp34_yard_search import score

SD = json.load(open("../system_D.json"))["cfg"]
SPACE = dict(N=[8, 10, 12, 15, 20, 30], adx_min=[None, 15, 20, 25], hi52_min=[None, 0.8, 0.9], macd_pos=[None, True],
             rsi_max=[None, 75, 80, 85], bo_recent=[None, 20, 60],
             score_mix=[None, {"hi52": 0.25}, {"resmom": 0.25}, {"resmom": 0.5}, {"ADX": 0.25}, {"RSI": 0.25}],
             trend_hold=[None, True, 0.0, 0.2, 0.5], chand_k=[None, 3, 4, 5],
             pause_rules=[("breadth", "hyg", "large"), ("breadth", "hyg", "large", "vix30"), ("breadth", "hyg", "large", "vix35"),
                          ("hyg", "large", "vix30"), ("breadth", "hyg", "large", "spymacd"), ("breadth", "hyg", "large", "credit"),
                          ("breadth", "hyg", "large", "vixrel1.3"), ("breadth", "hyg", "large", "momcrash"), ("breadth", "large", "credit")],
             rank_exempt_gain=[None, 0.5, 1.0], buffer=[1.5, 2.0, 3.0],
             narrow_size_w=[None, 0.25, 0.5, 1.0], thrust_days=[None, 21, 63], vol_max_pct=[None, 0.8, 0.9])
PER = None


def ev(pool, c, p):
    st, end = PER[p]
    return Y.summary(Y.evaluate(pool, c, st, end))


def yrs(c):
    rr = X.run(Y.MM, mat(c), start="2000-01-03")
    y = M.yearly(rr["curve"])
    return {str(k): y.get(k) for k in (2008, 2011, 2019, 2020, 2022, 2023, 2024)} | dict(cagr2000=M.cagr(rr["curve"]), dd2000=M.maxdd(rr["curve"]))


if __name__ == "__main__":
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    PER = {"train": (Y.starts("2012-01-01", "2016-12-31", 2), "2019-12-31"),
           "test 2000-08": (Y.starts("2000-01-01", "2008-12-31", 2), "2011-12-30"),
           "test 2020-25": (Y.starts("2020-01-01", "2025-10-31", 1), None), "all starts": (Y.starts(), None)}
    base = score(ev(pool, SD, "train"))
    print("System D train", round(base, 3), flush=True)
    single = []
    for k, vals in SPACE.items():
        for v in vals:
            if v == SD.get(k) or (v is None and SD.get(k) is None):
                continue
            c = dict(SD, **{k: v})
            s = ev(pool, c, "train")
            single.append(dict(key=k, val=json.dumps(v), score=score(s), beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"]))
            print("single", k, v, round(score(s), 3), round(s["yrs_beat_SPY_QQQ"], 3), round(s["dd_ok"], 2), flush=True)
            pd.DataFrame(single).to_csv("../results/exp42_single.csv", index=False)
    cur, best = dict(SD), base
    for p in range(2):
        imp = False
        for k, vals in SPACE.items():
            for v in vals:
                if v == cur.get(k):
                    continue
                c = dict(cur, **{k: v})
                sc = score(ev(pool, c, "train"))
                if sc > best + 0.005:
                    best, cur, imp = sc, c, True
                    print("  better", k, v, round(best, 3), flush=True)
        if not imp:
            break
    json.dump(cur, open("../results/exp42_best_cfg.json", "w"), indent=1, default=list)
    # finalists: System D, the tuned one, and each single change that beat System D on training
    sd = pd.DataFrame(single).sort_values("score", ascending=False)
    fin = [("System D", SD), ("exp42 tuned", cur)]
    for _, r in sd[sd.score > base + 0.005].head(6).iterrows():
        fin.append((f"D + {r.key}={r.val}", dict(SD, **{r.key: json.loads(r.val)})))
    rows = []
    for lab, c in fin:
        r = {"label": lab}
        for p in PER:
            s = ev(pool, c, p)
            r.update({f"{p}|beat": s["yrs_beat_SPY_QQQ"], f"{p}|smh3": s["r3_beat_SMH"], f"{p}|ddok": s["dd_ok"], f"{p}|cagr": s["med_cagr"],
                      f"{p}|worst": s["worst_dd"], f"{p}|score": score(s)})
        r.update(yrs(c))
        rows.append(r)
        print("final", lab, {k: round(v, 3) for k, v in r.items() if k != "label" and v is not None}, flush=True)
        pd.DataFrame(rows).to_csv("../results/exp42_final.csv", index=False)
