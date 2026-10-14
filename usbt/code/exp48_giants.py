"""Experiment 48 (owner, 15 Oct 2026: 'capture the giants, Treasuries are not the answer'; no inverse ETFs):
1) index ETFs (QQQ, SPY, DIA, IWM, SMH, sector ETFs) compete with stocks for System E's slots;
2) index hurdle: a stock is bought only if its 6-month return beats the best of SPY/QQQ/SMH, otherwise the
   money stays in the leading index (System E already parks idle money in the leading index unless paused);
3) market volatility targeting (exposure = target / SPY 63-day volatility) instead of, or with, the on/off pause.
System E with point-in-time signals; panel_idx.pkl (index ETFs as columns). Every monthly start 2000-2025."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat

E = json.load(open("../system_E.json"))["cfg"]
NOP = dict(pause_rules=(), pause_k=None)
G = [("System E (clean)", E),
     ("E + index ETFs compete for slots", dict(E, etf_in_universe=True)),
     ("E + index hurdle", dict(E, index_hurdle=True)),
     ("E + ETFs compete + hurdle", dict(E, etf_in_universe=True, index_hurdle=True)),
     ("E + market vol target 15%", dict(E, mkt_vol_target=0.15)),
     ("E + market vol target 20%", dict(E, mkt_vol_target=0.20)),
     ("E + market vol target 15%, hard", dict(E, mkt_vol_target=0.15, vol_hard=True)),
     ("E, no pause, market vol target 15% hard", dict(E, mkt_vol_target=0.15, vol_hard=True, **NOP)),
     ("E, no pause, market vol target 20% hard", dict(E, mkt_vol_target=0.20, vol_hard=True, **NOP)),
     ("E + ETFs + hurdle + vol target 20%", dict(E, etf_in_universe=True, index_hurdle=True, mkt_vol_target=0.20)),
     ("E + ETFs + hurdle, no pause, vol target 15% hard", dict(E, etf_in_universe=True, index_hurdle=True, mkt_vol_target=0.15, vol_hard=True, **NOP))]

if __name__ == "__main__":
    PATH = "../data/panel_idx.pkl"
    Y._init(PATH)
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    rows = []
    for lab, c in G:
        T = Y.evaluate(pool, c, Y.starts())
        r = {"label": lab}
        for era, (a, b) in {"2000-11": (2000, 2011), "2012-19": (2012, 2019), "2020-25": (2020, 2025), "all": (2000, 2025)}.items():
            s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
            r.update({f"{era}|beat": s["yrs_beat_SPY_QQQ"], f"{era}|ddok": s["dd_ok"]})
        s = Y.summary(T)
        r.update(cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], medcagr=s["med_cagr"], worst=s["worst_dd"])
        rr = X.run(Y.MM, mat(c), start="2000-01-03", record=True)
        d, y = rr["daily"], M.yearly(rr["curve"])
        tr = rr["trades"]
        etf_buys = int(sum(1 for t in tr.itertuples() if t[2] in ("QQQ", "SPY", "DIA", "IWM", "SMH", "SOXX", "XLK", "IGV") and t[3] == "BUY")) if len(tr) else 0
        r.update(cagr2000=M.cagr(rr["curve"]), dd2000=M.maxdd(rr["curve"]), in_stocks=(d.stocks / d.nav).mean(), etf_buys=etf_buys,
                 **{str(k): y[k] for k in (2002, 2008, 2009, 2019, 2020, 2022, 2023, 2024)})
        rows.append(r); print(lab, flush=True)
        pd.DataFrame(rows).to_csv("../results/exp48_giants.csv", index=False)
    print("DONE", flush=True)
