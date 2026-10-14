"""Experiment 47 (audit fix, 15 Oct 2026): the pause's breadth and large-cap signals were computed over the
top-500/100 by turnover of the WHOLE panel, which includes the live list's future survivors (20-50% non-S&P
on the day). Re-run Systems C, D, E with the same signals built from that day's S&P 500 members only."""
import json, multiprocessing as mp
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat

if __name__ == "__main__":
    Y._init("../data/panel.pkl")
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=("../data/panel.pkl",))
    rows, years = [], {}
    for name in ("system_C", "system_D", "system_E"):
        base = json.load(open(f"../{name}.json"))["cfg"]
        for tag, c in (("old signals", base), ("point-in-time signals", dict(base, pit_signals=True))):
            T = Y.evaluate(pool, c, Y.starts())
            r = {"label": f"{name} - {tag}"}
            for era, (a, b) in {"2000-11": (2000, 2011), "2012-19": (2012, 2019), "2020-25": (2020, 2025), "all": (2000, 2025)}.items():
                s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
                r.update({f"{era}|beat": s["yrs_beat_SPY_QQQ"], f"{era}|ddok": s["dd_ok"]})
            s = Y.summary(T)
            r.update(cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], medcagr=s["med_cagr"], worst=s["worst_dd"])
            rr = X.run(Y.MM, mat(c), start="2000-01-03", record=True)
            d = rr["daily"]
            r.update(cagr2000=M.cagr(rr["curve"]), dd2000=M.maxdd(rr["curve"]), paused=d.paused.mean(), in_stocks=(d.stocks / d.nav).mean())
            years[r["label"]] = (M.yearly(rr["curve"]) * 100).round(1)
            rows.append(r); print(r["label"], flush=True)
    pd.DataFrame(rows).to_csv("../results/exp47_pit_signals.csv", index=False)
    pd.DataFrame(years).to_csv("../results/exp47_years.csv")
    pd.set_option("display.width", 320, "display.max_columns", 40)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(years).to_string())
