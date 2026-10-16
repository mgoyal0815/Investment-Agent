"""K on the S&P 500 + Nasdaq-100 universe (point-in-time Nasdaq-100 membership from 2004), standard data rules.
Every second monthly start 2000-2025 on panel_ndx; K (S&P 500 only, same panel) vs K with Nasdaq-100 members added,
and the wider universe with 8 or 12 holdings."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y
PATH = "../data/panel_ndx.pkl"; Y._init(PATH)
pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
K = json.load(open("../system_K.json"))["cfg"]; ST = Y.starts(step=2)
rows = []
for lab, c in [("K, S&P 500 only", K), ("K, S&P 500 + Nasdaq-100", dict(K, universe="sp500+ndx")),
               ("K + Nasdaq-100, 8 holdings", dict(K, universe="sp500+ndx", N=8)), ("K + Nasdaq-100, 12 holdings", dict(K, universe="sp500+ndx", N=12))]:
    T = Y.evaluate(pool, c, ST)
    for era, (a, b) in {"all": (2000, 2025), "2000-11": (2000, 2011), "2012-19": (2012, 2019), "2020-25": (2020, 2025)}.items():
        s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
        rows.append(dict(test=lab, era=era, beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]))
    print(rows[-4], flush=True)
R = pd.DataFrame(rows); R.to_csv("../audit/a20_ndx.csv", index=False)
print(R.pivot_table(index="test", columns="era", values="beat").round(3).to_string()); print(R[R.era == "all"].round(3).to_string(index=False))
