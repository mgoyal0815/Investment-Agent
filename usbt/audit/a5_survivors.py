"""Audit 5: rerun G and H on a panel that puts back the S&P members the future-looking split-span gate removed
(Signature Bank, Sovereign Bancorp, Rite Aid, Tupperware), every monthly start 2000-2025, and compare."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, engine as X
from exp24_timing import mat
rows = []
for name in ("panel", "panel_audit"):
    PATH = f"../data/{name}.pkl"; Y._init(PATH)
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    for lab in ("G", "H"):
        c = json.load(open(f"../system_{lab}.json"))["cfg"]
        T = Y.evaluate(pool, c, Y.starts())
        for era, (a, b) in {"2000-11": (2000, 2011), "2012-19": (2012, 2019), "2020-25": (2020, 2025), "all": (2000, 2025)}.items():
            s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])
            rows.append(dict(panel=name, system=lab, era=era, beat=s["yrs_beat_SPY_QQQ"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]))
        r = X.run(Y.MM, mat(c), start="2000-01-03", record=True)
        held = sorted(set(r["lots"].sym) & {"SBNY", "SOV", "RAD", "TUP"})
        print(name, lab, "restored stocks it bought:", held, flush=True)
    pool.close(); pool.join()
R = pd.DataFrame(rows); R.to_csv("../audit/a5_survivors.csv", index=False)
print(R.pivot_table(index=["system", "era"], columns="panel", values=["beat", "ddok", "worst"]).round(3).to_string())
