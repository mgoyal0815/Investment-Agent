"""Close-out numbers for System E (clean, point-in-time signals): one run from 2000 vs SPY/QQQ/SMH and live v5,
year table, and the every-start scorecard by era. Writes results/final_<X>_*.csv/json.

    python3 final_R.py [X]        (X = R by default; R1 = System R with the deterministic tie-break, 17 Oct 2026)"""
import json, multiprocessing as mp, sys
import pandas as pd
import engine as X, metrics as M, yardstick as Y
from exp24_timing import mat

if __name__ == "__main__":
    PATH = "../data/panel_ndx.pkl"
    X_ = sys.argv[1] if len(sys.argv) > 1 else "R"
    LAB = f"System {X_}"
    Y._init(PATH)
    m = Y.MM
    E = json.load(open(f"../system_{X_}.json"))["cfg"]
    V5 = dict(universe="sp500")
    rE = X.run(m, mat(E), start="2000-01-03", end="2025-12-31")["curve"]
    rV = X.run(m, mat(V5), start="2000-01-03", end="2025-12-31")["curve"]
    b = M.bench_curves(m, rE.index)
    curves = pd.DataFrame({LAB: rE / rE.iloc[0] * 10000, "live v5": rV.reindex(rE.index) / rV.iloc[0] * 10000,
                           **{k: v / v.dropna().iloc[0] * 10000 for k, v in b.items()}})
    curves.to_csv(f"../results/final_{X_}_curves.csv")
    Yt = pd.DataFrame({k: M.yearly(curves[k].dropna()) * 100 for k in curves}).round(1)
    Yt.to_csv(f"../results/final_{X_}_years.csv")
    summ = {k: dict(cagr=M.cagr(curves[k].dropna()), maxdd=M.maxdd(curves[k].dropna()), end=float(curves[k].dropna().iloc[-1])) for k in curves}
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    rows = []
    for lab, c in ((LAB, E), ("live v5", V5)):
        T = Y.evaluate(pool, c, Y.starts())
        for era, (a, bb) in {"starts 2000-11": (2000, 2011), "starts 2012-19": (2012, 2019), "starts 2020-24": (2020, 2024), "every start": (2000, 2024)}.items():
            s = Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= bb)])
            rows.append(dict(system=lab, era=era, beat=s["yrs_beat_SPY_QQQ"], cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], ddok=s["dd_ok"], cagr=s["med_cagr"], worst=s["worst_dd"]))
    pd.DataFrame(rows).to_csv(f"../results/final_{X_}_scorecard.csv", index=False)
    summ["_env"] = Y.ENV                    # REBUILD.md: versions and panel fingerprint behind these numbers
    summ["_note"] = "curves are rescaled so the first close = $10,000; the account's own end value is curve[-1] * c[0] / 10000"
    json.dump(summ, open(f"../results/final_{X_}_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1)); print(pd.DataFrame(rows).round(3).to_string(index=False)); print(Yt.to_string())
