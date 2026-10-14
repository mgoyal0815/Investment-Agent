"""Rebuild usbt/report.html for the frozen System A against live v5 and India v8 on US data."""
import json
import pandas as pd
import engine as X, metrics as M, runner as R
from exp2_search import materialise

A = json.load(open("../system_A.json"))
SYS = [("v5 live", {}), ("System A", A["cfg"]), ("India v8", R.INDIA)]
if __name__ == "__main__":
    m = X.Market(pd.read_pickle("../data/panel.pkl"))
    data = {"curves": {}, "years": [], "windows": []}
    for run, s, e in (("2020", "2020-01-02", None), ("2012", "2012-01-03", None), ("2000", "2000-01-03", None)):
        ser = []
        for lab, c in SYS:
            cv = X.run(m, materialise(c), start=s, end=e)["curve"]
            ser.append({"name": lab, "kind": "sys", "d": [str(d.date()) for d in cv.index[::5]], "v": [round(float(v), 2) for v in cv.values[::5]]})
        for k, v in M.bench_curves(m, cv.index).items():
            v = v / v.dropna().iloc[0] * 10000
            ser.append({"name": k, "kind": "idx", "d": [str(d.date()) for d in v.index[::5]], "v": [round(float(x), 2) for x in v.values[::5]]})
        data["curves"][run] = ser
    spans = (("2000-01-03", "2011-12-30"), ("2012-01-03", "2019-12-31"), ("2020-01-02", None))
    cols = {lab: pd.concat([M.yearly(X.run(m, materialise(c), start=s, end=e)["curve"]) for s, e in spans]) * 100 for lab, c in SYS}
    full = m.dates[m.dates >= "2000-01-03"]
    for k, v in M.bench_curves(m, full).items():
        cols[k] = pd.concat([M.yearly(v[s:e]) for s, e in spans]) * 100
    t = pd.DataFrame(cols)
    for yr, row in t.iterrows():
        data["years"].append({"year": int(yr), "partial": int(yr) == 2026, **{k: round(float(v), 1) for k, v in row.items()}})
    rn = R.Runner()
    for per in ("train", "test", "early"):
        for (lab, c), a in zip(SYS, rn.eval_many([materialise(c) for _, c in SYS], per)):
            data["windows"].append({"sys": lab, "per": per, "cagr": round(a["cagr"] * 100, 1), "dd": round(a["maxdd"] * 100, 0),
                                    "ba": round(a["beat_all"] * 100), "ddidx": round(a["dd_idx"] * 100, 0)})
    tpl = open("page_template_A.html").read()
    open("../report.html", "w").write(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    print("written")
