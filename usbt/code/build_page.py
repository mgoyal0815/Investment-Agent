"""Build usbt/report.html (the shareable page) from results/. python build_page.py"""
import json
import pandas as pd

R = "../results"
SYS = [("US v5 (live)", "v5 live"), ("Candidate 3", "Candidate 3"), ("Local-search best", "Hybrid A"),
       ("India v8 on US", "India v8")]
cur = json.load(open(f"{R}/curves.json"))
data = {"curves": {}, "years": [], "windows": []}
for run in ("2020", "2012", "2000"):
    series = []
    for lab, short in SYS:
        c = cur[f"{lab}|{run}"]
        series.append({"name": short, "kind": "sys", "d": c["dates"], "v": c["nav"]})
    for b in ("SPY", "QQQ", "SMH"):
        c = cur[f"{b}|{run}"]
        series.append({"name": b, "kind": "idx", "d": c["dates"], "v": c["nav"]})
    data["curves"][run] = series
y = pd.read_csv(f"{R}/exp3_years.csv")
y = y.rename(columns={y.columns[0]: "year"})
w = y.pivot_table(index=["run", "year"], columns="label", values="sys")
b = y.groupby(["run", "year"])[["SPY", "QQQ", "SMH"]].first()
for (run, yr), row in pd.concat([w, b], axis=1).iterrows():
    if not ((run == 2000) or (run == 2012 and yr <= 2019) or (run == 2020)):
        continue
    data["years"].append({"year": int(yr), "partial": int(yr) == 2026,
                          **{short: round(float(row[lab]) * 100, 1) for lab, short in SYS},
                          **{k: round(float(row[k]) * 100, 1) for k in ("SPY", "QQQ", "SMH")}})
p = pd.read_csv(f"{R}/exp3_periods.csv")
for lab, short in SYS:
    for per in ("train", "test", "early"):
        r = p[(p.label == lab) & (p.period == per)].iloc[0]
        data["windows"].append({"sys": short, "per": per, "cagr": round(r.cagr * 100, 1), "dd": round(r.maxdd * 100, 0),
                                "ba": round(r.beat_all * 100), "ddidx": round(r.dd_idx * 100, 0)})
tpl = open("page_template.html").read()
open("../report.html", "w").write(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
print("written", len(json.dumps(data)) // 1024, "KB")
