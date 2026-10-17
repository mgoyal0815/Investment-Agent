"""System R1 in 2026 (owner, 17 Oct 2026): $10,000 and $1,000, whole shares (the live rule) and fractional shares
(stocks and ETFs, 6 decimals). Start 2026-01-02, end at the last session in the panel. Same cost model in every run
($0.005 a share + 0.10%, minimum $1 a trade). Live panel by default (/home/claude/r/data, to 2026-10-16).

    python3 a72_2026_sizes.py [panel dir]"""
import sys, os, json, hashlib
D = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/r/data"
CODE = "/home/user/Investment-Agent/usbt/code"
sys.path.insert(0, CODE)
os.chdir(os.path.join(os.path.dirname(D.rstrip("/")), "code") if os.path.isdir(os.path.join(os.path.dirname(D.rstrip("/")), "code")) else CODE)
import numpy as np, pandas as pd, engine as X


def mat(c):                       # same as exp24_timing.mat (rconf.materialise)
    c = dict(c)
    if c.get("sizing") == "slot":
        c["min_hold"] = c.get("rank_min_hold", 0)
    if c.get("lock_at") is None:
        c.pop("lock_at", None)
    return c


m = X.Market(pd.read_pickle(os.path.join(D, "panel.pkl")))
R1 = json.load(open(os.path.join(CODE, "..", "system_R1.json")))["cfg"]
start, end = "2026-01-02", str(m.dates[-1].date())
rows, trades = [], {}
for cap in (10000.0, 1000.0):
    for frac in (False, True):
        c = dict(R1, capital=cap, fractional=frac, whole_etf=not frac)
        r = X.run(m, mat(c), start=start, end=end, record=True)
        cv = r["curve"]; t = r["trades"]
        lab = f"${cap:,.0f} {'fractional' if frac else 'whole shares'}"
        rows.append(dict(run=lab, start_value=cap, end_value=round(float(cv.iloc[-1]), 2), ret=float(cv.iloc[-1] / cap - 1),
                         worst=float((cv / cv.cummax() - 1).min()), trades=len(t), costs=round(float(r["costs"]), 2),
                         cost_pct=float(r["costs"] / cap), holdings_end=len(r["book"]["positions"]),
                         cash_end=round(float(r["book"]["cash"]), 2)))
        trades[lab] = t
B = {k: pd.Series(m.E[k]["c"], index=m.dates) for k in ("SPY", "QQQ", "SMH")}
s0 = m.dates[m.dates.searchsorted(pd.Timestamp(start))]
bench = {k: float(v[m.dates[-1]] / v[s0] - 1) for k, v in B.items()}
T = pd.DataFrame(rows)
print(f"System R1, {start} to {end} (live panel, numpy {np.__version__})")
print(T.assign(ret=T.ret.map("{:+.1%}".format), worst=T.worst.map("{:.1%}".format), cost_pct=T.cost_pct.map("{:.2%}".format)).to_string(index=False))
print("benchmarks (total return, same dates):", {k: f"{v:+.1%}" for k, v in bench.items()})
T.to_csv("/home/user/Investment-Agent/usbt/audit/a72_2026_sizes.csv", index=False)
for lab, t in trades.items():
    t.to_csv("/home/user/Investment-Agent/usbt/audit/a72_trades_" + lab.replace("$", "").replace(",", "").replace(" ", "_") + ".csv", index=False)
json.dump(dict(start=start, end=end, bench=bench, rows=rows, panel=os.path.join(D, "panel.pkl"),
               panel_sha256=hashlib.sha256(open(os.path.join(D, "panel.pkl"), "rb").read()).hexdigest(),
               numpy=np.__version__, pandas=pd.__version__), open("/home/user/Investment-Agent/usbt/audit/a72_2026_sizes.json", "w"), indent=1)
