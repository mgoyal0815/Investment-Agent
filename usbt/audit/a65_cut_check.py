"""Cut-off check at 2003-06-30 and 2011-08-31 for R, S synthetic, S2 real (QLD/SSO) and S real (all), after the splice fix."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, numpy as np, pandas as pd, engine as X
from exp24_timing import mat
src = open("a62_s_truncation.py").read()
exec(src[src.index("P = pd.read_pickle"):src.index("full = X.run")].replace(src[src.index("cfg = mat"):src.index("\n", src.index("cfg = mat"))], ""))
R = json.load(open("../system_R.json"))["cfg"]
CF = {"R": R, "S synthetic": dict(R, lev_park=200, lev_after=126), "S2 real QLD/SSO": dict(R, lev_park=200, lev_after=126, lev_real=True, lev_only=["SPY", "QQQ"]),
      "S real all": dict(R, lev_park=200, lev_after=126, lev_real=True)}
MF = X.Market(P)
for T in ("2003-06-30", "2011-08-31"):
    mc = cut_market(T)
    for lab, c in CF.items():
        a = X.run(MF, mat(c), start="2000-01-03")["curve"].loc[:T].iloc[:-1]; b = X.run(mc, mat(c), start="2000-01-03")["curve"].loc[:T].iloc[:-1]
        d = float((a / b - 1).abs().max()); print((T, lab, f"{d:.2e}", "IDENTICAL" if d < 1e-9 else "DIFFERS"), flush=True)
