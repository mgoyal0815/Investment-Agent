"""Phase 2 of exp43: Bayesian ridge on main effects + pairwise interactions, per period."""
import json, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, ARDRegression

d = pd.read_csv("../results/exp43_design.csv")
fac = [c for c in d.columns if c.startswith("f:")]
X = pd.get_dummies(d[fac].astype(str), drop_first=False)
# drop the System D level of each factor so every coefficient = effect of switching that enhancement on
base = {c: d[c].iloc[0] for c in fac}
X = X[[c for c in X.columns if not any(c == f"{f}_{base[f]}" for f in fac)]].astype(float)
main = list(X.columns)
inter = {}
for a, b in itertools.combinations(main, 2):
    if a.split("_")[0] == b.split("_")[0]:
        continue
    v = X[a] * X[b]
    if v.sum() >= 15:
        inter[f"{a} x {b}"] = v
XI = pd.concat([X, pd.DataFrame(inter)], axis=1)
res = {}
for p in ("train", "early", "late"):
    y = d[f"{p}|beat"] - 0.5 * (1 - d[f"{p}|ddok"])
    m = BayesianRidge(compute_score=True).fit(XI.values, y.values)
    sd = np.sqrt(np.diag(m.sigma_))
    res[p] = pd.DataFrame({"effect": m.coef_, "sd": sd}, index=XI.columns)
T = res["train"].join(res["early"], rsuffix="_early").join(res["late"], rsuffix="_late")
T["credible_train"] = (T.effect.abs() > 2 * T.sd)
T["same_sign_all"] = (np.sign(T.effect) == np.sign(T.effect_early)) & (np.sign(T.effect) == np.sign(T.effect_late))
T["matters"] = T.credible_train & T.same_sign_all & ((T.effect_early.abs() > T.sd_early) | (T.effect_late.abs() > T.sd_late))
T = T.sort_values("effect", ascending=False)
T.to_csv("../results/exp43_effects.csv")
pd.set_option("display.width", 250, "display.max_rows", 300)
cols = ["effect", "sd", "effect_early", "effect_late", "credible_train", "same_sign_all", "matters"]
print("MAIN EFFECTS (change in [share of years beating SPY&QQQ - 0.5 x drawdown-rule failures], per period)")
print((T.loc[[i for i in T.index if " x " not in i], cols]).round(3).to_string())
print("\nINTERACTIONS THAT MATTER")
print(T.loc[[i for i in T.index if " x " in i and T.loc[i, "matters"]], cols].round(3).to_string())
