"""System C + a trend-gated leveraged index sleeve in ONE account, rebalanced to target weights monthly."""
import itertools
import pandas as pd
import exp26_lev as E

C = pd.read_csv("../results/final_System_C_curve2000.csv", index_col=0, parse_dates=True).nav
rC = C.pct_change().reindex(E.D).fillna(0)


def blend(rL, w):
    me = pd.Series(E.D, index=E.D).groupby([E.D.year, E.D.month]).last()
    nav, a, b = [], 1 - w, w
    reb = set(me)
    for t in E.D[E.D >= "2000-01-03"]:
        a *= 1 + rC[t]; b *= 1 + rL[t]
        nav.append(a + b)
        if t in reb:
            v = a + b; a, b = v * (1 - w), v * w
    return pd.Series(nav, index=E.D[E.D >= "2000-01-03"])


rows = []
for L, sma, lb, freq, assets, w in itertools.product((2, 3), (150, 200), (63, 126), ("D", "M"),
                                                     [("SPY", "QQQ", "SMH"), ("QQQ", "SMH"), ("SMH",), ("QQQ",)], (0.2, 0.3, 0.4, 0.5)):
    nav, _ = E.strat(L=L, sma=sma, lb=lb, freq=freq, assets=assets)
    rL = nav.pct_change().fillna(0)
    b = blend(rL, w)
    lab = f"w{w} L{L} sma{sma} lb{lb} {freq} {'+'.join(assets)}"
    for per, (x, y) in {"train": ("2012-01-03", "2019-12-31"), "all": ("2000-01-03", None),
                        "early": ("2000-01-03", "2011-12-30"), "test": ("2020-01-02", None)}.items():
        c = E.card(b, x, y, lab=lab); c["per"] = per; rows.append(c)
R = pd.DataFrame(rows); R.to_csv("../results/exp26b_blend.csv", index=False)
W = R.pivot(index="lab", columns="per", values=["beat_all", "cagr", "dd"])
W.columns = [f"{a}_{b}" for a, b in W.columns]
pd.set_option("display.width", 250, "display.max_columns", 20)
cols = ["beat_all_train", "beat_all_early", "beat_all_test", "beat_all_all", "cagr_all", "dd_all", "dd_train"]
print("System C alone:", {p: E.card(C.reindex(E.D).ffill().dropna(), x, y) for p, (x, y) in {"train": ("2012-01-03", "2019-12-31"), "all": ("2000-01-03", None)}.items()})
print(W.sort_values("beat_all_train", ascending=False)[cols].head(25).round(3))
print(W[W.dd_all > -0.45].sort_values("beat_all_all", ascending=False)[cols].head(15).round(3))
