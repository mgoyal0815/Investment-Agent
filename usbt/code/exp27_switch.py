"""Experiment 27: one account that SWITCHES (not blends) between System C's stocks and a trend-gated
leveraged index sleeve. The two win in different years (exp26 table), so a blend averages the misses
away while a switch could keep the wins. First pass uses System C's own daily return curve while it is
'on' (approximation: the real account would liquidate and re-enter); verified in the engine afterwards."""
import itertools
import numpy as np
import pandas as pd
import exp26_lev as E

D = E.D
C = pd.read_csv("../results/final_System_C_curve2000.csv", index_col=0, parse_dates=True).nav.reindex(D).ffill()
rC = C.pct_change().fillna(0)
px = pd.DataFrame(E.IDX)
ME = pd.Series(D, index=D).groupby([D.year, D.month]).last()


def run(sig, rL, cost=0.002):
    """sig: boolean Series on decision days (True = leveraged sleeve). Applied from the next session."""
    s = sig.reindex(D).ffill().fillna(False).astype(bool).shift(1).fillna(False).astype(bool)
    r = rL.where(s, rC)
    sw = s != s.shift(1)
    r = r - sw * cost
    return (1 + r.loc["2000-01-03":]).cumprod()


def signals():
    out = {}
    mom = {lb: px / px.shift(lb) - 1 for lb in (63, 126, 252)}
    for sma, band, lb, rel, dec in itertools.product((100, 150, 200), (0.0, 0.05, 0.1), (63, 126), ("none", "spy", "both", "sys"), ("M", "W")):
        ma = px.SMH.rolling(sma).mean()
        on = px.SMH > ma * (1 + band)
        if rel == "spy":
            on &= mom[lb].SMH > mom[lb].SPY
        elif rel == "both":
            on &= mom[lb].SMH > mom[lb][["SPY", "QQQ"]].max(axis=1)
        elif rel == "sys":
            on &= mom[lb].SMH > (C / C.shift(lb) - 1)
        days = ME.values if dec == "M" else pd.Series(D, index=D).groupby([D.year, D.isocalendar().week.values]).last().values
        out[f"sma{sma} band{band} lb{lb} rel={rel} {dec}"] = on.loc[days]
    return out


if __name__ == "__main__":
    sleeves = {}
    for L in (2, 3):
        sleeves[f"{L}xSMH"] = E.RET[("SMH", L)]   # always in: the switch is the gate
        n, _ = E.strat(L=L, sma=1, band=-0.99, assets=("QQQ", "SMH"), lb=126, freq="M")
        sleeves[f"{L}x QQQ/SMH mom"] = n.pct_change().fillna(0)
    rows = []
    for (sn, rL), (gn, g) in itertools.product(sleeves.items(), signals().items()):
        nav = run(g, rL)
        lab = f"{sn} | {gn}"
        for per, (a, b) in {"train": ("2012-01-03", "2019-12-31"), "early": ("2000-01-03", "2011-12-30"),
                            "test": ("2020-01-02", None), "all": ("2000-01-03", None)}.items():
            c = E.card(nav, a, b, lab=lab); c["per"] = per; rows.append(c)
    R = pd.DataFrame(rows); R.to_csv("../results/exp27_switch.csv", index=False)
    W = R.pivot(index="lab", columns="per", values=["beat_all", "cagr", "dd"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    pd.set_option("display.width", 260, "display.max_columns", 20, "display.max_colwidth", 60)
    cols = ["beat_all_train", "beat_all_early", "beat_all_test", "beat_all_all", "cagr_train", "dd_train", "cagr_all", "dd_all"]
    W["sel"] = W.beat_all_train + 0.5 * W.cagr_train + 0.5 * W.dd_train
    print(W.sort_values("sel", ascending=False)[cols].head(25).round(3))
    print(W[W.dd_all > -0.55].sort_values("beat_all_all", ascending=False)[cols].head(15).round(3))
