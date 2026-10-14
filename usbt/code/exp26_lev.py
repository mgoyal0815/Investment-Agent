"""Experiment 26: trend-gated leveraged index ETFs (ordinary ETFs: no options, no margin), alone and
blended with System C in one account. Calendar-year scorecard 2000-2026 + max drawdown."""
import itertools, sys
import numpy as np
import pandas as pd
import lev

D = lev.dates
IDX = {k: lev.idx_close(k) for k in ("SPY", "QQQ", "SMH")}
RC = lev.cash_ret()
COST = 0.001   # per switch, each side


def lev_series(base, L):
    """Real fund where it exists, synthetic before (with a conservative 2.5% fee drag)."""
    if L == 1:
        return IDX[base]
    sym = {("SMH", 3): "SOXL", ("SMH", 2): "USD", ("QQQ", 3): "TQQQ", ("QQQ", 2): "QLD", ("SPY", 3): "UPRO", ("SPY", 2): "SSO"}[(base, L)]
    rr = lev.real(sym)
    sy = lev.synth(base, L, fee=0.025)
    r = rr.pct_change(fill_method=None)
    r = r.where(rr.notna() & rr.shift(1).notna(), sy.pct_change(fill_method=None))
    return (1 + r.fillna(0)).cumprod()


RET = {(b, L): lev_series(b, L).pct_change(fill_method=None).fillna(0) for b in IDX for L in (1, 2, 3)}


def strat(L=2, sma=200, band=0.0, pick="mom", lb=126, assets=("SPY", "QQQ", "SMH"), freq="D", off="cash"):
    """Each day (or month end) decide: hold L x of one index or cash. The decision at close t earns from t+1."""
    px = pd.DataFrame(IDX)
    ma = px.rolling(sma, min_periods=sma).mean()
    mom = px / px.shift(lb) - 1
    up = px > ma * (1 + band)
    dn = px < ma * (1 - band)
    hold = []
    cur = None
    dec_days = set(D) if freq == "D" else set(pd.Series(D, index=D).groupby([D.year, D.month]).last())
    for i, t in enumerate(D):
        if t in dec_days and not np.isnan(ma.iloc[i]["SPY"]):
            cands = [a for a in assets if (up.iloc[i][a] or (cur == a and not dn.iloc[i][a]))]
            if pick == "mom" and cands:
                cur = max(cands, key=lambda a: mom.iloc[i][a])
            elif pick == "fixed" and cands:
                cur = cands[0] if assets[0] in cands else None
            else:
                cur = None
        hold.append(cur)
    h = pd.Series(hold, index=D).shift(1)
    r = pd.Series(0.0, index=D)
    for a in assets:
        r = r.where(h != a, RET[(a, L)])
    offr = RC if off == "cash" else 0.0
    r = r.where(h.notna(), offr)
    sw = (h != h.shift(1)) & (h.notna() | h.shift(1).notna())
    r = r - sw * COST
    return (1 + r).cumprod(), h


def yearly(s):
    ye = s.groupby(s.index.year).last()
    p = ye.shift(1); p.iloc[0] = s.iloc[0]
    return ye / p - 1


def card(nav, a="2000-01-03", b=None, show=False, lab=""):
    s = nav.loc[a:b]; s = s / s.iloc[0]
    y = pd.DataFrame({"sys": yearly(s), **{k: yearly(IDX[k].loc[a:b]) for k in IDX}})
    y = y[y.index < 2027]
    full = y.iloc[:-1] if b is None else y   # 2026 is partial: report but don't count
    win = (full.sys > full[["SPY", "QQQ", "SMH"]].max(axis=1))
    dd = float((s / s.cummax() - 1).min())
    ddi = {k: float((IDX[k].loc[a:b] / IDX[k].loc[a:b].cummax() - 1).min()) for k in IDX}
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    out = dict(lab=lab, cagr=(s.iloc[-1]) ** (1 / yrs) - 1, dd=dd, beat_all=win.mean(), n=len(full),
               ddSPY=ddi["SPY"], ddSMH=ddi["SMH"])
    if show:
        print((y * 100).round(1).assign(win=win).to_string())
    return out


if __name__ == "__main__":
    rows = []
    for L, sma, band, lb, freq, assets in itertools.product((1, 2, 3), (100, 150, 200), (0.0, 0.02), (63, 126, 252), ("D", "M"),
                                                            [("SPY", "QQQ", "SMH"), ("QQQ", "SMH"), ("SMH",), ("QQQ",)]):
        nav, _ = strat(L=L, sma=sma, band=band, lb=lb, freq=freq, assets=assets)
        lab = f"L{L} sma{sma} band{band} lb{lb} {freq} {'+'.join(assets)}"
        for per, (a, b) in {"all": ("2000-01-03", None), "train": ("2012-01-03", "2019-12-31")}.items():
            c = card(nav, a, b, lab=lab); c["per"] = per; rows.append(c)
    R = pd.DataFrame(rows)
    R.to_csv("../results/exp26_lev_grid.csv", index=False)
    T = R[R.per == "train"].set_index("lab"); A = R[R.per == "all"].set_index("lab")
    J = T.join(A, lsuffix="_tr", rsuffix="_all")
    pd.set_option("display.width", 250)
    print(J.sort_values("beat_all_tr", ascending=False)[["cagr_tr", "dd_tr", "beat_all_tr", "cagr_all", "dd_all", "beat_all_all"]].head(30).round(3))
