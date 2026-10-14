"""Synthetic daily-reset leveraged index ETFs (no options, no margin: these are ordinary ETFs).
r_L = L * r_index - (L - 1) * r_cash - fee/252. Validated against the real funds where they exist."""
import json
import numpy as np
import pandas as pd

P = pd.read_pickle("../data/panel.pkl")
dates = P["dates"]


def idx_close(s):
    return pd.Series(P["etf"][s][1], index=dates).ffill()


def cash_ret():
    bil = idx_close("BIL").pct_change(fill_method=None)
    shy = idx_close("SHY").pct_change(fill_method=None)
    # before BIL exists, use a smoothed SHY return (short-rate proxy without the duration noise)
    r = bil.where(bil.notna(), shy.rolling(63, min_periods=5).mean())
    return r.fillna(0).clip(-0.002, 0.002)


def synth(s, L, fee=0.0095):
    r = idx_close(s).pct_change(fill_method=None).fillna(0)
    rc = cash_ret()
    rl = L * r - (L - 1) * rc - fee / 252
    return (1 + rl).cumprod()


def real(sym):
    j = json.load(open(f"../data/lev/{sym}.json"))
    s = pd.Series({pd.Timestamp(x["date"]): x["adjClose"] for x in j}).sort_index()
    return s.reindex(dates)


if __name__ == "__main__":
    for sym, base, L in [("SOXL", "SMH", 3), ("USD", "SMH", 2), ("TQQQ", "QQQ", 3), ("QLD", "QQQ", 2), ("UPRO", "SPY", 3), ("SSO", "SPY", 2)]:
        rr = real(sym).dropna()
        sy = synth(base, L).reindex(rr.index)
        a = rr.pct_change().dropna(); b = sy.pct_change().dropna()
        yrs = len(rr) / 252
        ca = (rr.iloc[-1] / rr.iloc[0]) ** (1 / yrs) - 1
        cb = (sy.iloc[-1] / sy.iloc[0]) ** (1 / yrs) - 1
        print(f"{sym} vs synth {L}x{base}: corr {a.corr(b):.3f}  CAGR real {ca:.1%} synth {cb:.1%}  from {rr.index[0].date()}")
