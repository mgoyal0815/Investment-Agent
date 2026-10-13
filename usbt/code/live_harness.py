"""Drive the LIVE US v5 code itself (v5_engine, v5_sleeve, v5_alloc from the v5-forward-test
repo, imported read-only, nothing edited) over the backtest panel, one call per session in the
order daily_run.py uses:

    WS-9 ramp step -> fund() at the open -> run_sleeve_session() -> mark_units() -> run_session()

Signals come from engine.Market.scores (the vectorised form of sleeve_select.py). Prices are
handed over exactly as the live loader hands them: dividend-adjusted closes and opens, so the
live code's whole-share arithmetic runs on adjusted prices, as it does in the live repo's own
backtests. This is the reference the fast engine (engine.py) is checked against.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/v5-forward-test")
sys.path.insert(0, "/home/user/v5-forward-test/claude")
import v5_engine as E      # noqa: E402
import v5_sleeve as S      # noqa: E402
import v5_alloc as A       # noqa: E402
import engine as X         # noqa: E402

LEG = ["SPY", "QQQ", "SMH", "GLD", "TLT", "UUP", "DBC", "SHY", "HYG"]


def ws9(state, d):
    rm = int(state.get("ramp_months") or E.RAMP_MONTHS)
    if rm <= 1 or not state.get("ramp_start"):
        return state
    rf = E.get_ramp_factor(d, state["ramp_start"], rm)
    which = "pending_alloc" if state.get("pending_alloc") else "current_alloc"
    base = state.get(which) or {}
    deployed = sum(w for k, w in base.items() if k not in (E.RAMP_PARK, "cash"))
    steps = [min(1.0, k / rm) for k in range(1, rm + 1)]
    in_force = min(steps, key=lambda s: abs(s - deployed)) if deployed > 0 else None
    if in_force is None or abs(in_force - deployed) > 1e-6 or rf <= in_force + 1e-9:
        return state
    unramped = {k: w / deployed for k, w in base.items() if k not in (E.RAMP_PARK, "cash")}
    s = dict(state)
    s["pending_alloc"] = E.apply_ramp(unramped, rf)
    if which == "current_alloc":
        s["pending_regime"] = state.get("regime")
    return s


def run(m, start="2012-01-03", end=None, capital=10000.0, ramp=3):
    P = m.P
    C, O, Cf = P["C"], P["O"], P["Cf"]
    i0 = int(m.dates.searchsorted(pd.Timestamp(start)))
    i1 = len(m.dates) if end is None else int(m.dates.searchsorted(pd.Timestamp(end), side="right"))
    cfg = dict(X.CFG)
    d0 = str(m.dates[i0].date())
    book = S.new_book(capital, inception=d0)
    book["cash"] = 0.0
    state = E.state_template(d0, A.UNIT_BASE, {"cash": 1.0}, "FULL_BULL")
    state.update(portfolio_nav=capital, inception_nav=capital, ramp_months=ramp, current_month=None)
    tgt, reg = E.bull_target(bool(m.hyg_ok[i0]), {t: m.mom126[t][i0] for t in X.LEADERS},
                             {t: m.mom63[t][i0] for t in X.DEFENSIVES}, {t: bool(m.avail[t][i0]) for t in X.DEFENSIVES})
    state["pending_alloc"] = E.apply_ramp(tgt, E.get_ramp_factor(d0, d0, ramp)) if ramp > 1 else tgt
    state["pending_regime"] = reg
    curve, expo = [], []
    syms = m.syms
    J = {t: j for j, t in enumerate(syms)}
    Ec = {k: m.E[k]["c"] for k in LEG}
    for i in range(i0, i1):
        d = str(m.dates[i].date())
        if i > i0:
            state = ws9(state, d)
        alloc = dict(state.get("pending_alloc") or state["current_alloc"])
        target = float(alloc.get("stock", 0.0)) * float(state["portfolio_nav"])
        prev_closes = {t: float(p["last_close"]) for t, p in book["positions"].items() if p.get("last_close") is not None}
        opens_all = {syms[j]: float(O[i, j]) for j in [J[t] for t in book["positions"]] if O[i, j] == O[i, j]}
        book, frep = A.fund(book, target, prev_closes, opens_all, d)
        # signals: qualifying names UNION every held name (19.13)
        sc, route, u = m.scores(i, cfg)
        q = (Cf[i] - P["EMA20"][i]) / P["ATR"][i]
        held_j = {J[t] for t in book["positions"]}
        cand = set(np.nonzero(~np.isnan(sc) & (route > 0))[0].tolist()) | held_j
        sig = {}
        for j in cand:
            if C[i, j] != C[i, j] or not (P["ATR"][i, j] > 0):
                continue
            t = syms[j]
            rt = "Continuation" if route[j] == 1 else "Recovery" if route[j] == 2 else (
                book["positions"][t]["route"] if t in book["positions"] else None)
            if rt is None:
                continue
            f = lambda a: float(a[i, j]) if a[i, j] == a[i, j] else None
            sig[t] = {"route": rt, "rp_c": float(sc[j]) if sc[j] == sc[j] else None, "q": float(q[j]) if q[j] == q[j] else None,
                      "close": float(C[i, j]), "atr": f(P["ATR"]), "sma50": f(P["SMA50"]), "sma200": f(P["SMA200"]),
                      "ema20": f(P["EMA20"]), "adv": f(P["ADV20"]) or 0.0, "rv30": f(P["RV30"])}
        closes = {syms[j]: float(C[i, j]) for j in cand if C[i, j] == C[i, j]}
        pend = {o["ticker"] for o in book.get("pending_orders", [])}
        opens = {t: float(O[i, J[t]]) for t in (set(book["positions"]) | pend) if O[i, J[t]] == O[i, J[t]]}
        market = {"date": d, "session_index": i, "opens": opens, "closes": closes, "signals": sig,
                  "spy_above_200": bool(m.b200[i])}
        book, srep = S.run_sleeve_session(book, market)
        unit_nav = A.mark_units(book, closes)
        dret = {}
        for t in LEG:
            a, b = Ec[t][i], Ec[t][i - 1]
            if a == a and b == b:
                dret[t] = a / b - 1
        dret["cash"] = dret.get("SHY", 0.0)
        hist = state.get("sleeve_nav_history", [])
        h2 = [h for h in hist if h["date"] != d] + [{"date": d, "nav": round(unit_nav, 2)}]
        mom10 = (h2[-2]["nav"] / h2[-12]["nav"] - 1.0) if len(h2) >= 12 else 0.0
        om = {"d": d, "is_month_end": bool(m.month_end[i]), "b200": bool(m.b200[i]), "t63": bool(m.t63[i]),
              "hyg_ok": bool(m.hyg_ok[i]), "sleeve_nav": unit_nav, "sleeve_mom10": mom10,
              "mom126": {t: float(m.mom126[t][i]) for t in X.LEADERS},
              "mom63": {t: float(m.mom63[t][i]) for t in X.DEFENSIVES},
              "avail": {t: bool(m.avail[t][i]) for t in X.DEFENSIVES}, "daily_ret": dret}
        state, dec = E.run_session(state, om)
        book["trades"], book["closed"] = book["trades"][-50:], book["closed"][-50:]
        state["near_misses"] = state.get("near_misses", [])[-5:]
        curve.append(state["portfolio_nav"])
        expo.append(state["current_alloc"].get("stock", 0.0))
    return dict(curve=pd.Series(curve, index=m.dates[i0:i1]), expo=pd.Series(expo, index=m.dates[i0:i1]))
