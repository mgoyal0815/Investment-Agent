"""Short-term mean reversion (exp73, owner 16 Oct 2026): a second strategy to run beside the momentum system.
Universe: that day's S&P 500 members (point in time), stocks only, adjusted close above its 200-day average.
Signal at the close: 2-day RSI below rsi_in (a sharp short dip in an uptrend). Buy at the next open, most oversold
first, into `slots` equal slots (whole shares, same trading costs as the engine). Sell at the next open after the close
rises above its 5-day average, or after max_hold sessions. Idle money earns short Treasuries (SHY)."""
import numpy as np, pandas as pd
import engine as X


def rsi_n(cf, n=2):
    d = cf.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def run_mr(m, start="2000-01-03", end="2025-12-31", slots=5, rsi_in=10, max_hold=10, cash0=10000.0):
    P = m.P; D = m.dates
    cf = pd.DataFrame(np.asarray(P["Cf"], np.float64)); c_raw = np.asarray(P["C"], np.float64); o_raw = np.asarray(P["O"], np.float64)
    adj = cf.to_numpy() / c_raw                                   # adjustment factor so the open matches the adjusted close
    oa = o_raw * adj
    r2 = rsi_n(cf).to_numpy(); s200 = cf.rolling(200, min_periods=200).mean().to_numpy(); s5 = cf.rolling(5).mean().to_numpy()
    CF = cf.to_numpy(); spm = m.sp_member; etf = m.is_etf
    shy = pd.Series(m.E["SHY"]["c"], index=D).ffill().pct_change().fillna(0).to_numpy()
    i0 = int(np.searchsorted(D, pd.Timestamp(start))); i1 = int(np.searchsorted(D, pd.Timestamp(end), side="right"))
    cash, pos, nav, trades = cash0, {}, [], []                     # pos: j -> dict(sh, entry_i, px)
    want_buy, want_sell = [], []
    for i in range(i0, i1):
        # fills at today's open for yesterday's decisions
        for j in want_sell:
            p = pos.pop(j); px = oa[i, j]
            if not px == px: px = CF[i - 1, j]
            val = p["sh"] * px; cash += val - X.tcost(p["sh"], val); trades.append((D[i], j, "SELL", px / p["px"] - 1))
        navo = cash + sum(p["sh"] * (oa[i, j] if oa[i, j] == oa[i, j] else CF[i - 1, j]) for j, p in pos.items())
        for j in want_buy:
            if len(pos) >= slots: break
            px = oa[i, j]
            if not (px == px and px >= 1.0): continue
            sh = int(min(navo / slots, cash) // px)
            if sh <= 0: continue
            val = sh * px; cash -= val + X.tcost(sh, val); pos[j] = dict(sh=sh, entry_i=i, px=px); trades.append((D[i], j, "BUY", 0.0))
        cash *= 1 + shy[i] if cash > 0 else 1
        nav.append(cash + sum(p["sh"] * (CF[i, j] if CF[i, j] == CF[i, j] else p["px"]) for j, p in pos.items()))
        # decisions at today's close
        want_sell = [j for j, p in pos.items() if (CF[i, j] > s5[i, j]) or (i - p["entry_i"] >= max_hold - 1) or not CF[i, j] == CF[i, j]]
        free = slots - len(pos) + len(want_sell)
        if free > 0:
            ok = spm[i] & ~etf & (CF[i] > s200[i]) & (r2[i] < rsi_in) & (CF[i] >= 5)
            js = [j for j in np.flatnonzero(ok) if j not in pos]
            js.sort(key=lambda j: r2[i, j]); want_buy = js[:free]
        else:
            want_buy = []
    return pd.Series(nav, index=D[i0:i1]), pd.DataFrame(trades, columns=["date", "j", "side", "ret"])
