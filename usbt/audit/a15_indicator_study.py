"""Indicator study for short losing trades (owner, 15 Oct 2026).
For every indicator: its level, its slope (change over 5 and 10 sessions), whether the slope just changed direction,
and percentage forms (distance from averages, % moves). Measured on the stock, on the indexes (SPY, QQQ, VIX,
breadth) and on System K's own account, at the close before each buy decision.

Two samples:
  A. candidates: on every K decision day, the 30 best-scored stocks in K's universe that pass K's trend filter
     (what K could have bought). Outcome: next 63 sessions return, return vs SPY, and a 'quick loss' (the stock
     closes 10%+ below the decision close at any time within 63 sessions).
  B. K's own 559 positions. Outcome: win, return, and 'short loser' (sold within 63 sessions at a loss).
Discovery on decisions 2000-2015, confirmation on 2016-2026. A feature counts only if its effect has the same sign
and a meaningful size in both periods."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, pickle, numpy as np, pandas as pd, engine as X, yardstick as Y
from exp24_timing import mat
Y._init("../data/panel.pkl"); m = Y.MM; P = m.P; D = m.dates; T_ = len(D)
K = mat(json.load(open("../system_K.json"))["cfg"])
R = pickle.load(open("/tmp/claude-0/-home-user/611e8ebf-cd90-5f23-b4dc-f40b7c4b7cdf/scratchpad/runK.pkl", "rb"))
nav = R["daily"].nav.reindex(D)
Cf = P["Cf"].astype(np.float64)

def rsi_s(s, n=14):
    x = s.diff(); up = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn)).to_numpy()
def macd_s(s):
    mc = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean(); return ((mc - mc.ewm(span=9, adjust=False).mean()) / s * 100).to_numpy()

# ---- market-level series (index / VIX / breadth / K's account), each a length-T array
mk = {}
for t in ("SPY", "QQQ"):
    s = pd.Series(m.E[t]["c"]).ffill()
    mk[f"{t} RSI"] = rsi_s(s); mk[f"{t} MACD hist %"] = macd_s(s)
    mk[f"{t} vs 50d %"] = ((s / s.rolling(50).mean() - 1) * 100).to_numpy(); mk[f"{t} vs 200d %"] = ((s / s.rolling(200).mean() - 1) * 100).to_numpy()
    mk[f"{t} 5d move %"] = ((s / s.shift(5) - 1) * 100).to_numpy(); mk[f"{t} 21d move %"] = ((s / s.shift(21) - 1) * 100).to_numpy()
    mk[f"{t} below 1y high %"] = ((s / s.rolling(252, min_periods=60).max() - 1) * 100).to_numpy()
    mk[f"{t} 50d avg slope %/10d"] = ((s.rolling(50).mean() / s.rolling(50).mean().shift(10) - 1) * 100).to_numpy()
mk["VIX"] = np.asarray(P["VIX"], float)
mk["breadth % above 200d"] = np.asarray(m.breadth_pit, float) * 100
ns = nav.ffill()
mk["K account RSI"] = rsi_s(ns); mk["K account vs 50d %"] = ((ns / ns.rolling(50).mean() - 1) * 100).to_numpy()
mk["K account 21d move %"] = ((ns / ns.shift(21) - 1) * 100).to_numpy(); mk["K account below peak %"] = ((ns / ns.cummax() - 1) * 100).to_numpy()

# ---- stock-level arrays (T x N)
st = {"RSI": P["RSI"], "MACD hist": P["MACDH"], "ADX": P["ADX"], "OBV trend": P["OBVT"], "Up/down volume": P["UDVR"],
      "Money flow (CMF)": P["CMF"], "Relative volume": P["RVOL"], "6m vs SPY (rs126)": P["rs126"], "vs 200d %": P["dist200"],
      "near 52w high": P["hi52"], "30d volatility": P["RV30"]}
st = {k: np.asarray(v, np.float64) for k, v in st.items()}
st["vs EMA20 %"] = (Cf / P["EMA20"] - 1) * 100
st["vs SMA50 %"] = (Cf / P["SMA50"] - 1) * 100
st["vs EMA50 %"] = (Cf / P["EMA50"] - 1) * 100
for n in (5, 10, 21):
    a = np.full_like(Cf, np.nan); a[n:] = (Cf[n:] / Cf[:-n] - 1) * 100; st[f"{n}d move %"] = a
st["EMA50 slope %/10d"] = np.full_like(Cf, np.nan); e = np.asarray(P["EMA50"], np.float64); st["EMA50 slope %/10d"][10:] = (e[10:] / e[:-10] - 1) * 100
st["SMA200 slope %/21d"] = np.full_like(Cf, np.nan); s2 = np.asarray(P["SMA200"], np.float64); st["SMA200 slope %/21d"][21:] = (s2[21:] / s2[:-21] - 1) * 100
st["ATR % of price"] = np.asarray(P["ATR"], np.float64) / Cf * 100

def feats_at(i, js):
    """Feature dict for stocks js at close i: level, 5/10d slope, and slope turn (sign of 5d slope now vs 5 sessions ago)."""
    out = {}
    for k, a in st.items():
        v = a[i, js]; out[f"{k} | level"] = v
        if not k.endswith("move %"):
            s5 = a[i, js] - a[i - 5, js]; s5p = a[i - 5, js] - a[i - 10, js]
            out[f"{k} | 5d slope"] = s5; out[f"{k} | 10d slope"] = a[i, js] - a[i - 10, js]
            out[f"{k} | turn"] = np.where(np.isnan(s5) | np.isnan(s5p), np.nan, np.sign(s5) - np.sign(s5p))   # +2 turned up, -2 turned down, 0 same
    for k, a in mk.items():
        out[f"[mkt] {k} | level"] = np.full(len(js), a[i])
        out[f"[mkt] {k} | 5d slope"] = np.full(len(js), a[i] - a[i - 5])
        t_ = np.sign(a[i] - a[i - 5]) - np.sign(a[i - 5] - a[i - 10]); out[f"[mkt] {k} | turn"] = np.full(len(js), t_)
    return out

# ---- A. candidates on K's decision days
dec = [i for i in range(300, T_ - 64) if (m.month_end[i] or (D[i].day >= 11 and D[i - 1].day < 11))]
rows = []
for i in dec:
    sc, route, u = m.scores(i, K)
    ok = ~np.isnan(sc) & np.asarray(P["cont"][i], bool) & ~m.is_etf
    js = np.flatnonzero(ok); js = js[np.argsort(-sc[js])][:30]
    if not len(js): continue
    f = feats_at(i, js)
    c0 = Cf[i, js]; fut = Cf[i + 1:i + 64, js]
    f["fwd 63d %"] = (Cf[i + 63, js] / c0 - 1) * 100
    f["fwd 63d vs SPY pts"] = f["fwd 63d %"] - (m.E["SPY"]["c"][i + 63] / m.E["SPY"]["c"][i] - 1) * 100
    f["quick loss"] = (np.nanmin(fut, axis=0) / c0 - 1 <= -0.10).astype(float)
    f["rank"] = np.arange(len(js)); f["date"] = np.full(len(js), D[i]); f["sym"] = np.asarray(m.syms)[js]
    rows.append(pd.DataFrame(f))
A = pd.concat(rows, ignore_index=True)
A.to_pickle("/tmp/claude-0/-home-user/611e8ebf-cd90-5f23-b4dc-f40b7c4b7cdf/scratchpad/study_A.pkl")
print("candidates:", len(A), "decision days:", len(dec), "base quick-loss rate %.3f, base fwd63 %.2f%%" % (A["quick loss"].mean(), A["fwd 63d %"].mean()), flush=True)

# ---- B. K's own positions
L = R["lots"].copy(); L["entry"] = pd.to_datetime(L.entry); L = L[L.exit.notna()]
di = {x: k for k, x in enumerate(D)}
rowsB = []
for t in L.itertuples():
    k = di.get(t.entry)
    if k is None or k < 12: continue
    f = feats_at(k - 1, np.array([t.j]))
    f = {a: b[0] for a, b in f.items()}
    f.update(sym=t.sym, date=t.entry, ret=t.exit_ret, held=t.held, win=float(t.pnl > 0), short_loser=float(t.held < 63 and t.exit_ret < 0))
    rowsB.append(f)
B = pd.DataFrame(rowsB)
B.to_pickle("/tmp/claude-0/-home-user/611e8ebf-cd90-5f23-b4dc-f40b7c4b7cdf/scratchpad/study_B.pkl")
print("K positions:", len(B), "short-loser rate %.3f" % B.short_loser.mean(), flush=True)

# ---- discovery / confirmation
def study(df, outcome, good_low=True, minn=30):
    """For each feature: quintiles fixed on 2000-2015 data; outcome rate in the lowest and highest quintile, in both periods."""
    tr = df[df.date < "2016-01-01"]; te = df[df.date >= "2016-01-01"]
    res = []
    for c in df.columns:
        if "|" not in c: continue
        x = tr[c].astype(float)
        if x.nunique() < 3: continue
        if c.endswith("| turn"):
            cuts = None
            groups = {"turned down": lambda s: s < -1, "turned up": lambda s: s > 1}
            vals = {}
            for per, dd in (("2000-15", tr), ("2016-26", te)):
                s = dd[c].astype(float)
                for g, fn in groups.items():
                    sel = dd[fn(s)]
                    vals[(per, g)] = (sel[outcome].mean() if len(sel) >= minn else np.nan, len(sel))
            for g in groups:
                res.append(dict(feature=c, bucket=g, train=vals[("2000-15", g)][0], test=vals[("2016-26", g)][0],
                                n_train=vals[("2000-15", g)][1], n_test=vals[("2016-26", g)][1]))
            continue
        q = np.nanquantile(x, [0.2, 0.8])
        for per, dd in (("2000-15", tr), ("2016-26", te)):
            pass
        for g, fn in (("lowest 20%", lambda s: s <= q[0]), ("highest 20%", lambda s: s >= q[1])):
            a_ = tr[fn(tr[c].astype(float))]; b_ = te[fn(te[c].astype(float))]
            res.append(dict(feature=c, bucket=g, cut=q[0] if g.startswith("lowest") else q[1],
                            train=a_[outcome].mean() if len(a_) >= minn else np.nan, test=b_[outcome].mean() if len(b_) >= minn else np.nan,
                            n_train=len(a_), n_test=len(b_)))
    Rz = pd.DataFrame(res)
    Rz["base_train"] = tr[outcome].mean(); Rz["base_test"] = te[outcome].mean()
    Rz["lift_train"] = Rz.train - Rz.base_train; Rz["lift_test"] = Rz.test - Rz.base_test
    Rz["consistent"] = (np.sign(Rz.lift_train) == np.sign(Rz.lift_test)) & (Rz.lift_train.abs() >= 0.03) & (Rz.lift_test.abs() >= 0.03)
    return Rz

pd.set_option("display.width", 260, "display.max_rows", 400)
for name, df, outc in (("A_quickloss", A, "quick loss"), ("A_fwd63vsSPY", A, "fwd 63d vs SPY pts"), ("B_shortloser", B, "short_loser")):
    Rz = study(df, outc)
    if outc == "fwd 63d vs SPY pts":
        Rz["consistent"] = (np.sign(Rz.lift_train) == np.sign(Rz.lift_test)) & (Rz.lift_train.abs() >= 1.5) & (Rz.lift_test.abs() >= 1.5)
    Rz.to_csv(f"../audit/a15_{name}.csv", index=False)
    C_ = Rz[Rz.consistent].copy(); C_["strength"] = C_[["lift_train", "lift_test"]].abs().min(axis=1)
    print(f"\n==== {name}: base {Rz.base_train.iloc[0]:.3f} (2000-15) / {Rz.base_test.iloc[0]:.3f} (2016-26); "
          f"{len(C_)} of {len(Rz)} buckets consistent in both periods")
    print(C_.sort_values("strength", ascending=False).head(30).round(3).to_string(index=False))
