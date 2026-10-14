"""One account simulator that runs the US v5 live system and India v8 (ported to US data),
with each difference between them as a switch.

A real $10,000 account, whole shares for stocks AND ETFs (owner's rule), orders raised at
session d's close fill at d+1's open (framework 1.2 / A14). Share counts and the per-share
commission use the price actually traded (split-adjusted price x later split ratios); values
use the dividend-adjusted price, so dividends are earned.

Defaults = the live US v5 system (v5_engine + v5_sleeve + v5_alloc WS-3/WS-5/WS-9).
Each India difference is one key in CFG; see SWITCHES below.
"""
import math
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

LEADERS = ("SPY", "QQQ", "SMH")
DEFENSIVES = ("GLD", "TLT", "UUP", "DBC")

CFG = dict(
    # ---- universe and ranking
    universe="adv50",       # "adv50": 20-session median $ volume >= $50m (v5); "topN": N most-traded by 126-session median turnover (India)
    universe_n=500,
    min_adv=50e6,           # applied only with universe="adv50"
    score="robust",         # "robust": v5 RobustPair per route (19.1-19.3); "nifty": India 6m+12m return / vol z-score
    cont_cut=0.88, rec_cut=0.83,
    trend=True,             # nifty score: new buys must be in the Continuation route (India v8)
    zone_gate=True,         # v5 19.4: new entries only in Green/Yellow (pullback band near EMA20)
    # ---- sizing
    sizing="v5",            # "v5": 19.6 risk/vol/base sizing; "slot": NAV / N per position (India)
    N=10,
    # ---- exits
    exit="stop",            # "stop": ratcheting stop + 50% ladder (v5); "rank": sell when ranked outside buffer*N at a review (India); "stop+rank"
    min_hold=88,            # sessions; blocks stop exits (19.8)
    ladder=0.5,
    buffer=1.5,
    # ---- adds
    adds="v5",              # "v5": Green-zone adds any session (19.12); "india": +20% winners in top N, half slot, cap 3 slots; None
    add_gain=0.20, add_cap=3.0, add_size=0.5,
    # ---- cadence
    cadence="daily",        # "daily" (v5), "twice" (India: 11th and last session), "monthly"
    # ---- market risk layer
    overlay="v5",           # "v5": regime weights, stock sold to match (5.1); "pause": India buy pause; "none"; "v5_soft": v5 weights cap new buys only, no forced sale
    pause_rules=("breadth", "hyg", "large"),
    intramonth=True,        # v5 11.1 sleeve -12% MTD exit + 11.3 daily re-entry
    regime_w=None,          # override 5.1 weights: dict(FULL_BULL=(stock, leader), CREDIT_WARN=(stock, leader, def), MILD_BEAR=(stock,), DEEP_BEAR=(stock,))
    # ---- idle cash
    park=None,              # None (v5: sleeve cash earns nothing), "GLD", "SHY", "BIL", "SPY", "QQQ", "leader"
    gold_slots=0,           # India: GLD takes k of N slots while it leads
    # ---- start, orders
    ramp=3,                 # 9.5 ramp-in months (0/1 = invest from day one)
    adv_cap=None,           # India: no order above this fraction of median daily $ volume
    whole_etf=True,
    delist_cash=5,          # WS-3
    capital=10000.0,
)

COST_PS, COST_BPS, COST_MIN = 0.005, 0.0010, 1.00


def tcost(raw_sh, value):
    if raw_sh <= 0:
        return 0.0
    return max(COST_MIN, COST_PS * raw_sh + COST_BPS * value)


def pct_rank(x):
    """pandas rank(pct=True, method='average') on a 1-D array, NaN stays NaN."""
    out = np.full(x.shape, np.nan)
    m = ~np.isnan(x)
    n = m.sum()
    if n == 0:
        return out
    out[m] = pd.Series(x[m]).rank(pct=True).to_numpy()
    return out


class Market:
    """Panel plus the date-level series both systems read."""

    def __init__(self, P):
        self.P = P
        self.dates = P["dates"]
        self.T, self.N = P["C"].shape
        self.syms = P["syms"]
        E = {}
        for k, (o, c) in P["etf"].items():
            c = pd.Series(c, index=self.dates)
            E[k] = dict(raw=c.to_numpy(), c=c.ffill().to_numpy(), o=pd.Series(o, index=self.dates).to_numpy())
        self.E = E
        spy = pd.Series(E["SPY"]["c"], index=self.dates)
        s200 = spy.rolling(200, min_periods=200).mean()
        self.b200 = (spy >= s200).where(s200.notna(), other=True).astype(bool).shift(1).fillna(True).astype(bool).to_numpy()
        tlt = pd.Series(E["TLT"]["c"], index=self.dates)
        self.t63 = (tlt >= tlt.rolling(63, min_periods=63).mean()).astype(bool).shift(1).fillna(False).astype(bool).to_numpy()
        hyg = pd.Series(E["HYG"]["c"], index=self.dates)
        h63 = hyg.rolling(63, min_periods=63).mean()
        hraw = pd.Series(E["HYG"]["raw"], index=self.dates)
        self.hyg_ok = (hyg >= h63).where(h63.notna() & hraw.notna(), other=True).astype(bool).shift(1).fillna(True).astype(bool).to_numpy()
        self.mom126 = {t: (pd.Series(E[t]["c"], index=self.dates) / pd.Series(E[t]["c"], index=self.dates).shift(126) - 1).to_numpy()
                       for t in LEADERS}
        self.mom63 = {t: (pd.Series(E[t]["c"], index=self.dates) / pd.Series(E[t]["c"], index=self.dates).shift(63) - 1).shift(1).to_numpy()
                      for t in DEFENSIVES}
        self.avail = {t: pd.Series(E[t]["raw"], index=self.dates).notna().shift(1, fill_value=False).to_numpy() for t in DEFENSIVES}
        mo = pd.Series(self.dates.to_period("M"), index=self.dates)
        self.month_end = (mo != mo.shift(-1)).to_numpy()
        self.day_in_month = mo.groupby(mo).cumcount().to_numpy()
        self.ym = (self.dates.year * 12 + self.dates.month).to_numpy()
        wk = pd.Series(self.dates.to_period("W"), index=self.dates)
        self.week_end = (wk != wk.shift(-1)).to_numpy()
        # India market signals on US data
        C, Cf = P["C"], P["Cf"]
        traded = ~np.isnan(C)
        elig = traded & (P["hist"] >= 240) & ~np.isnan(P["medv"])
        medv = np.where(elig, P["medv"], np.nan)
        self.is_etf = P.get("is_etf", np.zeros(self.N, bool))
        # turnover rank among STOCKS (sector ETFs excluded) and among stocks + sector ETFs
        trk = pd.DataFrame(np.where(self.is_etf, np.nan, medv)).rank(axis=1, ascending=False).to_numpy()
        self.trank = trk
        self.trank_etf = pd.DataFrame(medv).rank(axis=1, ascending=False).to_numpy()
        top500 = trk <= 500
        above = Cf > P["SMA200"]
        self.breadth = (above & top500).sum(1) / np.maximum(top500.sum(1), 1)
        top100 = pd.DataFrame(trk <= 100).shift(1, fill_value=False).to_numpy()
        r = np.where(top100, np.nan_to_num(P["ret1"]), np.nan)
        cnt = top100.sum(1)
        ew = np.where(cnt > 0, np.nansum(np.where(top100, np.nan_to_num(P["ret1"]), 0), 1) / np.maximum(cnt, 1), 0.0)
        self.large_idx = np.cumprod(1 + ew)
        self.large_sma50 = pd.Series(self.large_idx).rolling(50, min_periods=50).mean().to_numpy()
        self._ucache = {}
        self._esma = {}
        # exp49: point-in-time industry assignment (sectors.py) and intermediate-horizon momentum
        try:
            import json as _js
            sec = np.load("../data/sector_idx.npy")
            self.sector = sec if sec.shape == (self.T, self.N) else None
            self.sector_list = _js.load(open("../data/sector_list.json"))
        except Exception:
            self.sector, self.sector_list = None, []
        Cf_ = P["Cf"]
        m121 = np.full(Cf_.shape, np.nan, np.float32); m127 = np.full(Cf_.shape, np.nan, np.float32); m63 = np.full(Cf_.shape, np.nan, np.float32)
        m121[252:] = Cf_[231:-21] / Cf_[:-252] - 1          # 12-1 month
        m127[252:] = Cf_[126:-126] / Cf_[:-252] - 1         # 12-7 month (Novy-Marx)
        m63[63:] = Cf_[63:] / Cf_[:-63] - 1
        P["mom12_1"], P["mom12_7"], P["mom63"] = m121, m127, m63
        # volume indicators (volume_ind.py), matched to this panel by shape
        for fn in ("../data/vol_panel.npz", "../data/vol_panel_idx.npz"):
            try:
                z = np.load(fn)
                if z["UDVR"].shape == (self.T, self.N):
                    for k in z.files:
                        P[k] = z[k]
                    break
            except Exception:
                pass
        # ---- regime indicators (exp43); all use closes up to the session itself, acted on at the next open
        a50 = Cf > P["SMA50"]
        self.b50 = (a50 & top500).sum(1) / np.maximum(top500.sum(1), 1)            # share of top-500 above 50-day
        b50s = pd.Series(self.b50)
        # breadth thrust (Zweig-style): from under 25% above the 50-day to over 55% within 20 sessions
        self.thrust = ((b50s.rolling(20, min_periods=20).min() < 0.25) & (b50s > 0.55)).to_numpy()
        spy_c = pd.Series(E["SPY"]["c"]).ffill()
        # narrow market: cap-weighted SPY beating the equal-weight top-100 over 63 sessions, SPY in an uptrend
        rel = spy_c / pd.Series(self.large_idx)
        self.narrow = ((rel > rel.rolling(63, min_periods=63).mean()) & (spy_c > spy_c.rolling(50, min_periods=50).mean())).to_numpy()
        vix = pd.Series(P["VIX"]) if P.get("VIX") is not None else None
        self.vix_rel = (vix / vix.rolling(50, min_periods=50).mean()).to_numpy() if vix is not None else np.ones(self.T)
        hy, ie = pd.Series(E["HYG"]["c"]).ffill(), pd.Series(E["IEF"]["c"]).ffill()
        cr = hy / ie
        self.credit_bad = (cr < cr.rolling(50, min_periods=50).mean()).to_numpy()
        self.mom_crash = (spy_c / spy_c.shift(504) - 1 < 0).to_numpy()            # market down over 2 years
        self.size_pct = 1 - pd.DataFrame(trk).rank(axis=1, pct=True).to_numpy()   # 1 = biggest by turnover
        # point-in-time S&P 500 membership (fja05680/sp500): no knowledge of the future
        try:
            h = pd.read_csv("../data/sp500_hist.csv", parse_dates=["date"]).sort_values("date")
            J = {s: j for j, s in enumerate(self.syms)}
            snap = np.zeros((len(h), self.N), bool)
            for k, t in enumerate(h.tickers):
                for s in t.split(","):
                    j = J.get(s.replace(".", "-"))
                    if j is not None:
                        snap[k, j] = True
            pos_ = np.searchsorted(h.date.values, self.dates.values, side="right") - 1
            self.sp_member = snap[np.clip(pos_, 0, None)]
        except Exception:
            self.sp_member = None
        # ---- point-in-time versions of the market signals (audit, 15 Oct 2026): the originals rank the top
        # 500/100 by turnover over the WHOLE panel, which includes the live list's future survivors (20-50% of
        # that set were not S&P 500 members on the day). These use only that day's S&P 500 members.
        if self.sp_member is not None:
            spm = self.sp_member
            self.breadth_pit = (above & spm).sum(1) / np.maximum(spm.sum(1), 1)
            self.b50_pit = (a50 & spm).sum(1) / np.maximum(spm.sum(1), 1)
            trk_sp = pd.DataFrame(np.where(spm & ~self.is_etf, medv, np.nan)).rank(axis=1, ascending=False).to_numpy()
            t100 = pd.DataFrame(trk_sp <= 100).shift(1, fill_value=False).to_numpy()
            ew_p = np.where(t100.sum(1) > 0, np.nansum(np.where(t100, np.nan_to_num(P["ret1"]), 0), 1) / np.maximum(t100.sum(1), 1), 0.0)
            self.large_idx_pit = np.cumprod(1 + ew_p)
            self.large_sma50_pit = pd.Series(self.large_idx_pit).rolling(50, min_periods=50).mean().to_numpy()
        # point-in-time Nasdaq-100 membership 2004-2026 (ndx_members.py); nobody counts as a member before 2004
        try:
            h = pd.read_csv("../data/ndx_hist.csv", parse_dates=["date"]).sort_values("date")
            J = {s: j for j, s in enumerate(self.syms)}
            snap = np.zeros((len(h) + 1, self.N), bool)
            for k, t in enumerate(h.tickers):
                for s in t.split(","):
                    j = J.get(s)
                    if j is not None:
                        snap[k + 1, j] = True
            pos_ = np.searchsorted(h.date.values, self.dates.values, side="right")
            self.ndx_member = snap[pos_]
        except Exception:
            self.ndx_member = None
        self.e_sma50 = {t: pd.Series(E[t]["c"]).rolling(50, min_periods=50).mean().to_numpy() for t in LEADERS}
        _s = pd.Series(E["SPY"]["c"])
        self.spy_dd = (_s / _s.rolling(252, min_periods=60).max() - 1).to_numpy()

    def etf_vs_sma(self, t, n, i):
        """+1 if ETF t closes above its n-session average at session i, -1 below, 0 unknown."""
        key = (t, n)
        if key not in self._esma:
            c = pd.Series(self.E[t]["c"])
            self._esma[key] = np.sign((c - c.rolling(n, min_periods=n).mean()).fillna(0).to_numpy())
        return self._esma[key][i]

    def etf_macd(self, t, i):
        key = ("macd", t)
        if key not in self._esma:
            c = pd.Series(self.E[t]["c"]).ffill()
            mc = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
            self._esma[key] = (mc - mc.ewm(span=9, adjust=False).mean()).fillna(0).to_numpy()
        return self._esma[key][i]

    def universe(self, i, cfg):
        """Boolean mask of names that can be ranked and bought at session i."""
        P = self.P
        base = ~np.isnan(P["C"][i]) & ~np.isnan(P["SMA200"][i]) & (P["ATR"][i] > 0)
        etf_in = cfg.get("etf_in_universe", False)
        if not etf_in:
            base &= ~self.is_etf
        if cfg["universe"] == "adv50":
            return base & (P["ADV20"][i] >= cfg["min_adv"])
        ex = cfg.get("exclude")
        if ex:
            key = tuple(sorted(ex))
            if key not in self._ucache:
                self._ucache[key] = ~np.isin(self.syms, list(ex))
            base &= self._ucache[key]
        if cfg["universe"] == "sp500":
            # etf_in_universe: index / sector ETFs (panel columns flagged is_etf) compete for slots like stocks
            return base & (self.sp_member[i] | (self.is_etf if etf_in else False))
        if cfg["universe"] == "sp500+ndx":
            return base & (self.sp_member[i] | self.ndx_member[i])
        if etf_in:
            return base & ((self.trank[i] <= cfg["universe_n"]) | self.is_etf)
        u = base & (self.trank[i] <= cfg["universe_n"])
        if cfg.get("sp500_only"):
            u &= self.sp_member[i]
        return u

    def scores(self, i, cfg):
        """Return (score array, route array) for session i. route: 1 Continuation, 2 Recovery, 0 none."""
        P = self.P
        u = self.universe(i, cfg)
        cont, rec = P["cont"][i], P["rec"][i]
        route = np.where(cont, 1, np.where(rec, 2, 0))
        if cfg["score"] == "robust" and cfg["sizing"] == "v5":
            sc = np.full(self.N, np.nan)
            for m in (u & cont, u & rec):
                k1, k2 = cfg.get("rp_metrics", ("rs126", "dist200"))
                a = pct_rank(np.where(m, P[k1][i], np.nan))
                b = pct_rank(np.where(m, P[k2][i], np.nan))
                core = 0.5 * a + 0.5 * b
                r = pct_rank(core)
                sc = np.where(m, r, sc)
            return sc, route, u
        if cfg["score"] == "robust":      # India's "robust" option: one pool, no route split
            a = pct_rank(np.where(u, P["rs126"][i], np.nan))
            b = pct_rank(np.where(u, P["dist200"][i], np.nan))
            return pct_rank(0.5 * a + 0.5 * b), route, u
        a = np.where(u, P["m6"][i], np.nan)
        b = np.where(u, P["m12"][i], np.nan)
        z = lambda x: (x - np.nanmean(x)) / np.nanstd(x, ddof=1)
        return (z(a) + z(b)) / 2, route, u


def regime_target(m, i, in_bear, cfg):
    W = cfg["regime_w"] or {}
    ldr = max(LEADERS, key=lambda t: (np.nan_to_num(m.mom126[t][i], nan=-999), -LEADERS.index(t)))

    def defs(n):
        c = [(k, np.nan_to_num(m.mom63[k][i], nan=-999)) for k in DEFENSIVES if m.avail[k][i]]
        if not c:
            return [("cash", 1.0)]
        c.sort(key=lambda x: -x[1])
        top = c[:n]
        return [(k, 1.0 / len(top)) for k, _ in top]

    if not in_bear:
        if m.hyg_ok[i]:
            s, l = W.get("FULL_BULL", (0.90, 0.10))
            return {"stock": s, ldr: l}, "FULL_BULL"
        s, l, dfw = W.get("CREDIT_WARN", (0.70, 0.10, 0.20))
        t = {"stock": s, ldr: l}
        for k, w in defs(1):
            t[k] = t.get(k, 0) + dfw * w
        return t, "CREDIT_WARN"
    if m.t63[i]:
        s = W.get("MILD_BEAR", (0.30,))[0]
        t = {"stock": s}
        for k, w in defs(2):
            t[k] = t.get(k, 0) + (1 - s) * w
        return t, "MILD_BEAR"
    s = W.get("DEEP_BEAR", (0.40,))[0]
    t = {"stock": s}
    for k, w in defs(1):
        t[k] = t.get(k, 0) + (1 - s) * w
    return t, "DEEP_BEAR"


def run(m, cfg=None, start="2001-01-02", end=None, record=False):
    c = dict(CFG)
    c.update(cfg or {})
    cfg = c
    P = m.P
    C, Cf, O, SF = P["C"], P["Cf"], P["O"], P["SF"]
    SMA50, EMA20, ATR, RV30, SMA200 = P["SMA50"], P["EMA20"], P["ATR"], P["RV30"], P["SMA200"]
    i0 = int(m.dates.searchsorted(pd.Timestamp(start)))
    i1 = len(m.dates) if end is None else int(m.dates.searchsorted(pd.Timestamp(end), side="right"))
    v5_size = cfg["sizing"] == "v5"
    use_stop = "stop" in cfg["exit"]
    use_rank = "rank" in cfg["exit"]
    N = cfg["N"]

    cash = float(cfg["capital"])
    pos = {}          # j -> dict(sh (adjusted shares), cost (avg adj px incl. nothing), entry_i, route, stop, rung, below, peak, last, stale)
    etf = {}          # overlay legs: ticker -> shares
    parked = {}       # idle-cash parking: ticker -> shares
    pend_stock = []   # orders for next open
    pend_alloc = None
    alloc = {"stock": 1.0}        # overlay weights in force
    regime = "FULL_BULL"
    in_bear, consec_bull, exit_fired = False, 0, False
    cur_month, mtd_start = None, None
    unit, unit_hist = 1.0, []
    ramp_start_ym = m.ym[i0]
    ramp_applied = None
    curve, expo, events, trades = [], [], [], []
    costs = 0.0
    paused = False
    vf = [1.0]                 # volatility-managed exposure factor (1 = no cap)
    lsma = m.large_sma50 if cfg.get("large_n", 50) == 50 else \
        pd.Series(m.large_idx).rolling(cfg["large_n"], min_periods=cfg["large_n"]).mean().to_numpy()
    core = {"k": None, "w": 0.0}
    thrust_until = [-1]
    if cfg.get("pit_signals"):
        BREADTH, LARGE, LSMA_PIT = m.breadth_pit, m.large_idx_pit, m.large_sma50_pit
    else:
        BREADTH, LARGE, LSMA_PIT = m.breadth, m.large_idx, None
    prev_paused = [False]   # index core: ETF held while it is in a strong uptrend
    rebound = [False]
    park_state = [None]
    pend_core = None
    flow = [0.0, 0.0]
    lots, daily = [], []          # today's stock sells / buys at fill value (for the flow-free unit NAV)
    sv_prev, sleeve_base = 0.0, 0.0
    park_tk = cfg["park"]

    def etf_px(k, i, at="c"):
        e = m.E[k]
        if at == "o":
            v = e["o"][i]
            if v != v:
                v = e["c"][i - 1] if i > 0 else e["c"][i]
            return v
        return e["c"][i]

    def stock_val(i):
        v = 0.0
        for j, p in pos.items():
            px = Cf[i, j]
            if px != px:
                px = p["last"]
            v += p["sh"] * px
        return v

    def etf_val(i):
        return (sum(sh * etf_px(k, i) for k, sh in etf.items() if sh)
                + sum(sh * etf_px(k, i) for k, sh in parked.items() if sh))

    def park_trade(k, n, i):
        """Buy (n > 0) or sell (n < 0) n units of the parking ETF at the open."""
        nonlocal cash, costs
        px = etf_px(k, i, "o")
        if px != px or px <= 0 or n == 0:
            return
        if n > 0:
            while n > 0 and n * px + tcost(n, n * px) > cash:
                n = n - 1 if cfg["whole_etf"] else n * 0.99
            if n <= 0:
                return
            ch = tcost(n, n * px)
            cash -= n * px + ch
            parked[k] = parked.get(k, 0) + n
        else:
            n = min(-n, parked.get(k, 0))
            ch = tcost(n, n * px)
            cash += n * px - ch
            parked[k] -= n
            if parked[k] <= 1e-9:
                del parked[k]
        costs += ch

    def nav(i):
        return cash + stock_val(i) + etf_val(i)

    def ramp_factor(i):
        if not cfg["ramp"] or cfg["ramp"] <= 1:
            return 1.0
        return min(1.0, (m.ym[i] - ramp_start_ym + 1) / cfg["ramp"])

    def apply_ramp(t, f):
        s = {k: w * f for k, w in t.items() if k != "cash"}
        rest = 1.0 - sum(s.values())
        if rest > 1e-9:
            k = "SHY" if m.E["SHY"]["c"][i] == m.E["SHY"]["c"][i] else "cash"
            s[k] = s.get(k, 0) + rest
        return s

    def sell_stock(j, adj_sh, px, i, why):
        nonlocal cash, costs
        p = pos[j]
        adj_sh = min(adj_sh, p["sh"])
        raw_sh = adj_sh / max(SF[i, j], 1e-12)
        val = adj_sh * px
        ch = tcost(raw_sh, val)
        cash += val - ch
        costs += ch
        p["sh"] -= adj_sh
        flow[0] += val
        p["pnl"] = p.get("pnl", 0.0) + adj_sh * (px - p["cost"]) - ch
        p.setdefault("whys", []).append(why)
        if record:
            trades.append((m.dates[i], m.syms[j], "SELL", val, why, px / p["cost"] - 1))
        if p["sh"] <= 1e-9:
            if record:
                lots.append(dict(sym=m.syms[j], entry=m.dates[p["entry_i"]], exit=m.dates[i], held=i - p["entry_i"],
                                 route=p["route"], zone=p.get("meta", {}).get("zone"), score=p.get("meta", {}).get("score"),
                                 q=p.get("meta", {}).get("q"), adds=p["adds"], invested=p.get("inv", 0.0),
                                 pnl=p["pnl"] - p.get("buyc", 0.0), peak_ret=p["peak"] / p["cost"] - 1,
                                 exit_ret=px / p["cost"] - 1, why=p["whys"][-1], n_sells=len(p["whys"]), j=j, exit_i=i))
            del pos[j]

    def buy_stock(j, dollars, i, route, kind, stop, min_val=0.0, meta=None):
        """Whole shares at the open, at the traded price. Returns $ spent."""
        nonlocal cash, costs
        px = O[i, j]
        if px != px or px <= 0:
            return 0.0
        raw_px = px * SF[i, j]
        dollars = min(dollars, cash)
        if cfg["adv_cap"] and P["ADV20"][i - 1, j] == P["ADV20"][i - 1, j]:
            dollars = min(dollars, cfg["adv_cap"] * P["ADV20"][i - 1, j])
        n = int(dollars // raw_px)
        while n > 0 and n * raw_px + tcost(n, n * raw_px) > cash:
            n -= 1
        if n <= 0 or n * raw_px < min_val:
            return 0.0
        val = n * raw_px
        ch = tcost(n, val)
        adj_sh = val / px
        cash -= val + ch
        costs += ch
        flow[1] += val
        if j in pos:
            p = pos[j]
            p["cost"] = (p["cost"] * p["sh"] + val) / (p["sh"] + adj_sh)
            p["sh"] += adj_sh
            p["adds"] += 1
        else:
            pos[j] = dict(sh=adj_sh, cost=px, entry_i=i, route=route, stop=stop, rung=0, below=0,
                          peak=px, last=px, stale=0, adds=0, hold_done=False, meta=meta or {}, px0=px, rung_in=0)
        pos[j]["inv"] = pos[j].get("inv", 0.0) + val
        pos[j]["buyc"] = pos[j].get("buyc", 0.0) + ch
        if record:
            trades.append((m.dates[i], m.syms[j], "BUY", val, kind, 0.0))
        return val + ch

    def trade_etf(k, target_adj_sh, i):
        nonlocal cash, costs
        if k == "cash":
            return
        px = etf_px(k, i, "o")
        if px != px or px <= 0:
            return
        cur = etf.get(k, 0.0)
        delta = target_adj_sh - cur
        if abs(delta) * px < 1.0:
            return
        if delta > 0:
            dollars = min(delta * px, cash)
            n = dollars / px
            if cfg["whole_etf"]:
                # whole shares at the traded price; ETF splits are rare and ignored (SF=1)
                n = math.floor(n)
                while n > 0 and n * px + tcost(n, n * px) > cash:
                    n -= 1
            if n <= 0:
                return
            ch = tcost(n, n * px)
            cash -= n * px + ch
            etf[k] = cur + n
        else:
            n = -delta
            if cfg["whole_etf"] and target_adj_sh > 0:
                n = math.ceil(n - 1e-9)
            n = min(n, cur)
            ch = tcost(n, n * px)
            cash += n * px - ch
            etf[k] = cur - n
        costs += ch

    def set_etfs(weights, i, navv):
        """Rebalance ETF legs to weights x NAV (sells first). Weights exclude 'stock'/'cash'."""
        tg = {}
        for k in set(etf) | set(weights):
            if k in ("stock", "cash"):
                continue
            w = weights.get(k, 0.0)
            px = etf_px(k, i, "o")
            tg[k] = (w * navv / px) if (px == px and px > 0) else 0.0
            if cfg["whole_etf"]:
                tg[k] = math.floor(tg[k])
        for k in sorted(tg, key=lambda k: tg[k] - etf.get(k, 0)):
            trade_etf(k, tg[k], i)

    def scale_down(i, need):
        """WS-5a: proportional whole-share trim, largest first, band 1% of book."""
        sv = sum(p["sh"] * (O[i, j] if O[i, j] == O[i, j] else p["last"]) for j, p in pos.items())
        if sv <= 0 or need < 0.01 * sv:
            return
        ratio = max(0.0, (sv - need) / sv)
        for j in sorted(pos, key=lambda j: -pos[j]["sh"] * pos[j]["last"]):
            p = pos[j]
            px = O[i, j] if O[i, j] == O[i, j] else p["last"]
            raw_held = p["sh"] / SF[i, j]
            k = int(round(raw_held * (1 - ratio)))
            if k <= 0:
                continue
            sell_stock(j, min(p["sh"], k * SF[i, j]), px, i, "scale")
            need -= k * SF[i, j] * px
            if need <= 0:
                break

    def tranche_of(i):
        """In-account staggering: which part of the book is reviewed at session i (None = no review)."""
        for t, d_ in enumerate(cfg.get("stagger") or []):
            if (d_ < 0 and m.month_end[i]) or (d_ >= 0 and m.day_in_month[i] == d_):
                return t
        return None

    def is_decision(i):
        if cfg["cadence"] == "daily":
            return True
        if cfg.get("stagger"):
            return tranche_of(i) is not None
        if cfg["cadence"] == "monthly":
            dd_ = cfg.get("decide_day")
            return m.month_end[i] if dd_ is None or dd_ < 0 else m.day_in_month[i] == dd_
        if cfg["cadence"] == "weekly":
            return m.week_end[i]
        return m.month_end[i] or m.day_in_month[i] == 10

    # -------------------------------------------------------------- the loop
    for i in range(i0, i1):
        d = m.dates[i]
        # ===== (1) at the open: overlay allocation, stock exits, funding, ETF legs, stock buys
        navo = cash + sum(p["sh"] * (Cf[i - 1, j] if Cf[i - 1, j] == Cf[i - 1, j] else p["last"]) for j, p in pos.items()) \
            + sum(sh * etf_px(k, i - 1) for k, sh in etf.items() if sh) \
            + sum(sh * etf_px(k, i - 1) for k, sh in parked.items() if sh)
        if pend_alloc is not None:
            alloc = pend_alloc
            pend_alloc = None
            alloc_changed = True
        else:
            alloc_changed = False
        # WS-9: a new ramp month rescales the allocation at this open
        if cfg["overlay"] in ("v5", "v5_soft") and cfg["ramp"] and cfg["ramp"] > 1:
            f = ramp_factor(i)
            if ramp_applied is not None and f > ramp_applied + 1e-9:
                alloc = apply_ramp(alloc_raw, f)
                ramp_applied = f
                alloc_changed = True
        sells = [o for o in pend_stock if o["kind"] == "exit"]
        buys = [o for o in pend_stock if o["kind"] != "exit"]
        pend_stock = []
        for o in sells:
            j = o["j"]
            if j not in pos:
                continue
            px = O[i, j]
            if px != px or px <= 0:
                px = pos[j]["last"]
            p = pos[j]
            if o.get("all"):
                sh = p["sh"]
            else:
                raw_held = p["sh"] / SF[i, j]
                k = int(raw_held * o.get("frac", cfg["ladder"]))
                sh = p["sh"] if k == 0 else k * SF[i, j]
            sell_stock(j, sh, px, i, o["why"])
        # stock budget
        if cfg["overlay"] == "v5":
            wst = alloc.get("stock", 1.0)
            target_stock = wst * navo
            sv_open = sum(p["sh"] * (O[i, j] if O[i, j] == O[i, j] else p["last"]) for j, p in pos.items())
            if sv_open > target_stock:
                scale_down(i, sv_open - target_stock)
        # park: draw back what today's buys (and a changed allocation) need
        drew = False
        if parked:
            need = sum(o["dollars"] for o in buys)
            if cfg["overlay"] in ("v5", "v5_soft"):
                sv_o = sum(p["sh"] * (O[i, j] if O[i, j] == O[i, j] else p["last"]) for j, p in pos.items())
                need = min(need, max(0.0, alloc.get("stock", 1.0) * navo - sv_o))
            need += navo if alloc_changed else 0.0
            drew = need > cash
            for k in list(parked):
                if need <= cash:
                    break
                px = etf_px(k, i, "o")
                chunk = max((need - cash) * 1.01, 0.0 if not v5_size else 0.05 * navo)
                sell_n = parked[k] if alloc_changed else min(parked[k], math.ceil(chunk / px))
                park_trade(k, -sell_n, i)
        if alloc_changed and cfg["overlay"] in ("v5", "v5_soft"):
            set_etfs({k: w for k, w in alloc.items() if k != "stock"}, i, navo)
        # stock buys (new first by rank, then adds, India order)
        # index core: switch or exit at this open
        if pend_core is not None:
            k_new, w_new = pend_core
            pend_core = None
            for k in list(parked):
                park_trade(k, -parked[k], i)
            for k in [k for k in etf if etf[k] and k != k_new]:
                trade_etf(k, 0.0, i)
            if k_new:
                px = etf_px(k_new, i, "o")
                if px == px and px > 0:
                    trade_etf(k_new, math.floor(w_new * navo / px), i)
            core["k"], core["w"] = k_new, (w_new if k_new else 0.0)
        cw = core["w"] if cfg["overlay"] not in ("v5", "v5_soft") else 0.0
        sleeve_cash_cap = None
        if cfg["overlay"] in ("v5", "v5_soft"):
            wst = alloc.get("stock", 1.0)
            sv_open = sum(p["sh"] * (O[i, j] if O[i, j] == O[i, j] else p["last"]) for j, p in pos.items())
            sleeve_cash_cap = max(0.0, wst * navo - sv_open)
        elif ramp_factor(i) * vf[0] * (1 - cw) < 1:
            sv_open = sum(p["sh"] * (O[i, j] if O[i, j] == O[i, j] else p["last"]) for j, p in pos.items())
            cap_v = ramp_factor(i) * vf[0] * (1 - cw) * navo
            if cfg.get("vol_hard") and sv_open > 1.05 * cap_v:
                scale_down(i, sv_open - cap_v)
                sv_open = sum(p["sh"] * (O[i, j] if O[i, j] == O[i, j] else p["last"]) for j, p in pos.items())
            sleeve_cash_cap = max(0.0, cap_v - sv_open)
        for o in buys:
            j = o["j"]
            dollars = o["dollars"]
            if sleeve_cash_cap is not None:
                dollars = min(dollars, sleeve_cash_cap)
            if o["kind"] == "new" and j in pos:
                continue
            if o["kind"] == "add" and j not in pos:
                continue
            if dollars <= 0:
                continue
            min_d = o.get("min_dollars", 0.0)
            if dollars < min_d:
                continue
            spent = buy_stock(j, dollars, i, o["route"], o["kind"], o.get("stop"), min_d, o.get("meta"))
            if sleeve_cash_cap is not None:
                sleeve_cash_cap -= spent
        # park idle cash (India: at least half a slot idle)
        if park_tk and not drew:
            # money held back by the volatility cap must not ride the market in an index ETF
            risk_off = paused if cfg.get("safe_when", "paused") == "paused" else (not m.b200[i])
            if cfg.get("park_on_decision"):
                # only re-decide the parking ETF on a decision day; hold the last choice in between
                if is_decision(i - 1) or park_state[0] is None:
                    park_state[0] = risk_off
                risk_off = park_state[0]
            pk = _park_name(park_tk, m, i, cfg, risk_off or (bool(cfg.get("vol_target") or cfg.get("mkt_vol_target")) and vf[0] < 0.999))
            # rotate: a parking ETF that is no longer the target is sold first
            for k in [k for k in list(parked) if k != pk]:
                park_trade(k, -parked[k], i)
            # park only a meaningful idle balance and keep a buffer, or the account churns a
            # $1-minimum round trip every session (v5 buys a little most days)
            thresh, keep = (0.5 * navo / N, 0.01 * navo) if not v5_size else (0.10 * navo, 0.04 * navo)
            if pk and cash > max(thresh + keep, 50.0):
                px = etf_px(pk, i, "o")
                if px == px and px > 0:
                    n = (cash - keep) / px
                    n = math.floor(n) if cfg["whole_etf"] else n
                    park_trade(pk, n, i)

        # ===== (2) at the close: mark, delisting, unit NAV
        for j in list(pos):
            p = pos[j]
            px = C[i, j]
            if px != px:
                p["stale"] += 1
                if cfg["delist_cash"] and p["stale"] >= cfg["delist_cash"]:
                    sell_stock(j, p["sh"], p["last"], i, "delisted")
            else:
                p["stale"] = 0
                p["last"] = px
                p["peak"] = max(p["peak"], px)
        sv = stock_val(i)
        navc = cash + sv + etf_val(i)
        # sleeve unit NAV: stock P&L over yesterday's sleeve value (time-weighted, flow-free)
        if unit_hist and sleeve_base > 1.0:
            pnl = sv + flow[0] - flow[1] - sv_prev
            unit *= 1 + pnl / sleeve_base
        flow[0] = flow[1] = 0.0
        sv_prev = sv
        unit_hist.append(unit)
        # volatility-managed exposure (Barroso & Santa-Clara): the stock book may use at most
        # vol_target / (its own realised volatility over 63 sessions) of the account
        if cfg.get("mkt_vol_target"):
            sp_ = m.E["SPY"]["c"]
            if i >= 64:
                rr_ = np.diff(sp_[i - 63:i + 1]) / sp_[i - 63:i]
                rv = float(np.nanstd(rr_, ddof=1) * np.sqrt(252))
                vf[0] = min(1.0, max(cfg.get("vol_floor", 0.3), cfg["mkt_vol_target"] / rv)) if rv > 0 else 1.0
        elif cfg.get("vol_target") and len(unit_hist) > 64:
            u = np.array(unit_hist[-64:])
            rv = np.std(np.diff(u) / u[:-1], ddof=1) * np.sqrt(252)
            vf[0] = min(1.0, max(cfg.get("vol_floor", 0.3), cfg["vol_target"] / rv)) if rv > 0 else 1.0
        curve.append(navc)
        if record:
            daily.append((d, navc, sv, sum(sh * etf_px(k, i) for k, sh in parked.items()),
                          sum(sh * etf_px(k, i) for k, sh in etf.items() if sh), cash, paused, len(pos),
                          sum(sh * etf_px(k, i) for k, sh in parked.items() if k == "SHY"), unit))
        expo.append(sv / navc if navc > 0 else 0.0)
        wst_now = alloc.get("stock", 1.0) if cfg["overlay"] in ("v5", "v5_soft") else 1.0
        sleeve_base = sv if cfg["overlay"] not in ("v5", "v5_soft") else max(wst_now * navc, sv)

        # ===== (3) at the close: overlay signals (v5_engine 8.2)
        if cfg["overlay"] in ("v5", "v5_soft"):
            ym = m.ym[i]
            if ym != cur_month:
                cur_month, mtd_start, exit_fired = ym, unit, False
            b200, hyg_ok = m.b200[i], m.hyg_ok[i]
            slv_mtd = unit / mtd_start - 1
            sl_val = (unit_hist[-2] / unit_hist[-12] - 1) if len(unit_hist) >= 12 else 0.0
            if in_bear:
                consec_bull = consec_bull + 1 if b200 else 0
            tgt = None
            if cfg["intramonth"] and not in_bear and not exit_fired and not m.month_end[i]:
                if slv_mtd < -0.12 and hyg_ok:
                    in_bear, consec_bull, exit_fired = True, 0, True
                    tgt = regime_target(m, i, True, cfg)
            if m.month_end[i]:
                if not b200:
                    in_bear, consec_bull = True, 0
                    tgt = regime_target(m, i, True, cfg)
                elif not in_bear:
                    tgt = regime_target(m, i, False, cfg)
                elif hyg_ok and consec_bull >= 5 and sl_val >= 0.03:
                    in_bear, consec_bull = False, 0
                    tgt = regime_target(m, i, False, cfg)
                else:
                    tgt = regime_target(m, i, True, cfg)
            elif cfg["intramonth"] and in_bear and b200:
                if hyg_ok and consec_bull >= 5 and sl_val >= 0.03:
                    in_bear, consec_bull = False, 0
                    tgt = regime_target(m, i, False, cfg)
            if i == i0:
                tgt = regime_target(m, i, False, cfg)     # A8: start invested (bull) on day one
            if tgt is not None:
                alloc_raw, regime = tgt
                f = ramp_factor(i)
                pend_alloc = apply_ramp(alloc_raw, f) if f < 1 else dict(alloc_raw)
                ramp_applied = f
                if record:
                    events.append((d, regime))
        elif i == i0:
            pend_alloc, alloc_raw = {"stock": 1.0}, {"stock": 1.0}

        # index core decision: hold core_w in the leading index ETF while its core_lb-session return
        # is above core_min and it closes above its 50-session average; exit as soon as that fails
        if cfg.get("core_w") and cfg["overlay"] not in ("v5", "v5_soft"):
            lb = cfg.get("core_lb", 126)
            def cm(t):
                c = m.E[t]["c"]
                return c[i] / c[i - lb] - 1 if i - lb >= 0 and c[i] == c[i] and c[i - lb] == c[i - lb] else -999
            ldr = max(LEADERS, key=cm)
            strong = cm(ldr) > cfg.get("core_min", 0.15) and m.E[ldr]["c"][i] > m.e_sma50[ldr][i]
            if cfg.get("core_only_smh_qqq", True) and ldr == "SPY":
                strong = False
            if cfg.get("core_mode") == "rebound":
                # rebound state: SPY fell at least rebound_trig from its 1-year high within the last
                # 126 sessions and now closes above its 50-session average
                lo = np.nanmin(m.spy_dd[max(0, i - 126):i + 1])
                strong = lo <= -cfg.get("rebound_trig", 0.15) and m.E["SPY"]["c"][i] > m.e_sma50["SPY"][i] \
                    and m.E[ldr]["c"][i] > m.e_sma50[ldr][i]
                rebound[0] = strong
            want = ldr if strong else None
            held_ok = core["k"] is not None and m.E[core["k"]]["c"][i] > m.e_sma50[core["k"]][i]
            if cfg.get("core_mode") == "fixed":
                # a fixed share of the account in one index ETF (core_etf), rebalanced on decision days when
                # it drifts more than core_band from target; core_trend: only while the ETF is above its
                # 200-session average (checked on decision days)
                ke = cfg.get("core_etf", "QQQ")
                c = m.E[ke]["c"]
                ok = c[i] == c[i]
                if cfg.get("core_trend") and ok:
                    s200 = np.nanmean(c[max(0, i - 199):i + 1])
                    ok = i >= 199 and c[i] > s200
                want = ke if ok else None
                held_ok = not (want is None and is_decision(i))   # trend exit on decision days only
                if want and core["k"] == want and is_decision(i):
                    cur = etf.get(ke, 0) * etf_px(ke, i) / navc if navc > 0 else 0.0
                    if abs(cur - cfg["core_w"]) > cfg.get("core_band", 0.05):
                        pend_core = (want, cfg["core_w"])
                if want != core["k"] and not is_decision(i) and core["k"] is not None:
                    want = core["k"]                            # changes only on decision days
            if want != core["k"]:
                if want is None and not held_ok:
                    pend_core = (None, 0.0)                     # fast exit, any session
                elif want is not None and (is_decision(i) or core["k"] is None):
                    pend_core = (want, cfg["core_w"])

        # India pause (checked every session; acts only on decision days)
        if cfg["overlay"] == "pause" or "pause" in str(cfg.get("extra_pause", "")):
            r = cfg["pause_rules"]
            ls_ = LSMA_PIT if LSMA_PIT is not None else lsma
            sig = [("breadth" in r and BREADTH[i] < cfg.get("breadth_level", 0.40)), ("hyg" in r and not m.hyg_ok[i]),
                   ("large" in r and ls_[i] == ls_[i] and LARGE[i] < ls_[i])]
            # index / credit / bond votes: "<ETF><n>" = that ETF closes below its n-session average
            # (e.g. spy200, qqq100, smh50, hyg63); "tlt_up" = TLT above its 63-session average (flight to safety)
            for tok in r:
                if tok in ("breadth", "hyg", "large"):
                    continue
                if tok == "tlt_up":
                    sig.append(m.etf_vs_sma("TLT", 63, i) > 0)
                elif tok.startswith("vix") and not tok.startswith("vixrel"):   # e.g. vix25: VIX closes above 25
                    v_ = m.P.get("VIX")
                    sig.append(v_ is not None and v_[i] == v_[i] and v_[i] > float(tok[3:]))
                elif tok.endswith("macd"):           # e.g. spymacd: SPY MACD below its signal line
                    sig.append(m.etf_macd(tok[:3].upper(), i) < 0)
                elif tok.startswith("vixrel"):       # e.g. vixrel1.3: VIX 30% above its 50-session average
                    sig.append(m.vix_rel[i] == m.vix_rel[i] and m.vix_rel[i] > float(tok[6:]))
                elif tok == "credit":                # high yield losing to Treasuries (HYG/IEF below its 50-day)
                    sig.append(bool(m.credit_bad[i]))
                elif tok == "momcrash":              # market down over two years: momentum-crash risk
                    sig.append(bool(m.mom_crash[i]))
                elif tok.startswith("b50_"):         # e.g. b50_20: under 20% of stocks above their 50-day
                    sig.append(m.b50[i] < float(tok[4:]) / 100)
                else:
                    t_, n_ = tok[:3].upper(), int(tok[3:])
                    sig.append(m.etf_vs_sma(t_, n_, i) < 0)
            # "any": one signal pauses (India v8); "two": two must agree, so buying resumes sooner
            paused = sum(sig) >= (cfg["pause_k"] if cfg.get("pause_k") else (2 if cfg.get("pause_mode", "any") == "two" else 1))
            # own-book stress (owner, 15 Oct 2026): pause on OUR stock book's own drawdown / trend instead of, or
            # together with, the market signals. unit = the stock book's flow-free value.
            bp = cfg.get("book_pause")
            if bp:
                u_ = unit_hist
                bs = False
                if len(u_) > 21:
                    if bp.get("dd"):
                        lb_ = bp.get("dd_lb", 63)
                        bs |= u_[-1] / max(u_[-lb_:]) - 1 < -bp["dd"]
                    if bp.get("ma") and len(u_) > bp["ma"]:
                        bs |= u_[-1] < float(np.mean(u_[-bp["ma"]:]))
                mode = bp.get("mode", "replace")
                paused = bs if mode == "replace" else ((paused and bs) if mode == "and" else (paused or bs))
            # breadth thrust: a broad surge off a washout ends any pause for thrust_days sessions
            if cfg.get("thrust_days"):
                if m.thrust[i]:
                    thrust_until[0] = i + cfg["thrust_days"]
                if i <= thrust_until[0]:
                    paused = False
            # a pause only counts inside a real bear market (SPY below its 200-session average)
            if cfg.get("pause_needs_bear") and m.b200[i]:
                paused = False

        # ===== (4) at the close: stock exits and buys for the next open
        exiting = set()
        if cfg.get("hard_stop") or (cfg.get("pause_cut") and paused):
            for j, p in pos.items():
                if p["stale"] or j in exiting or C[i, j] != C[i, j]:
                    continue
                if cfg.get("hard_stop") and C[i, j] < p["cost"] * (1 - cfg["hard_stop"]):
                    pend_stock.append(dict(kind="exit", j=j, why="hard", all=True)); exiting.add(j)
                elif cfg.get("pause_cut") and paused and C[i, j] < p["cost"] * (1 - cfg.get("pause_cut_loss", 0.0)):
                    pend_stock.append(dict(kind="exit", j=j, why="pausecut", all=True)); exiting.add(j)
        if cfg.get("udvr_exit"):
            # distribution: sell a profitable holding when down-day volume dominates (UDVR below udvr_exit)
            for j, p in pos.items():
                if p["stale"] or j in exiting or C[i, j] != C[i, j] or i - p["entry_i"] < 21:
                    continue
                if m.P["UDVR"][i, j] < cfg["udvr_exit"] and C[i, j] > p["cost"] * 1.1:
                    pend_stock.append(dict(kind="exit", j=j, why="distribution", all=True)); exiting.add(j)
        if cfg.get("adx_exit"):
            # sell a profitable holding when its trend weakens: ADX below adx_exit, or sellers take over (-DI > +DI)
            for j, p in pos.items():
                if p["stale"] or j in exiting or C[i, j] != C[i, j] or i - p["entry_i"] < 21:
                    continue
                weak = (m.P["ADX"][i, j] < cfg["adx_exit"]) if cfg["adx_exit"] > 0 else (not m.P["PDI_GT"][i, j])
                if weak and C[i, j] > p["cost"] * 1.1:
                    pend_stock.append(dict(kind="exit", j=j, why="adx_exit", all=True)); exiting.add(j)
        if cfg.get("macd_exit") is not None:
            # sell a profitable holding when its MACD histogram falls below macd_exit (momentum rolling over)
            for j, p in pos.items():
                if p["stale"] or j in exiting or C[i, j] != C[i, j]:
                    continue
                if m.P["MACDH"][i, j] < cfg["macd_exit"] and C[i, j] > p["cost"] * (1 + cfg.get("macd_exit_min_gain", 0.1)) \
                        and i - p["entry_i"] >= 21:
                    fr = cfg.get("macd_exit_frac", 1.0)
                    pend_stock.append(dict(kind="exit", j=j, why="macd_exit", all=fr >= 0.999, frac=fr)); exiting.add(j)
        if cfg.get("rsi_take"):
            # overbought take-profit (owner, 15 Oct 2026): a holding in profit whose 14-day RSI is above rsi_take
            # sells rsi_take_frac of itself (once per run-up; re-armed when RSI falls back below 60)
            for j, p in pos.items():
                if p["stale"] or j in exiting or C[i, j] != C[i, j]:
                    continue
                rs_ = m.P["RSI"][i, j]
                if rs_ < 60:
                    p["rsi_armed"] = True
                if rs_ > cfg["rsi_take"] and p.get("rsi_armed", True) and C[i, j] > p["cost"] * (1 + cfg.get("rsi_take_min_gain", 0.1)):
                    fr = cfg.get("rsi_take_frac", 0.5)
                    pend_stock.append(dict(kind="exit", j=j, why="rsi_take", all=fr >= 0.999, frac=fr))
                    p["rsi_armed"] = False
                    exiting.add(j)
        if cfg.get("lock_at"):
            for j, p in pos.items():
                if p["stale"] or j in exiting:
                    continue
                pg = p["peak"] / p["cost"] - 1
                trail = cfg["lock_trail"]
                # lock_tiers: [(gain, trail), ...] ascending; the biggest gain reached sets the give-back allowed
                for g_, t_ in cfg.get("lock_tiers") or []:
                    if pg >= g_:
                        trail = t_
                if cfg.get("lock_half") and p.get("locked"):
                    trail = cfg.get("lock_trail2", trail)
                e50_ = m.P["EMA50"]
                if cfg.get("lock_trend") and C[i, j] > e50_[i, j] and e50_[i, j] > e50_[i - 10, j]:
                    continue                                    # journal: lock sells of intact trends kept rising
                if pg >= cfg["lock_at"] and C[i, j] < p["peak"] * (1 - trail):
                    if cfg.get("lock_half") and not p.get("locked"):
                        # first trigger sells half, the rest rides on a wider trail from a fresh peak
                        pend_stock.append(dict(kind="exit", j=j, why="lock1", all=False, frac=0.5))
                        p["locked"], p["peak"] = True, C[i, j]
                    else:
                        pend_stock.append(dict(kind="exit", j=j, why="lock", all=True))
                    exiting.add(j)
        if cfg.get("chand_k"):
            # chandelier: sell when the close falls more than k ATRs below the highest close since entry
            for j, p in pos.items():
                if p["stale"] or j in exiting or C[i, j] != C[i, j]:
                    continue
                a = ATR[i, j]
                if a == a and a > 0 and i - p["entry_i"] >= cfg.get("chand_min_hold", 5) and C[i, j] < p["peak"] - cfg["chand_k"] * a:
                    pend_stock.append(dict(kind="exit", j=j, why="chand", all=True)); exiting.add(j)
        if use_stop:
            for j, p in pos.items():
                if p["stale"] or j in exiting:
                    continue
                route = p["route"]
                a = ATR[i, j]
                if a == a and a > 0:
                    base = SMA50[i, j] if route == 1 else EMA20[i, j]
                    live = base - (1.0 if route == 1 else 0.5) * a if base == base else None
                else:
                    live = None
                held = i - p["entry_i"]
                floor_ = cfg["min_hold"]
                # framework 19.8's own option: a position below its cost may exit after mh_loser_hold
                if cfg.get("mh_profit_only") and C[i, j] < p["cost"]:
                    floor_ = cfg.get("mh_loser_hold", 5)
                if held < floor_:
                    if live is not None:
                        p["stop"] = live if p["stop"] is None or p["stop"] != p["stop"] else max(p["stop"], live)
                    p["below"] = 0
                    continue
                if live is not None:
                    p["stop"] = live if p["stop"] is None or p["stop"] != p["stop"] else max(p["stop"], live)
                st = p["stop"]
                if st is None or st != st:
                    continue
                if C[i, j] < st:
                    pend_stock.append(dict(kind="exit", j=j, why="stop"))
                    exiting.add(j)
                else:
                    p["below"] = 0
        # re-enter at the first session after a pause ends instead of waiting for the next decision day
        fast_re = bool(cfg.get("reenter_fast") and prev_paused[0] and not paused)
        prev_paused[0] = paused
        if not is_decision(i) and not fast_re:
            continue
        navd = navc
        sc, route, u = m.scores(i, cfg)
        if cfg.get("dm_guard") and m.mom_crash[i] and m.etf_vs_sma("SPY", 50, i) > 0:
            # Daniel-Moskowitz: after a bear market, in a rebound, rank by short-term recovery instead of 6-month momentum
            sc = np.where(np.isnan(sc), np.nan, pct_rank(np.where(u & ~np.isnan(sc), m.P["mom63"][i].astype(float), np.nan)))
        if (cfg.get("sector_top_k") or cfg.get("sector_w")) and m.sector is not None:
            lbs = cfg.get("sector_lb", 126)
            smom = []
            for t_ in m.sector_list:
                jj = list(m.syms).index(t_) if t_ in m.syms else None
                c_ = m.P["Cf"][:, jj] if jj is not None else None
                smom.append(c_[i] / c_[i - lbs] - 1 if c_ is not None and i >= lbs and c_[i] == c_[i] and c_[i - lbs] == c_[i - lbs] else -9)
            smom = np.array(smom)
            sid = m.sector[i]
            ok_s = sid >= 0
            if cfg.get("sector_top_k"):
                top = np.argsort(-smom)[:cfg["sector_top_k"]]
                good = ok_s & np.isin(sid, top)
                sc = np.where(good | m.is_etf, sc, np.nan)          # new buys and rank only inside leading industries
            if cfg.get("sector_w"):
                sp_ = pct_rank(smom)
                sv_ = np.where(ok_s, sp_[np.clip(sid, 0, None)], 0.0)
                w_ = cfg["sector_w"]
                sc = np.where(np.isnan(sc), np.nan, (sc + w_ * sv_) / (1 + w_))
        if cfg.get("narrow_size_w") and m.narrow[i]:
            # narrow market: lean the ranking toward the biggest companies (they are carrying the index)
            w_ = cfg["narrow_size_w"]
            sc = np.where(np.isnan(sc), np.nan, (sc + w_ * np.where(u, m.size_pct[i], 0.0)) / (1 + w_))
        for key_, w_ in (cfg.get("score_mix") or {}).items():
            # blend the rank score with another indicator's cross-sectional percentile (hi52, resmom, ADX, RSI)
            x_ = pct_rank(np.where(u & ~np.isnan(sc), m.P[key_][i].astype(float), np.nan))
            sc = np.where(np.isnan(sc), np.nan, (sc + w_ * np.nan_to_num(x_, nan=0.0)) / (1 + w_))
        if use_rank or not v5_size:
            order = np.argsort(-np.nan_to_num(sc, nan=-np.inf))
            nvalid = int((~np.isnan(sc)).sum())
            pos_rank = {int(j): k for k, j in enumerate(order[:nvalid])}
        if use_rank and cfg.get("exit_score") == "pool":
            P_ = m.P
            a_ = pct_rank(np.where(u, P_["rs126"][i], np.nan))
            b_ = pct_rank(np.where(u, P_["dist200"][i], np.nan))
            ps = pct_rank(0.5 * a_ + 0.5 * b_)
            order = np.argsort(-np.nan_to_num(ps, nan=-np.inf))
            pos_rank = {int(j): k for k, j in enumerate(order[:int((~np.isnan(ps)).sum())])}
        if use_rank:
            xr = cfg["buffer"] * N
            if paused and cfg.get("pause_tight_exit", True):
                xr = N                                          # hard-coded India rule: a pause tightens the sell rank
            t_now = tranche_of(i) if cfg.get("stagger") else None
            for j in list(pos):
                if j in exiting or pos[j]["stale"]:
                    continue
                if t_now is not None and pos[j].get("meta", {}).get("tr", 0) != t_now:
                    continue
                if (i - pos[j]["entry_i"]) < cfg.get("rank_min_hold", 0):
                    continue
                if cfg.get("rank_exempt_gain") and not paused and C[i, j] / pos[j]["cost"] - 1 >= cfg["rank_exempt_gain"]:
                    continue
                if cfg.get("udvr_hold") and not paused and m.P["UDVR"][i, j] > cfg["udvr_hold"]:
                    continue                                    # still being accumulated: keep it despite the rank
                if cfg.get("adx_hold") and not paused and m.P["ADX"][i, j] > cfg["adx_hold"] and m.P["PDI_GT"][i, j]:
                    continue                                    # strong uptrend (ADX high, buyers in charge): keep it
                if cfg.get("macd_hold") is not None and not paused and m.P["MACDH"][i, j] > cfg["macd_hold"]:
                    continue                                    # MACD still rising: keep it despite the rank
                if cfg.get("rsi_sell_floor") and not paused and m.P["RSI"][i, j] < cfg["rsi_sell_floor"]:
                    continue                                    # journal: rank sells at oversold RSI bounced afterwards
                if cfg.get("trend_hold") and not paused:
                    # keep a holding while its own trend is intact: above a rising 50-session EMA
                    e50 = m.P["EMA50"]
                    if C[i, j] > e50[i, j] and i >= 10 and e50[i, j] > e50[i - 10, j] and \
                            (cfg["trend_hold"] is True or C[i, j] / pos[j]["cost"] - 1 >= cfg["trend_hold"]):
                        continue
                rl = cfg.get("rank_ladder")
                if rl and not paused:
                    rk = pos_rank.get(j, 10 ** 9)
                    if rk >= rl * N:
                        pend_stock.append(dict(kind="exit", j=j, why="rank", all=True)); exiting.add(j)
                    elif rk >= xr and not pos[j].get("halved"):
                        pend_stock.append(dict(kind="exit", j=j, why="rank1", all=False, frac=0.5))
                        pos[j]["halved"] = True; exiting.add(j)
                    continue
                if pos_rank.get(j, 10 ** 9) >= xr:
                    p_ = pos[j]
                    p_["out_n"] = p_.get("out_n", 0) + 1
                    if cfg.get("rank_losers_only") and C[i, j] >= p_["cost"]:
                        continue
                    if p_["out_n"] < cfg.get("rank_persist", 1):
                        continue
                    pend_stock.append(dict(kind="exit", j=j, why="rank", all=True))
                    exiting.add(j)
                else:
                    pos[j]["out_n"] = 0
        if paused:
            continue
        park_v = sum(sh * etf_px(k, i) for k, sh in parked.items())
        if cfg["overlay"] in ("v5", "v5_soft"):
            wst = (pend_alloc or alloc).get("stock", 1.0)
            sleeve_nav = wst * navd
            sleeve_cash = min(cash + park_v, max(0.0, sleeve_nav - sv))
        else:
            f = ramp_factor(i) * vf[0] * (1 - (core["w"] if core["k"] else 0.0))
            sleeve_nav = navd
            sleeve_cash = cash + park_v if f >= 1 else min(cash + park_v, max(0.0, f * navd - sv))
        # projected cash: + queued exits
        for o in pend_stock:
            if o["kind"] == "exit" and o["j"] in pos:
                p = pos[o["j"]]
                frac = 1.0 if o.get("all") else o.get("frac", cfg["ladder"])
                sleeve_cash += p["sh"] * frac * C[i, o["j"]] * 0.998
        if v5_size:
            ocfg = cfg if not (rebound[0] and cfg.get("rec_cut_rebound")) else dict(cfg, rec_cut=cfg["rec_cut_rebound"])
            _v5_orders(m, ocfg, i, sc, route, u, pos, exiting, sleeve_nav, sleeve_cash, pend_stock)
        else:
            scfg = cfg if not cfg.get("stagger") else dict(cfg, _tr=tranche_of(i), _ntr=len(cfg["stagger"]))
            _slot_orders(m, scfg, i, sc, route, u, pos, exiting, navd, sleeve_cash, pend_stock, pos_rank)

    out = dict(curve=pd.Series(curve, index=m.dates[i0:i1]), expo=pd.Series(expo, index=m.dates[i0:i1]),
               costs=costs, final_positions=len(pos))
    if record:
        out["events"] = events
        out["trades"] = pd.DataFrame(trades, columns=["date", "sym", "side", "value", "why", "ret"])
        out["lots"] = pd.DataFrame(lots)
        out["daily"] = pd.DataFrame(daily, columns=["date", "nav", "stocks", "parked", "core", "cash", "paused", "npos", "shy", "unit"]).set_index("date")
    return out


_PARKABLE = ("GLD", "SHY", "BIL", "SPY", "QQQ", "SMH", "IEF")


def _park_name(park, m, i, cfg=None, paused=False):
    """Where idle cash goes. "leader": the index ETF (SPY/QQQ/SMH) with the best return over
    cfg["leader_lb"] sessions to yesterday's close. "leader_safe": the same, but SHY while the
    buy pause is on, so parked cash does not ride a falling market."""
    if park in ("leader", "leader_safe"):
        if park == "leader_safe" and paused:
            return "SHY" if m.E["SHY"]["c"][i] == m.E["SHY"]["c"][i] else None
        lb = (cfg or {}).get("leader_lb", 126)
        def mom(t):
            c = m.E[t]["c"]
            a, b = c[i - 1], c[i - 1 - lb] if i - 1 - lb >= 0 else np.nan
            return a / b - 1 if (a == a and b == b and b > 0) else -999
        return max(LEADERS, key=mom)
    k = park
    if np.isnan(m.E[k]["raw"][i]) and np.isnan(m.E[k]["c"][i]):
        return None
    return k


def _v5_orders(m, cfg, i, sc, route, u, pos, exiting, nav, cash, out):
    """19.11 + 19.12 + 19.6 on session i's close."""
    P = m.P
    C, ATR, SMA50, EMA20, RV30, ADV = P["C"], P["ATR"], P["SMA50"], P["EMA20"], P["RV30"], P["ADV20"]
    cand = []
    q = (P["Cf"][i] - EMA20[i]) / ATR[i]
    ok = u & ~np.isnan(sc)
    cut = np.where(route == 1, cfg["cont_cut"], cfg["rec_cut"])
    ok &= (route > 0) & (sc >= cut)
    for j in np.nonzero(ok)[0]:
        j = int(j)
        qq = q[j]
        zone = ("Green" if -0.5 <= qq <= 1.0 else "Yellow" if (1.0 < qq <= 2.0 or -1.0 <= qq < -0.5)
                else "Red" if qq > 2.0 else "Weak") if qq == qq else "Unknown"
        if j in pos:
            if route[j] != pos[j]["route"] or cfg["adds"] != "v5":
                continue
            if zone != "Green":
                # framework 10.1 add_red_hi: an extended held winner may take a Yellow-sized add
                hi = cfg.get("add_red_hi")
                if hi is None or not (qq == qq and -1.0 <= qq <= hi):
                    continue
                zone = "Yellow"
            kind = "add"
        else:
            if cfg["zone_gate"] and zone not in ("Green", "Yellow"):
                continue
            if not cfg["zone_gate"] and zone in ("Unknown",):
                continue
            if zone not in ("Green", "Yellow"):
                zone = "Yellow"
            kind = "new"
        cand.append((-sc[j], route[j] != 1, j, zone, kind))
    # leader tilt: while SMH (or QQQ) is the leading index, names that move with it get priority
    # and a bigger position
    tilt = {}
    if cfg.get("tilt_mult", 1.0) != 1.0:
        lb = cfg.get("leader_lb", 126)
        def mom(t):
            c = m.E[t]["c"]
            return c[i] / c[i - lb] - 1 if i - lb >= 0 and c[i] == c[i] and c[i - lb] == c[i - lb] else -999
        ldr = max(LEADERS, key=mom)
        if ldr in cfg.get("tilt_leaders", ("SMH",)):
            cr = P["corr_" + ldr][i]
            for k, (a0, a1, j, z, kd) in enumerate(cand):
                if cr[j] == cr[j] and cr[j] >= cfg.get("tilt_corr", 0.7):
                    tilt[j] = cfg["tilt_mult"]
                    cand[k] = (a0 - 1.0, a1, j, z, kd)      # moves ahead of every untilted name
    cand.sort()
    mp = cfg.get("max_pos")
    n_held = len([j for j in pos if j not in exiting])
    sm = cfg.get("size_mult", 1.0)
    for _, _, j, zone, kind in cand:
        if cash <= 0:
            break
        if kind == "new" and mp is not None and n_held >= mp:
            continue
        close = C[i, j]
        rt = int(route[j])
        stop_line = (SMA50[i, j] - ATR[i, j]) if rt == 1 else (EMA20[i, j] - 0.5 * ATR[i, j])
        held = pos.get(j)
        stop = held["stop"] if held else stop_line
        sh = held["sh"] if held else 0.0
        avg = held["cost"] if held else 0.0
        em = cfg.get("etf_size_mult", 1.0) if m.is_etf[j] else 1.0
        em *= tilt.get(j, 1.0)
        base = (0.08 if zone == "Green" else 0.04) * nav * sm * em * cfg.get("base_mult", 1.0)
        target = min(base, cash)
        rb = min(0.015 if zone == "Green" else 0.010, 0.02) * sm * em * cfg.get("risk_mult", 1.0)
        if stop is None or stop != stop or close <= stop:
            continue
        eff = sh
        if sh:
            if stop >= avg:
                eff = 0.0
            elif close > avg:
                eff = max(sh - (close - avg) * sh / (close - stop), 0.0)
        rr = (rb * nav) / (close - stop) - eff
        target = min(target, rr * close)
        rv = RV30[i, j]
        if rv == rv and rv > 0:
            vr = ((0.03 if zone == "Green" else 0.015) * nav * sm * em * cfg.get("vol_mult", 1.0)) / (3.0 * rv * close) - sh
            target = min(target, vr * close)
        raw_px = close * P["SF"][i, j]
        n = int(target // raw_px) if raw_px > 0 else 0
        if n <= 0:
            continue
        dollars = n * raw_px
        if kind == "new" and dollars < 0.005 * nav:
            continue
        out.append(dict(kind=kind, j=j, dollars=dollars, route=rt, stop=stop_line if kind == "new" else None,
                        min_dollars=0.005 * nav if kind == "new" else 0.0,
                        meta=dict(zone=zone, score=float(sc[j]), q=float(q[j]))))
        cash -= dollars + tcost(n, dollars)
        if kind == "new":
            n_held += 1


def _slot_orders(m, cfg, i, sc, route, u, pos, exiting, nav, cash, out, pos_rank):
    """India v8 review: fill free slots best-ranked first, then add to winners."""
    P = m.P
    N = cfg["N"]
    na = cfg.get("N_adapt")
    if na:
        # adaptive book: fewer, larger slots when breadth is strong; more, smaller ones when weak
        N = na[0] if m.breadth[i] >= na[2] else na[1]
    n_stock = N
    if cfg["gold_slots"]:
        g = m.E["GLD"]["c"]
        if i >= 200 and g[i] == g[i] and g[i - 126] == g[i - 126]:
            g126 = g[i] / g[i - 126] - 1
            uni_med = np.nanmedian(np.where(u, P["Cf"][i] / P["Cf"][i - 126] - 1, np.nan))
            if g126 > uni_med and g[i] > np.nanmean(g[i - 200:i]):
                n_stock = N - cfg["gold_slots"]
    slot = nav / N
    tr, ntr = cfg.get("_tr"), cfg.get("_ntr", 1)
    if tr is None:
        held_after = len([j for j in pos if j not in exiting])
        free = n_stock - held_after
    else:
        # this tranche owns its share of the slots; the other tranche's holdings are not touched
        mine = [j for j in pos if j not in exiting and pos[j].get("meta", {}).get("tr", 0) == tr]
        free = int(round(n_stock / ntr)) - len(mine)
    if cfg.get("narrow_index") and m.narrow[i]:
        free = 0                                                # journal: new buys in narrow markets lagged SPY
    order = np.argsort(-np.nan_to_num(sc, nan=-np.inf))
    nvalid = int((~np.isnan(sc)).sum())
    q = (P["Cf"][i] - P["EMA20"][i]) / P["ATR"][i]
    for j in order[:nvalid]:
        if free <= 0:
            break
        j = int(j)
        if j in pos:
            continue
        if cfg["trend"] and not P["cont"][i, j]:
            continue
        if cfg["zone_gate"] and not (-1.0 <= q[j] <= 2.0):
            continue
        if cfg.get("index_hurdle") and not m.is_etf[j]:
            # a stock is only bought if its 6-month return beats the best of SPY/QQQ/SMH (else the index wins the slot)
            c_ = P["Cf"]
            if i < 126 or not (c_[i, j] / c_[i - 126, j] - 1 > max(m.E[t]["c"][i] / m.E[t]["c"][i - 126] - 1 for t in ("SPY", "QQQ", "SMH")
                                                                     if m.E[t]["c"][i - 126] == m.E[t]["c"][i - 126])):
                continue
        if cfg.get("vol_max_pct"):
            if "_rvp" not in cfg or cfg["_rvp"][0] != i:
                cfg["_rvp"] = (i, pct_rank(np.where(u, P["RV30"][i].astype(float), np.nan)))
            if not (cfg["_rvp"][1][j] <= cfg["vol_max_pct"]):
                continue
        if cfg.get("udvr_min") and not (P["UDVR"][i, j] >= cfg["udvr_min"]):
            continue                                            # more volume on up days than down days (accumulation)
        if cfg.get("cmf_min") is not None and not (P["CMF"][i, j] >= cfg["cmf_min"]):
            continue
        if cfg.get("obv_min") is not None and not (P["OBVT"][i, j] >= cfg["obv_min"]):
            continue
        if cfg.get("rvol_min") and not (P["RVOL"][i, j] >= cfg["rvol_min"]):
            continue
        if cfg.get("adx_max") and not (P["ADX"][i, j] <= cfg["adx_max"]):
            continue                                            # buy only early in a trend (ADX still low)
        if cfg.get("adx_min") and not (P["ADX"][i, j] >= cfg["adx_min"] and P["PDI_GT"][i, j]):
            continue
        if cfg.get("hi52_min") and not (P["hi52"][i, j] >= cfg["hi52_min"]):
            continue
        if cfg.get("macd_pos") and not (P["MACDH"][i, j] > 0):
            continue
        if cfg.get("macd_buy_min") is not None and not (P["MACDH"][i, j] >= cfg["macd_buy_min"]):
            continue                                            # buy only with MACD histogram >= x (share of price)
        if cfg.get("macd_buy_max") is not None and not (P["MACDH"][i, j] <= cfg["macd_buy_max"]):
            continue                                            # buy only with MACD histogram <= x (momentum pause / pullback)
        if cfg.get("rsi_max") and not (P["RSI"][i, j] <= cfg["rsi_max"]):
            continue                                            # rsi_max 40-50 = buy pullbacks only; 75-90 = skip overbought
        if cfg.get("rsi_min") and not (P["RSI"][i, j] >= cfg["rsi_min"]):
            continue                                            # buy only on strength
        if cfg.get("bo_recent") and not P["BO55"][max(0, i - cfg["bo_recent"]):i + 1, j].any():
            continue
        w = 1.0
        wt = cfg.get("weight")
        if wt == "rank":            # best-ranked new names get bigger slots
            rk = pos_rank.get(j, 10 ** 9)
            w = 1.5 if rk < 3 else (1.0 if rk < N else 0.75)
        elif wt == "invvol":        # calmer names get bigger slots
            rv = P["RV30"][i, j]; med = np.nanmedian(np.where(u, P["RV30"][i], np.nan))
            w = float(np.clip(med / rv, 0.5, 1.5)) if rv == rv and rv > 0 else 1.0
        elif wt == "mom":           # stronger 6-month trend gets a bigger slot
            z = P["m6"][i, j]; med = np.nanmedian(np.where(u, P["m6"][i], np.nan))
            w = float(np.clip(1.0 + 0.5 * (z - med), 0.5, 1.5)) if z == z else 1.0
        f0 = cfg.get("scale_in", [1.0])[0] if cfg.get("scale_in") else 1.0
        d0 = slot * w * f0
        out.append(dict(kind="new", j=j, dollars=d0, route=1 if P["cont"][i, j] else 2,
                        stop=(P["SMA50"][i, j] - P["ATR"][i, j]), min_dollars=0.5 * d0, meta={"tr": tr or 0, "w": w}))
        free -= 1
    si = cfg.get("scale_in")
    if si:
        # ladder up: start with f0 of the slot; each further rung (step, 2*step ... above the first entry
        # price) adds an equal share of the rest while the name is still ranked inside the top N
        f0, step, k = si
        for j, p in pos.items():
            if j in exiting or (tr is not None and p.get("meta", {}).get("tr", 0) != tr) or p["rung_in"] >= k:
                continue
            if pos_rank.get(j, 10 ** 9) < N and P["Cf"][i, j] / p["px0"] - 1 >= step * (p["rung_in"] + 1):
                amt = slot * p.get("meta", {}).get("w", 1.0) * (1 - f0) / k
                out.append(dict(kind="add", j=j, dollars=amt, route=p["route"], min_dollars=0.5 * amt))
                p["rung_in"] += 1
    if cfg["adds"] == "india":
        adds = []
        for j, p in pos.items():
            if j in exiting or (tr is not None and p.get("meta", {}).get("tr", 0) != tr):
                continue
            val = p["sh"] * P["Cf"][i, j]
            if (pos_rank.get(j, 10 ** 9) < N and P["cont"][i, j] and P["Cf"][i, j] / p["cost"] - 1 >= cfg["add_gain"]
                    and val < cfg["add_cap"] * slot):
                adds.append((pos_rank[j], j, min(cfg["add_size"] * slot, cfg["add_cap"] * slot - val)))
        for _, j, amt in sorted(adds):
            out.append(dict(kind="add", j=j, dollars=amt, route=pos[j]["route"], min_dollars=0.5 * amt))
