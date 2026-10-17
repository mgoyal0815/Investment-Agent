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
    # audit: a broker caps per-share commission at a share of the trade value; never more than the trade itself
    return min(max(COST_MIN, COST_PS * raw_sh + COST_BPS * value), max(COST_MIN, 0.01 * value + COST_BPS * value))


def pct_rank(x):
    """pandas rank(pct=True, method='average') on a 1-D array, NaN stays NaN."""
    out = np.full(x.shape, np.nan)
    m = ~np.isnan(x)
    n = m.sum()
    if n == 0:
        return out
    out[m] = pd.Series(x[m]).rank(pct=True).to_numpy()
    return out


def rank_desc(x, key):
    """Indices ordering x from highest to lowest; NaN last; ties broken by `key` ascending (Market.tie_key: an array,
    or a tuple of arrays, the first one deciding first). np.lexsort is a stable sort, so the order is fully determined
    by (x, key) on every numpy version."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=-np.inf)
    keys = tuple(key) if isinstance(key, tuple) else (key,)
    return np.lexsort(tuple(reversed(keys)) + (-x,))


class Market:
    """Panel plus the date-level series both systems read."""

    def __init__(self, P):
        self.P = P
        self.dates = P["dates"]
        self.T, self.N = P["C"].shape
        if "fip" not in P:
            # exp71 ranking features (owner, 16 Oct 2026), price-only, known at each close:
            # fip  = momentum quality ("frog in the pan", Da-Gurun-Warachka 2014): -sign(126d return) x (share of down
            #        days - share of up days) over 126 sessions; high = rose through many small steady gains
            # accel = momentum acceleration: last 63-session return minus the 63 sessions before that
            cf_ = pd.DataFrame(np.asarray(P["Cf"], np.float64))
            r1 = cf_.pct_change()
            up_ = (r1 > 0).astype(float).rolling(126, min_periods=100).mean()
            dn_ = (r1 < 0).astype(float).rolling(126, min_periods=100).mean()
            r126 = cf_ / cf_.shift(126) - 1
            P["fip"] = (-np.sign(r126) * (dn_ - up_)).to_numpy(np.float32)
            P["accel"] = ((cf_ / cf_.shift(63) - 1) - (cf_.shift(63) / cf_.shift(126) - 1)).to_numpy(np.float32)
        for sd in (1, 2, 3):
            # audit placebo (exp74): fixed random noise per stock and day, to blend into the ranking at a tiny weight and
            # measure how much the result depends on the exact ranking (a fair system should barely move)
            if f"noise{sd}" not in P:
                P[f"noise{sd}"] = np.random.default_rng(sd).random(P["Cf"].shape, dtype=np.float32)
        self.syms = P["syms"]
        # Tie-break (owner, 17 Oct 2026; REBUILD.md cause 3). Percentile scores tie often and np.argsort's order for
        # ties is not fixed (it changed between numpy 1.26 and 2.5). Every ranking now sorts with rank_desc(): score
        # first, then an explicit key. Default key = alphabetical rank of the symbol, so the same names break ties the
        # same way in any panel (backtest or live) on any numpy. cfg["tie_seed"] = s replaces it with a fixed random
        # order of the symbols (seeded), to measure how much a result owes to tie luck (robust.py).
        self.tie_alpha = np.argsort(np.argsort(np.asarray(self.syms, dtype=str), kind="stable"), kind="stable")
        self._tie = {}
        E = {}
        for k, (o, c) in P["etf"].items():
            c = pd.Series(c, index=self.dates)
            E[k] = dict(raw=c.to_numpy(), c=c.ffill().to_numpy(), o=pd.Series(o, index=self.dates).to_numpy())
        # exp78 (owner, 16 Oct 2026: leveraged index ETFs allowed if the worst fall does not get worse): synthetic
        # daily-reset 2x funds "SPY2x", "QQQ2x", "SMH2x": r = 2 r_index - r_cash - 0.95%/252 (lev.py; validated against
        # SSO, QLD and USD where they trade). The open applies 2x the overnight gap to the previous close.
        bil_ = pd.Series(E["BIL"]["c"], index=self.dates).pct_change(fill_method=None) if "BIL" in E else None
        shy_ = pd.Series(E["SHY"]["c"], index=self.dates).pct_change(fill_method=None)
        rc_ = (bil_.where(bil_.notna(), shy_.rolling(63, min_periods=5).mean()) if bil_ is not None else shy_.rolling(63, min_periods=5).mean()).fillna(0).clip(-0.002, 0.002)
        for t_ in ("SPY", "QQQ", "SMH"):
            c_ = pd.Series(E[t_]["c"], index=self.dates)
            r_ = c_.pct_change(fill_method=None)
            fee_ = 0.0095 + (0.035 if t_ == "SMH" else 0.0)    # SMH: +3.5%/yr so the synthetic matches the real USD fund
            rl_ = (2 * r_ - rc_ - fee_ / 252).where(r_.notna())
            cl_ = (1 + rl_.fillna(0)).cumprod().where(c_.notna()) * 100
            o_ = pd.Series(E[t_]["o"], index=self.dates)
            ol_ = cl_.shift(1) * (1 + 2 * (o_ / c_.shift(1) - 1))
            E[t_ + "2x"] = dict(raw=cl_.where(pd.Series(E[t_]["raw"], index=self.dates).notna()).to_numpy(), c=cl_.to_numpy(), o=ol_.to_numpy())
        # "SPY2xr", "QQQ2xr", "SMH2xr": the real funds (SSO, QLD, USD; FMP dividend-adjusted) from their first trading
        # day, with the synthetic series before that, rescaled to join at the first real close
        import json as _json, os as _os
        _ld = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lev")
        for t_, f_ in (("SPY", "SSO"), ("QQQ", "QLD"), ("SMH", "USD")):
            fp_ = _os.path.join(_ld, f_ + ".json")
            if not _os.path.exists(fp_):
                continue
            j_ = pd.DataFrame(_json.load(open(fp_)))
            j_["date"] = pd.to_datetime(j_["date"]); j_ = j_.drop_duplicates("date").set_index("date").sort_index().reindex(self.dates)
            rc_r, ro_r = j_["adjClose"], j_["adjOpen"]
            first_ = rc_r.first_valid_index()
            sc_ = pd.Series(E[t_ + "2x"]["c"], index=self.dates); so_ = pd.Series(E[t_ + "2x"]["o"], index=self.dates)
            if first_ is None:                                 # data ends before the fund launched: synthetic only
                E[t_ + "2xr"] = dict(E[t_ + "2x"]); continue
            # keep the synthetic price level and rescale the real fund to it (fixed 16 Oct 2026: rescaling the synthetic
            # part to the real fund's launch price used a future price level, which changed whole-share counts)
            k_ = sc_[first_] / rc_r[first_]
            cc_ = sc_.where(self.dates < first_, rc_r * k_); oo_ = so_.where(self.dates < first_, ro_r * k_)
            E[t_ + "2xr"] = dict(raw=cc_.to_numpy(), c=cc_.ffill().to_numpy(), o=oo_.to_numpy())
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
            import os as _os
            sec = np.load(_os.environ.get("SECTOR_FILE", "../data/sector_idx.npy"))
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
        for fn in ("../data/vol_panel.npz", "../data/vol_panel_idx.npz", "../data/vol_panel_audit.npz", "../data/vol_panel_ndx.npz"):
            try:
                z = np.load(fn)
                if z["UDVR"].shape == (self.T, self.N):
                    for k in z.files:
                        P[k] = z[k]
                    break
            except Exception:
                pass
        # sector risk-appetite ratios (exp56): discretionary/staples, cyclicals/defensives (sector ETFs in the panel)
        _sy = list(P["syms"])
        def _px(t):
            return pd.Series(Cf_[:, _sy.index(t)]).ffill().to_numpy() if t in _sy else None
        self.ra_etf = {}
        if all(_px(t) is not None for t in ("XLY", "XLP", "XLK", "XLI", "XLF", "XLU", "XLV")):
            self.ra_etf["dp"] = _px("XLY") / _px("XLP")
            self.ra_etf["cd"] = (_px("XLY") + _px("XLK") + _px("XLI") + _px("XLF")) / (_px("XLP") + _px("XLU") + _px("XLV"))
        # Minervini trend template parts (exp55), from closes up to the session itself
        _cf = pd.DataFrame(Cf_)
        P["SMA150"] = _cf.rolling(150, min_periods=150).mean().to_numpy(np.float32)
        P["LO52"] = _cf.rolling(252, min_periods=200).min().to_numpy(np.float32)
        P["HI52C"] = _cf.rolling(252, min_periods=200).max().to_numpy(np.float32)
        del _cf
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
        try:                                            # exp68: 10y-2y Treasury spread (FRED T10Y2Y), known the next day
            yc = pd.read_csv("../data/t10y2y.csv", parse_dates=["observation_date"]).set_index("observation_date")["T10Y2Y"]
            yc = pd.to_numeric(yc, errors="coerce").reindex(self.dates, method="ffill").shift(1)
            self.curve = yc.to_numpy()
        except Exception:
            self.curve = np.full(self.T, np.nan)
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
            # ---- re-entry signals (exp53), from that day's S&P 500 members only
            r1 = np.where(spm, P["ret1"], np.nan)
            adv, dec = (r1 > 0).sum(1).astype(float), (r1 < 0).sum(1).astype(float)
            rana = pd.Series(1000 * (adv - dec) / np.maximum(adv + dec, 1))
            self.mcc_osc = (rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()).to_numpy()
            self.mcc_sum = np.cumsum(self.mcc_osc)
            hi = pd.DataFrame(Cf).rolling(252, min_periods=200).max().to_numpy()
            lo = pd.DataFrame(Cf).rolling(252, min_periods=200).min().to_numpy()
            nh = ((Cf >= hi * 0.999) & spm).sum(1); nl = ((Cf <= lo * 1.001) & spm).sum(1)
            self.nhnl = pd.Series((nh - nl) / np.maximum(spm.sum(1), 1)).rolling(10, min_periods=5).mean().to_numpy()
            # ---- risk-appetite ratios (exp56): rising = investors paying up for risk. hblv: equal-weight 100 most volatile
            # vs 100 least volatile of that day's S&P 500 members (30-day volatility known the day before)
            rv = pd.DataFrame(np.where(spm & ~self.is_etf & np.isfinite(P["RV30"]), P["RV30"], np.nan))
            rk_hi, rk_lo = rv.rank(axis=1, ascending=False).shift(1).to_numpy(), rv.rank(axis=1).shift(1).to_numpy()
            r0 = np.nan_to_num(P["ret1"])
            def _ew(mask):
                return np.cumprod(1 + np.where(mask.sum(1) > 0, (np.where(mask, r0, 0)).sum(1) / np.maximum(mask.sum(1), 1), 0.0))
            self.ra = {"hblv": _ew(rk_hi <= 100) / _ew(rk_lo <= 100)}
            spy_ = pd.Series(E["SPY"]["c"]).ffill()
            roc = (spy_ / spy_.shift(294) - 1) + (spy_ / spy_.shift(231) - 1)
            w = np.arange(1, 211, dtype=float)
            self.coppock = roc.rolling(210, min_periods=210).apply(lambda x: np.dot(x, w) / w.sum(), raw=True).to_numpy()
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

    def lit(self):
        """exp75 (owner, 16 Oct 2026): features from the literature review, built once on first use. Everything at
        row i uses closes up to i only. Universe for the market-level ones: that day's S&P 500 + Nasdaq-100 members."""
        if hasattr(self, "_lit"):
            return self._lit
        import os
        P = self.P; T, N = self.T, self.N
        dd_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"lit_{T}x{N}")
        keys_ = ("stress", "gap", "disp_hi", "gap90", "volr", "on63", "in63", "spy1m", "spy12m", "spy_sma50", "spy_c")
        if os.path.exists(os.path.join(dd_, "in63.npy")):          # computed once, read-only memory map in every worker
            self._lit = {k: np.load(os.path.join(dd_, k + ".npy"), mmap_mode="r") for k in keys_}
            return self._lit
        cf = pd.DataFrame(np.asarray(P["Cf"], np.float64))
        r1 = cf.pct_change().to_numpy()
        mem = (self.sp_member if self.sp_member is not None else np.ones((T, N), bool))
        if getattr(self, "ndx_member", None) is not None:
            mem = mem | self.ndx_member
        mem = mem & ~self.is_etf
        spy = pd.Series(self.E["SPY"]["c"]).ffill(); spy_r = spy.pct_change().to_numpy()
        m6 = np.asarray(P["m6"], np.float64)
        # 1. momentum-stress (Heidari 2015; Barroso & Santa-Clara 2015): top-decile 6m winners (chosen at i-1) minus SPY
        prox = np.full(T, np.nan)
        for i in range(1, T):
            x = np.where(mem[i - 1], m6[i - 1], np.nan)
            ok = ~np.isnan(x)
            if ok.sum() < 50:
                continue
            top = ok & (x >= np.nanpercentile(x, 90))
            rr = r1[i, top]
            prox[i] = np.nanmean(rr) - spy_r[i]
        ps = pd.Series(prox)
        v21 = ps.rolling(21, min_periods=15).std(); dv = v21 - v21.shift(21)
        pct = dv.rolling(1260, min_periods=504).rank(pct=True)
        stress = ((pct > 0.9) & (ps.rolling(21).sum() < 0)).to_numpy()
        # 2. mega-cap concentration gap: SPY 63d return minus the equal-weight member average 63d return
        r63 = (cf / cf.shift(63) - 1).to_numpy()
        ew63 = np.array([np.nanmean(np.where(mem[i], r63[i], np.nan)) if mem[i].any() else np.nan for i in range(T)])
        gap = (spy / spy.shift(63) - 1).to_numpy() - ew63
        # 6. dispersion: cross-sectional std of 21d returns, 63d mean, above its trailing-5y 80th percentile
        r21 = (cf / cf.shift(21) - 1).to_numpy()
        dsp = pd.Series([np.nanstd(np.where(mem[i], r21[i], np.nan)) if mem[i].any() else np.nan for i in range(T)]).rolling(63, min_periods=40).mean()
        disp_hi = (dsp > dsp.rolling(1260, min_periods=504).quantile(0.8)).to_numpy()
        # 14. largest one-day move in 90 sessions; 12. 50-day vs 250-day average dollar volume (no shares outstanding)
        gap90 = pd.DataFrame(np.abs(r1)).rolling(90, min_periods=60).max().to_numpy(np.float32)
        adv = pd.DataFrame(np.asarray(P["ADV20"], np.float64))
        volr = (adv.rolling(50, min_periods=40).mean() / adv.rolling(250, min_periods=200).mean()).to_numpy(np.float32)
        # 8. overnight vs intraday split of the 63-day gain (opens adjusted to the adjusted closes)
        oa = np.asarray(P["O"], np.float64) * (np.asarray(P["Cf"], np.float64) / np.asarray(P["C"], np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            on = np.log(oa / cf.shift(1).to_numpy()); intra = np.log(cf.to_numpy() / oa)
        on63 = pd.DataFrame(on).rolling(63, min_periods=50).sum().to_numpy(np.float32)
        in63 = pd.DataFrame(intra).rolling(63, min_periods=50).sum().to_numpy(np.float32)
        s1 = spy / spy.shift(21) - 1; s12 = spy / spy.shift(252) - 1
        self._lit = dict(stress=stress, gap=gap, disp_hi=disp_hi, gap90=gap90, volr=volr, on63=on63, in63=in63,
                         spy1m=s1.to_numpy(), spy12m=s12.to_numpy(), spy_sma50=spy.rolling(50).mean().to_numpy(), spy_c=spy.to_numpy())
        os.makedirs(dd_, exist_ok=True)
        for k in keys_:
            np.save(os.path.join(dd_, k + ".npy"), np.asarray(self._lit[k]))
        return self._lit

    def volf(self):
        """Volume features from volfeat.py (cached .npy files, read-only memory maps)."""
        if not hasattr(self, "_volf"):
            import os
            dd_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"volf_{self.T}x{self.N}")
            self._volf = {k: np.load(os.path.join(dd_, k + ".npy"), mmap_mode="r") for k in ("pp10", "vdry", "climax", "dd25", "ftd")}
        return self._volf

    def etf_rsi(self, t):
        """14-day Wilder RSI of an index ETF from its closes (exp62), cached."""
        if not hasattr(self, "_ersi"):
            self._ersi = {}
        if t not in self._ersi:
            s_ = pd.Series(self.E[t]["c"]); d_ = s_.diff()
            up = d_.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-d_.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
            self._ersi[t] = (100 - 100 / (1 + up / dn)).to_numpy()
        return self._ersi[t]

    # cfg["tie_key"]: break ties on information first (alphabetical after it). Higher value wins unless noted.
    TIE_KEYS = {"rs126": lambda P, i: P["rs126"][i], "dist200": lambda P, i: P["dist200"][i],
                "medv": lambda P, i: P["medv"][i],                                   # more traded first
                "lowvol": lambda P, i: -(P["ATR"][i] / P["Cf"][i]),                   # calmer first
                "hi52": lambda P, i: P["hi52"][i] if "hi52" in P else None,           # nearer the 1-year high first
                "m12": lambda P, i: P["m12"][i] if "m12" in P else None,
                "zsum": None}                                                         # raw z-score sum, see tie_key

    def tie_key(self, cfg, i=None):
        cfg = cfg or {}
        tk = cfg.get("tie_key")
        if tk and i is not None:
            P = self.P
            if tk == "rev_alpha":                                 # control: reverse alphabetical (no information)
                return -self.tie_alpha
            if tk == "zsum":
                # the two inputs of the robust score as raw z-scores (continuous, so they almost never tie)
                z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
                v = z(np.asarray(P["rs126"][i], float)) + z(np.asarray(P["dist200"][i], float))
            else:
                v = np.asarray(self.TIE_KEYS[tk](P, i), float)
            return (-np.nan_to_num(v, nan=-np.inf), self.tie_alpha)
        sd = cfg.get("tie_seed")
        if sd is None:
            return self.tie_alpha
        if sd not in self._tie:
            self._tie[sd] = np.random.default_rng(int(sd)).permutation(self.N)
        return self._tie[sd]

    def giants(self, i, k, rise=63):
        """exp58: the k most-traded (median dollar volume) S&P 500 members that day, as a boolean mask, and
        whether each is above a rising 200-session average (rising over `rise` sessions). Dollar volume only, no caps."""
        key = (i, k, rise)
        if getattr(self, "_gk", (None,))[0] != key:
            P = self.P
            mem = (self.sp_member[i] if self.sp_member is not None else np.ones(self.N, bool)) & ~self.is_etf
            mv = np.where(mem & np.isfinite(P["medv"][i]), P["medv"][i], -1)
            top = np.zeros(self.N, bool); top[rank_desc(mv, self.tie_alpha)[:k]] = True; top &= mv > 0
            s2 = P["SMA200"]
            up = (P["Cf"][i] > s2[i]) & (s2[i] > s2[max(0, i - rise)])
            self._gk = (key, top, np.nan_to_num(up, nan=0).astype(bool))
        return self._gk[1], self._gk[2]

    def ra_series(self, tok):
        name = tok.rstrip("0123456789"); n = int(tok[len(name):] or 50)
        rr = {**getattr(self, "ra_etf", {}), **getattr(self, "ra", {})}.get(name)
        if rr is not None and not hasattr(self, "_ra_sma"):
            self._ra_sma = {}
        if rr is not None and (name, n) not in self._ra_sma:
            self._ra_sma[(name, n)] = pd.Series(rr).rolling(n, min_periods=n).mean().to_numpy()
        return (None if rr is None else (rr, self._ra_sma[(name, n)])), n

    def ra_below(self, rs, n, i):
        rr, sma = rs
        return sma[i] == sma[i] and rr[i] < sma[i]

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
            # etf_in_universe also applies here (fixed 16 Oct 2026: it was silently ignored for this universe)
            return base & (self.sp_member[i] | self.ndx_member[i] | (self.is_etf if etf_in else False))
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
            w_ = cfg.get("robust_w", 0.5)                  # exp68: weight on 6-month strength vs distance above the 200-day
            return pct_rank(w_ * a + (1 - w_) * b), route, u
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
    if cfg.get("lev_after"):
        cfg["_i0"] = i0                                       # exp78: run start, for the no-2x-in-a-new-account rule
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
        if cfg.get("large_n", 50) != 50:          # exp69: longer average for the large-cap pause vote (whipsaw test)
            LSMA_PIT = pd.Series(m.large_idx_pit).rolling(cfg["large_n"], min_periods=cfg["large_n"]).mean().to_numpy()
    else:
        BREADTH, LARGE, LSMA_PIT = m.breadth, m.large_idx, None
    prev_paused = [False]
    hyst = [False, -10 ** 6]   # exp60 pause hysteresis: current vote state, session it last flipped
    unpause_until = [-1]   # index core
    up_anchor, up_block = [np.nan], [-1]   # re-entry guard (exp54): SPY level when the window opened; refire blocked until: ETF held while it is in a strong uptrend
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
        if raw_px < cfg.get("min_buy_price", 1.0):
            return 0.0                                  # audit: never buy below $1 a share (bad data or a penny stock)
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

    def whole_review(i):
        """India round 11 (17 Oct 2026): extra review days (day_in_month) on which the WHOLE book is reviewed - rank
        sells across both halves, then each half's free slots filled. A tranche's own review day takes precedence."""
        ex = cfg.get("extra_reviews")
        return bool(ex) and m.day_in_month[i] in ex and tranche_of(i) is None

    def is_decision(i):
        if cfg["cadence"] == "daily":
            return True
        if cfg.get("stagger"):
            return tranche_of(i) is not None or whole_review(i)
        if cfg["cadence"] == "monthly":
            dd_ = cfg.get("decide_day")
            return m.month_end[i] if dd_ is None or dd_ < 0 else m.day_in_month[i] == dd_
        if cfg["cadence"] == "weekly":
            return m.week_end[i]
        return m.month_end[i] or m.day_in_month[i] == 10

    # ---------------------------------------------------- live takeover (owner, 16 Oct 2026)
    # cfg["init"] = {"cash": $, "stocks": {sym: {"shares": raw shares, "cost": raw $ per share, "entry_date": "YYYY-MM-DD"}},
    #                "etfs": {sym: shares}} starts the run from an existing account instead of cash. The first session
    # (i0) is held as it stands (no trades at its open); at its close the system makes a full review of the whole book:
    # carried stocks are kept only if ranked inside the sell buffer (no exemptions for them), the rest are sold at the
    # next open, ETFs the system does not park in are rotated out, and free slots are filled in both tranches.
    init = cfg.get("init")
    carried = set()
    i_init = i0
    if init:
        # the run starts at `start` with no money (a warm-up, so pause / restart state is built from real history);
        # the account is injected at the open of init["first_session"]
        i_init = int(m.dates.searchsorted(pd.Timestamp(init["first_session"])))
        cash = 0.0

    def inject():
        nonlocal cash
        cash = float(init.get("cash", 0.0))
        i0 = i_init
        J_ = {s: j for j, s in enumerate(m.syms)}
        for k_, (sym_, h_) in enumerate(sorted((init.get("stocks") or {}).items())):
            j = J_.get(sym_)
            if j is None or not (Cf[i0 - 1, j] == Cf[i0 - 1, j]):
                raise ValueError(f"takeover: no price for carried stock {sym_} at {m.dates[i0 - 1].date()}")
            sf_ = float(SF[i0 - 1, j]) or 1.0
            adj_sh = float(h_["shares"]) * sf_
            cost_ = float(h_.get("cost") or Cf[i0 - 1, j] * sf_) / sf_
            ei = int(m.dates.searchsorted(pd.Timestamp(h_["entry_date"]))) if h_.get("entry_date") else i0
            ei = min(max(ei, 0), i0)
            pk_ = float(np.nanmax(Cf[ei:i0, j])) if ei < i0 else float(Cf[i0 - 1, j])
            meta_ = {"tr": int(h_.get("tr", k_ % 2)), "carried": bool(h_.get("carried", init.get("mode", "takeover") == "takeover"))}
            if h_.get("giant"):
                meta_["giant"] = True
            pk_ = max(pk_, float(h_["peak"]) / sf_) if h_.get("peak") else pk_
            pos[j] = dict(sh=adj_sh, cost=cost_, entry_i=ei, route=1, stop=None, rung=0, below=0, peak=max(pk_, cost_),
                          last=float(Cf[i0 - 1, j]), stale=0, adds=int(h_.get("adds", 0)), hold_done=False, meta=meta_,
                          px0=cost_, rung_in=0, inv=float(h_["shares"]) * float(h_.get("cost") or 0.0), buyc=0.0)
            if init.get("mode", "takeover") == "takeover":
                carried.add(j)
        # "continue" mode: the orders decided at the previous close are placed at this session's open
        for o_ in init.get("pending") or []:
            j = J_.get(o_["sym"])
            if j is None:
                continue
            od = dict(kind=o_["kind"], j=j, why=o_.get("why"), all=o_.get("all"), dollars=o_.get("dollars") or 0.0)
            if o_.get("frac") is not None:
                od["frac"] = o_["frac"]
            if o_["kind"] in ("new", "add"):
                od.update(route=1, stop=None, min_dollars=o_.get("min_dollars", 0.0),
                          meta={"tr": int(o_.get("tr", 0)), "giant": bool(o_.get("giant"))} if o_.get("giant") else {"tr": int(o_.get("tr", 0))})
            pend_stock.append(od)
        for t_, sh_ in (init.get("etfs") or {}).items():
            if t_ in m.E and sh_:
                parked[t_] = float(sh_)

    # -------------------------------------------------------------- the loop
    for i in range(i0, i1):
        d = m.dates[i]
        if init and i == i_init:
            inject()
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
        held_open = bool(init) and (i < i_init or (i == i_init and init.get("mode", "takeover") == "takeover"))
        if park_tk and held_open:
            _park_name(park_tk, m, i, cfg, paused)               # warm-up / held open: tick the crash-hold timer, no trades
        if park_tk and not drew and not held_open:
            # money held back by the volatility cap must not ride the market in an index ETF
            risk_off = paused if cfg.get("safe_when", "paused") == "paused" else (not m.b200[i])
            if cfg.get("park_on_decision"):
                # only re-decide the parking ETF on a decision day; hold the last choice in between
                if is_decision(i - 1) or park_state[0] is None:
                    park_state[0] = risk_off
                risk_off = park_state[0]
            if cfg.get("lev_dd") and curve:
                cfg["_acct_dd"] = curve[-1] / max(curve) - 1          # exp78: account fall from its peak, known at yesterday's close
            pk = _park_name(park_tk, m, i, cfg, risk_off or (bool(cfg.get("vol_target") or cfg.get("mkt_vol_target")) and vf[0] < 0.999))
            pf, split_done = cfg.get("park_frac"), False
            if pf and park_tk == "leader_safe" and pk == "SHY":
                # exp58: during a pause keep part (park_frac) of the parked money in the leading index while it is
                # above its park_frac_sma-session average; the rest stays in SHY. Rebalanced only on a 10% drift.
                ld = _park_name("leader", m, i, cfg, False); ps = cfg.get("park_frac_sma", 200)
                tg = {ld: pf, "SHY": 1 - pf} if (i > ps and m.etf_vs_sma(ld, ps, i - 1) > 0) else {"SHY": 1.0}
                for k in [k for k in list(parked) if k not in tg]:
                    park_trade(k, -parked[k], i)
                keep = 0.01 * navo
                tot = cash - keep + sum(parked.get(k, 0) * etf_px(k, i, "o") for k in tg if etf_px(k, i, "o") == etf_px(k, i, "o"))
                if tot > 0.5 * navo / N:
                    for side in (-1, 1):                         # sells first, then buys
                        for k, w_ in tg.items():
                            px = etf_px(k, i, "o")
                            if px != px or px <= 0:
                                continue
                            diff = w_ * tot - parked.get(k, 0) * px
                            if abs(diff) > 0.10 * tot and np.sign(diff) == side:
                                n = diff / px
                                n = (math.floor(n) if n > 0 else -math.floor(-n)) if cfg["whole_etf"] else n
                                if n < 0:
                                    n = max(n, -parked.get(k, 0))
                                park_trade(k, n, i)
                pk, split_done = None, True
            lc = cfg.get("lev_cap")
            if lc and pk and "2x" in pk and not split_done:
                # exp78: at most lev_cap of the account in the 2x fund; the rest of the parked money in the 1x index.
                # Rebalanced only on a 10% drift, like park_frac.
                base_ = pk.split("2x")[0]; tg = {pk: 0.0, base_: 1.0}
                for k in [k for k in list(parked) if k not in tg]:
                    park_trade(k, -parked[k], i)
                keep = 0.01 * navo
                tot = cash - keep + sum(parked.get(k, 0) * etf_px(k, i, "o") for k in tg if etf_px(k, i, "o") == etf_px(k, i, "o"))
                if tot > 0:
                    w2 = min(1.0, lc * navo / tot); tg = {pk: w2, base_: 1 - w2}
                if tot > 0.5 * navo / N:
                    for side in (-1, 1):
                        for k, w_ in tg.items():
                            px = etf_px(k, i, "o")
                            if px != px or px <= 0:
                                continue
                            diff = w_ * tot - parked.get(k, 0) * px
                            if abs(diff) > 0.10 * tot and np.sign(diff) == side:
                                n = diff / px
                                n = (math.floor(n) if n > 0 else -math.floor(-n)) if cfg["whole_etf"] else n
                                if n < 0:
                                    n = max(n, -parked.get(k, 0))
                                park_trade(k, n, i)
                pk, split_done = None, True
            # rotate: a parking ETF that is no longer the target is sold first
            for k in [k for k in list(parked) if k != pk and not split_done]:
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
                   ("large" in r and ls_[i] == ls_[i] and LARGE[i] < ls_[i] * (1 - cfg.get("large_margin", 0.0)))]   # exp69: margin below the average
            # index / credit / bond votes: "<ETF><n>" = that ETF closes below its n-session average
            # (e.g. spy200, qqq100, smh50, hyg63); "tlt_up" = TLT above its 63-session average (flight to safety)
            for tok in r:
                if tok in ("breadth", "hyg", "large", "momstress", "distday"):
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
                elif tok == "maytooct":             # exp68: 'sell in May' - May to October counts as a pause vote
                    sig.append(m.dates[i].month in (5, 6, 7, 8, 9, 10))
                elif tok == "midterm":              # exp68: January-September of US midterm years
                    sig.append(m.dates[i].year % 4 == 2 and m.dates[i].month <= 9)
                elif tok == "uupup63":              # exp68: the dollar above its 63-day average (risk-off)
                    sig.append(m.etf_vs_sma("UUP", 63, i) > 0)
                elif tok == "curve_inv":            # exp68: yield curve inverted (10y below 2y)
                    sig.append(bool(m.curve[i] == m.curve[i] and m.curve[i] < 0))
                elif tok == "curve_uninv":          # exp68: curve un-inverting: was inverted in the last year, now positive
                    w_ = m.curve[max(0, i - 252):i + 1]
                    sig.append(bool(m.curve[i] == m.curve[i] and m.curve[i] > 0 and np.nanmin(w_) < 0))
                elif tok.startswith("ra_"):          # e.g. ra_dp50: risk-appetite ratio below its 50-session average
                    rr, nn = m.ra_series(tok[3:])
                    sig.append(rr is not None and m.ra_below(rr, nn, i))
                elif tok.startswith("b50_"):         # e.g. b50_20: under 20% of stocks above their 50-day
                    sig.append(m.b50[i] < float(tok[4:]) / 100)
                else:
                    t_, n_ = tok[:3].upper(), int(tok[3:])
                    sig.append(m.etf_vs_sma(t_, n_, i) < 0)
            # "any": one signal pauses (India v8); "two": two must agree, so buying resumes sooner
            if "distday" in r:                      # exp76: O'Neil distribution days on SPY/QQQ
                sig.append(bool(m.volf()["dd25"][i] >= cfg.get("dd_n", 6)))
            if "momstress" in r:                    # exp75 idea 1: momentum's own volatility shock (Heidari 2015)
                sig.append(bool(m.lit()["stress"][i]))
            if cfg.get("state_pause"):
                # exp75 idea 4 (Goulding-Harvey-Mazzoleni 2023): in a Correction (SPY 12m up, 1m down) ignore the fast
                # large-cap vote; in a Rebound (12m down, 1m up) a close back above SPY's 50-day lifts the pause
                L_ = m.lit(); s12_, s1_ = L_["spy12m"][i], L_["spy1m"][i]
                if s12_ > 0 and s1_ < 0 and "large" in r:
                    sig[2] = False
            paused = sum(sig) >= (cfg["pause_k"] if cfg.get("pause_k") else (2 if cfg.get("pause_mode", "any") == "two" else 1))
            if cfg.get("state_pause") and paused:
                L_ = m.lit()
                if L_["spy12m"][i] < 0 and L_["spy1m"][i] > 0 and L_["spy_c"][i] > L_["spy_sma50"][i]:
                    paused = False
            pmd = cfg.get("pause_min_days")
            if pmd:                                     # exp60: once the vote flips, hold the new state at least pmd sessions
                if paused != hyst[0] and i - hyst[1] < pmd:
                    paused = hyst[0]
                elif paused != hyst[0]:
                    hyst[0], hyst[1] = paused, i
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
            # re-entry signals (exp53): end the pause early for unpause_days sessions when a bottom signal fires
            ue = cfg.get("unpause_on")
            if ue and i > 30:
                fire = False
                if "coppock" in ue and m.coppock[i] == m.coppock[i]:
                    fire |= m.coppock[i] < 0 and m.coppock[i] > m.coppock[i - 21] and m.coppock[i - 21] <= m.coppock[i - 42]
                if "ftd" in ue:                       # exp76: O'Neil follow-through day restarts buying
                    fire |= bool(m.volf()["ftd"][i])
                if "mcclellan" in ue:
                    fire |= m.mcc_osc[i] > 0 and np.nanmin(m.mcc_osc[i - 30:i]) < -cfg.get("mcc_depth", 50)
                for u_ in ue:
                    if u_ in ("spymacdup", "qqqmacdup"):     # exp65: the index's MACD crosses back above its signal line
                        t_ = u_[:3].upper(); fire |= bool(m.etf_macd(t_, i) > 0 and m.etf_macd(t_, i - 1) <= 0)
                    if u_[:3] in ("spy", "qqq") and u_[3:6] == "rsi":  # exp62: a panic (RSI <= NN) ends the pause
                        fire |= bool(m.etf_rsi(u_[:3].upper())[i] <= float(u_[6:]))
                if "random" in ue and i in cfg.get("_rand_days", ()):
                    fire = True                      # audit placebo: restart on random dates at the real signal's frequency
                for u_ in ue:
                    if u_.startswith("ra_"):         # e.g. ra_hblv50: ratio crosses above its 50-session average
                        rr, nn = m.ra_series(u_[3:])
                        f_ = rr is not None and not m.ra_below(rr, nn, i) and m.ra_below(rr, nn, i - 1)
                        # exp59: a second signal must agree on the day (all listed): "spy50" SPY above its 50-day,
                        # "b50_40" over 40% of members above their 50-day, "mcc" McClellan oscillator > 0,
                        # "hblv50" high-beta/low-vol above its 50-day, "vixrel1" VIX below its 50-day average
                        for cf_ in cfg.get("unpause_confirm", []):
                            if cf_.startswith("spy"):
                                f_ &= m.etf_vs_sma("SPY", int(cf_[3:]), i) > 0
                            elif cf_.startswith("b50_"):
                                b_ = getattr(m, "b50_pit", m.b50); f_ &= bool(b_[i] > float(cf_[4:]) / 100)
                            elif cf_ == "mcc":
                                f_ &= bool(m.mcc_osc[i] > 0)
                            elif cf_.startswith("vixrel"):
                                f_ &= bool(m.vix_rel[i] == m.vix_rel[i] and m.vix_rel[i] < float(cf_[6:]))
                            else:
                                r2, n2 = m.ra_series(cf_)
                                f_ &= r2 is not None and not m.ra_below(r2, n2, i)
                        fire |= f_
                if "nhnl" in ue:
                    fire |= m.nhnl[i] > 0 and np.nanmin(m.nhnl[i - 40:i]) < -cfg.get("nhnl_depth", 0.05)
                spy_now = m.E["SPY"]["c"][i]
                if fire and i > up_block[0]:
                    if i > unpause_until[0]:
                        up_anchor[0] = spy_now
                    unpause_until[0] = i + cfg.get("unpause_days", 42)
                us = cfg.get("unpause_stop")
                if us is not None and i <= unpause_until[0] and spy_now < up_anchor[0] * (1 - us):
                    unpause_until[0], up_block[0] = -1, i + cfg.get("unpause_cool", 63)
                if i <= unpause_until[0]:
                    paused = False
            rv_ = cfg.get("ra_veto")                 # e.g. "dp50": no pause while the ratio is above its average and rising
            if rv_ and paused and i > 260:
                rr, nn = m.ra_series(rv_)
                if rr is not None and not m.ra_below(rr, nn, i) and rr[0][i] > rr[0][i - cfg.get("ra_veto_lb", 21)]:
                    paused = False
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
        if cfg.get("climax_exit"):
            # exp76: O'Neil climax run with a stall day: sell half once ("half") or all ("all")
            cx_ = m.volf()["climax"]
            for j, p in pos.items():
                if p["stale"] or j in exiting or not cx_[i, j]:
                    continue
                if cfg["climax_exit"] == "half":
                    if p.get("climaxed"):
                        continue
                    pend_stock.append(dict(kind="exit", j=j, why="climax1", all=False, frac=0.5)); p["climaxed"] = True
                else:
                    pend_stock.append(dict(kind="exit", j=j, why="climax", all=True)); exiting.add(j)
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
        fill_only = False
        takeover = bool(init) and i == i_init and init.get("mode", "takeover") == "takeover"
        if init and i < i_init:
            continue                                            # warm-up: signals only, no account yet
        if not is_decision(i) and not fast_re and not takeover:
            # exp77: on the last session of each week, fill empty slots without any rank sells (calendar known in advance)
            if cfg.get("fill_weekly") and m.week_end[i] and cfg.get("stagger"):
                fill_only = True
            else:
                continue
        navd = navc
        sc, route, u = m.scores(i, cfg)
        ks = cfg.get("score_smooth")
        if ks and ks > 1 and i >= ks:
            # exp70 (root cause of review-day luck): rank on the score averaged over the last ks sessions, so a stock must
            # be strong for days, not just on review day. Only stocks eligible today count; past scores use past data only.
            past = np.vstack([sc] + [m.scores(i - k, cfg)[0] for k in range(1, ks)])
            sc = np.where(np.isnan(sc), np.nan, np.nanmean(np.where(np.isnan(past), sc, past), axis=0))
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
                top = rank_desc(smom, np.arange(len(smom)))[:cfg["sector_top_k"]]
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
        gb = cfg.get("giant_bonus")
        if gb:
            # exp64: the gb[0] most-traded S&P members get gb[1] added to their rank percentile (missed-giants fix)
            top_, _ = m.giants(i, gb[0])
            sc = np.where(np.isnan(sc), np.nan, np.minimum(1.0, sc + np.where(top_, gb[1], 0.0)))
        if cfg.get("size_w"):
            w_ = cfg["size_w"]                          # exp64: always lean the ranking toward the most-traded names
            sc = np.where(np.isnan(sc), np.nan, (sc + w_ * np.where(u, m.size_pct[i], 0.0)) / (1 + w_))
        for key_, w_ in (cfg.get("score_mix") or {}).items():
            # blend the rank score with another indicator's cross-sectional percentile (hi52, resmom, ADX, RSI)
            x_ = pct_rank(np.where(u & ~np.isnan(sc), m.P[key_][i].astype(float), np.nan))
            sc = np.where(np.isnan(sc), np.nan, (sc + w_ * np.nan_to_num(x_, nan=0.0)) / (1 + w_))
        if use_rank or not v5_size:
            order = rank_desc(sc, m.tie_key(cfg, i))
            nvalid = int((~np.isnan(sc)).sum())
            pos_rank = {int(j): k for k, j in enumerate(order[:nvalid])}
        if use_rank and cfg.get("exit_score") == "pool":
            P_ = m.P
            a_ = pct_rank(np.where(u, P_["rs126"][i], np.nan))
            b_ = pct_rank(np.where(u, P_["dist200"][i], np.nan))
            ps = pct_rank(0.5 * a_ + 0.5 * b_)
            order = rank_desc(ps, m.tie_key(cfg, i))
            pos_rank = {int(j): k for k, j in enumerate(order[:int((~np.isnan(ps)).sum())])}
        if use_rank and not fill_only:
            xr = cfg["buffer"] * N
            if paused and cfg.get("pause_tight_exit", True):
                xr = N                                          # hard-coded India rule: a pause tightens the sell rank
            t_now = tranche_of(i) if (cfg.get("stagger") and not takeover) else None
            for j in list(pos):
                if j in exiting or pos[j]["stale"]:
                    continue
                if t_now is not None and pos[j].get("meta", {}).get("tr", 0) != t_now:
                    continue
                if takeover and j in carried:
                    # takeover review: a carried holding stays only if it ranks inside the sell buffer; none of the
                    # system's hold exemptions apply to it (they protect the system's own winners)
                    if pos_rank.get(j, 10 ** 9) >= xr:
                        pend_stock.append(dict(kind="exit", j=j, why="takeover", all=True)); exiting.add(j)
                    continue
                if (i - pos[j]["entry_i"]) < cfg.get("rank_min_hold", 0):
                    # exp60: a new buy is given time, unless it is already down more than rank_grace_loss
                    if not (cfg.get("rank_grace_loss") and C[i, j] / pos[j]["cost"] - 1 < -cfg["rank_grace_loss"]):
                        continue
                if cfg.get("crash_no_rank") and m.spy_dd[i] == m.spy_dd[i] and m.spy_dd[i] < -cfg["crash_no_rank"]:
                    continue                                    # exp60: no rank sells while SPY is deep below its 1-year high
                if (cfg.get("giant_slots") or cfg.get("pause_giants")) and pos[j].get("meta", {}).get("giant") and (not paused or cfg.get("pause_giants")):
                    top_, up_ = m.giants(i, 2 * cfg.get("giant_top", 20))
                    if top_[j] and up_[j]:
                        continue                                # exp58: a giant is held while it stays big and above a rising 200-day
                    pend_stock.append(dict(kind="exit", j=j, why="giant", all=True)); exiting.add(j)
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
        dip = cfg.get("pause_dip_rsi")
        if paused and not (dip or cfg.get("pause_giants")):
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
            if fill_only:
                # fill the tranche with the most empty slots (each tranche keeps its own slot count)
                ntr_ = len(cfg["stagger"]); cnt_ = [0] * ntr_
                for j_, p_ in pos.items():
                    if j_ not in exiting:
                        cnt_[p_.get("meta", {}).get("tr", 0) % ntr_] += 1
                scfg = dict(cfg, _tr=int(np.argmin(cnt_)), _ntr=ntr_)
            if paused and cfg.get("pause_giants") and not dip:
                # exp63: during a pause, free slots go only to giants (most-traded members above a rising 200-day)
                scfg = dict(scfg, giant_slots=cfg["pause_giants"], _giants_only=True)
            elif paused:
                # exp61: buy the dip during a pause. Only top-ranked candidates whose RSI is at or below pause_dip_rsi,
                # and at most pause_dip_slots new names per decision day.
                scfg = dict(scfg, rsi_max=dip, _dip_max=cfg.get("pause_dip_slots", 2))
            if (takeover or whole_review(i)) and cfg.get("stagger") and not fill_only:
                # takeover review (and an extra whole-book review) fills free slots in every tranche, using the cash each tranche's buys leave
                for tr_ in range(len(cfg["stagger"])):
                    spent_ = sum(o["dollars"] for o in pend_stock if o.get("kind") in ("new", "add"))
                    gl_ = any((o.get("meta") or {}).get("giant") for o in pend_stock if o.get("kind") == "new") or \
                        any(p_.get("meta", {}).get("giant") for p_ in pos.values())
                    tcfg = dict(scfg, _tr=tr_, _ntr=len(cfg["stagger"]), dedupe=True)
                    if gl_:
                        tcfg["giant_slots"] = 0                 # one giants slot for the whole book
                    _slot_orders(m, tcfg, i, sc, route, u, pos, exiting, navd, max(0.0, sleeve_cash - spent_), pend_stock, pos_rank)
            else:
                _slot_orders(m, scfg, i, sc, route, u, pos, exiting, navd, sleeve_cash, pend_stock, pos_rank)

    out = dict(curve=pd.Series(curve, index=m.dates[i0:i1]), expo=pd.Series(expo, index=m.dates[i0:i1]),
               costs=costs, final_positions=len(pos))
    # live (owner, 16 Oct 2026): the book at the last close and the orders the system will place at the next open
    out["book"] = dict(cash=cash, parked=dict(parked), positions={m.syms[j]: dict(adj_sh=p["sh"], cost=p["cost"], peak=p["peak"],
                       entry=str(m.dates[p["entry_i"]].date()), tr=p.get("meta", {}).get("tr", 0), giant=bool(p.get("meta", {}).get("giant")),
                       carried=bool(p.get("meta", {}).get("carried")), adds=p["adds"], sf=float(SF[i1 - 1, j]))
                       for j, p in pos.items()})
    out["pending"] = [dict(kind=o["kind"], sym=str(m.syms[o["j"]]), why=o.get("why"), all=o.get("all"), frac=o.get("frac"),
                           dollars=o.get("dollars"), min_dollars=o.get("min_dollars", 0.0), tr=int((o.get("meta") or {}).get("tr", 0)),
                           giant=bool((o.get("meta") or {}).get("giant"))) for o in pend_stock]
    out["paused"] = bool(paused)
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
        if park == "leader_safe" and paused and ((cfg or {}).get("pause_dip_index_rsi") or (cfg or {}).get("dip_vix") or (cfg or {}).get("dip_fall")):
            # exp61: during a pause, park in the leading index for dip_hold sessions once SPY's 14-day RSI closes
            # at or below pause_dip_index_rsi (capitulation); otherwise short Treasuries as before
            c_ = cfg
            if c_.get("_dipx_last", -1) > i:
                c_["_dipx_until"] = -1                      # a new run started: forget the previous run's timer
            c_["_dipx_last"] = i
            if c_.get("_dipx_until", -1) < i:
                if c_.get("_dip_rand") is not None:          # audit placebo: the index hold starts on random dates instead
                    trig = (i - 1) in c_["_dip_rand"]
                else:
                  trig = bool(c_.get("pause_dip_index_rsi")) and any(m.etf_rsi(t_)[i - 1] <= c_["pause_dip_index_rsi"] for t_ in c_.get("dip_on", ("SPY",)))
                dv = None if c_.get("_dip_rand") is not None else c_.get("dip_vix")
                if dv and not trig:
                    # exp66: VIX spike. VIX peaked at or above dv[0] within 10 sessions and is now dv[1] below that peak
                    v_ = m.P.get("VIX")
                    if v_ is not None and i > 12:
                        pk = np.nanmax(v_[i - 11:i]); trig = bool(pk >= dv[0] and v_[i - 1] <= pk * (1 - dv[1]))
                df_ = None if c_.get("_dip_rand") is not None else c_.get("dip_fall")
                if df_ and not trig:
                    # exp66: steep fall. SPY closed at least df_[0] below its highest close of the last df_[1] sessions
                    sp_ = m.E["SPY"]["c"]
                    if i > df_[1] + 1:
                        trig = bool(sp_[i - 1] <= np.nanmax(sp_[i - 1 - df_[1]:i]) * (1 - df_[0]))
                if trig:
                    c_["_dipx_until"] = i + c_.get("dip_hold", 42)
            if c_.get("_dipx_until", -1) >= i:
                lb = c_.get("leader_lb", 126)
                def mom(t):
                    c = m.E[t]["c"]; a, b = c[i - 1], c[i - 1 - lb] if i - 1 - lb >= 0 else np.nan
                    return a / b - 1 if (a == a and b == b and b > 0) else -999
                return max(LEADERS, key=mom)
        if park == "leader_safe" and paused and (cfg or {}).get("tom_park"):
            # exp72 turn of month: during a pause, idle money holds the leading index over the last 2 and first 3
            # sessions of each month (the trading calendar is known in advance), short Treasuries otherwise
            if m.month_end[i] or m.day_in_month[i] <= 2 or (i + 1 < len(m.month_end) and m.month_end[i + 1]):
                lb = (cfg or {}).get("leader_lb", 126)
                def mom2(t):
                    c = m.E[t]["c"]; a, b = c[i - 1], c[i - 1 - lb] if i - 1 - lb >= 0 else np.nan
                    return a / b - 1 if (a == a and b == b and b > 0) else -999
                return max(LEADERS, key=mom2)
        pp = (cfg or {}).get("pause_park")
        if park == "leader_safe" and paused and pp and pp not in ("GLD", "GLD_trend") and pp in m.E:
            g = m.E[pp]["c"]                            # exp63: any other ETF as the pause refuge (TLT, IEF ...)
            if g[i] == g[i]:
                return pp
        if park == "leader_safe" and paused and pp in ("GLD", "GLD_trend"):
            # exp63: park in gold during a pause (always, or only while gold is above its 200-day average)
            g = m.E["GLD"]["c"]
            if g[i] == g[i] and (pp == "GLD" or m.etf_vs_sma("GLD", 200, i - 1) > 0):
                return "GLD"
        if park == "leader_safe" and paused:
            # exp57: optionally keep idle money in the leading index during a pause while that index is still
            # above its park_sma-session average (the pause stops new stock buys, not index exposure)
            ps = (cfg or {}).get("park_sma")
            if ps and i > ps:
                lb_ = (cfg or {}).get("leader_lb", 126)
                def mom_(t):
                    c = m.E[t]["c"]; a, b = c[i - 1], c[i - 1 - lb_]
                    return a / b - 1 if (a == a and b == b and b > 0) else -999
                ld = max(LEADERS, key=mom_)
                if m.etf_vs_sma(ld, ps, i - 1) > 0:
                    return ld
            return "SHY" if m.E["SHY"]["c"][i] == m.E["SHY"]["c"][i] else None
        lb = (cfg or {}).get("leader_lb", 126)
        def mom(t):
            c = m.E[t]["c"]
            a, b = c[i - 1], c[i - 1 - lb] if i - 1 - lb >= 0 else np.nan
            return a / b - 1 if (a == a and b == b and b > 0) else -999
        ld = max(LEADERS, key=mom)
        lv = (cfg or {}).get("lev_park")
        lvx = (cfg or {}).get("lev_vix")
        vx_ok = True
        if lvx:
            v_ = m.P.get("VIX"); vx_ok = v_ is not None and v_[i - 1] == v_[i - 1] and v_[i - 1] < lvx
        la = (cfg or {}).get("lev_after")
        if la and i - (cfg or {}).get("_i0", 0) < la:
            vx_ok = False                                      # exp78: no 2x in a new account's first lev_after sessions
        ldd = (cfg or {}).get("lev_dd")
        if ldd and (cfg or {}).get("_acct_dd", 0.0) < -ldd:
            vx_ok = False                                      # exp78: no 2x while the account is more than lev_dd below its peak
        lspy = (cfg or {}).get("lev_spy")
        if lspy and not (m.etf_vs_sma("SPY", lspy, i - 1) > 0):
            vx_ok = False
        sfx = "2xr" if (cfg or {}).get("lev_real") else "2x"
        lo_ = (cfg or {}).get("lev_only")
        if lo_ and ld not in lo_:
            vx_ok = False                                      # exp78: 2x only through these leaders (QLD/SSO track well; USD does not)
        if lv and vx_ok and not paused and park != "leader_safe_1x" and i > lv and m.etf_vs_sma(ld, lv, i - 1) > 0 and m.E[ld + sfx]["c"][i] == m.E[ld + sfx]["c"][i]:
            return ld + sfx                                    # exp78: idle money in the 2x leader while it is above its lv-day average
        return ld
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
    if cfg.get("disp_cap") and m.lit()["disp_hi"][i]:
        n_stock = min(n_stock, cfg["disp_cap"])           # exp75 idea 6: high return dispersion (Stivers & Sun 2010)
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
    if cfg.get("_dip_max") is not None:
        free = min(free, cfg["_dip_max"])
    hh = cfg.get("skip_buys_near_high")
    if hh is not None:
        # exp67: no new stock buys while SPY closes within hh of its 1-year high (money waits in the leading index)
        sp_ = m.E["SPY"]["c"]
        if i > 252 and sp_[i] >= np.nanmax(sp_[i - 251:i + 1]) * (1 - hh):
            free = 0
    if cfg.get("narrow_index") and m.narrow[i]:
        free = 0                                                # journal: new buys in narrow markets lagged SPY
    gs = cfg.get("giant_slots")
    gd = cfg.get("giant_dyn")
    if gd and m.lit()["gap"][i] > gd[1] and m.E["QQQ"]["c"][i] > m.e_sma50["QQQ"][i] and m.e_sma50["QQQ"][i] > m.e_sma50["QQQ"][i - 5]:
        gs = gd[0]                                        # exp75 idea 2: mega-cap-led market, more giant slots
    if gs and free > 0:
        # exp58: second entry path. Up to giant_slots of the slots hold the giant_top most-traded members that are
        # above a rising 200-day average, strongest 6-month first; no rank, ADX or volume gate.
        top_, up_ = m.giants(i, cfg.get("giant_top", 20))
        have = sum(1 for j in pos if j not in exiting and pos[j].get("meta", {}).get("giant"))
        cand = [j for j in np.flatnonzero(top_ & up_ & u) if j not in pos]
        cand.sort(key=lambda j: -np.nan_to_num(P["rs126"][i, j], nan=-9))
        for j in cand[:max(0, min(gs - have, free))]:
            out.append(dict(kind="new", j=int(j), dollars=slot, route=1, stop=(P["SMA50"][i, j] - P["ATR"][i, j]),
                            min_dollars=0.5 * slot, meta={"tr": tr or 0, "w": 1.0, "giant": True}))
            free -= 1
    if cfg.get("_giants_only"):
        free = 0
    order = rank_desc(sc, m.tie_key(cfg, i))
    nvalid = int((~np.isnan(sc)).sum())
    q = (P["Cf"][i] - P["EMA20"][i]) / P["ATR"][i]
    for j in order[:nvalid]:
        if free <= 0:
            break
        j = int(j)
        if j in pos:
            continue
        if cfg.get("dedupe") and any(o.get("j") == j and o.get("kind") == "new" for o in out):
            continue                                            # exp77: the giants slot already took this stock
        if cfg["trend"] and not P["cont"][i, j]:
            continue
        if cfg["zone_gate"] and not (-1.0 <= q[j] <= 2.0):
            continue
        if cfg.get("index_hurdle") and not m.is_etf[j]:
            # a stock is only bought if its 6-month return beats the best of SPY/QQQ/SMH (else the index wins the slot)
            c_ = P["Cf"]; hl = cfg.get("index_hurdle_lb", 126); hm = cfg.get("index_hurdle_margin", 0.0)
            if i < hl or not (c_[i, j] / c_[i - hl, j] - 1 > hm + max(m.E[t]["c"][i] / m.E[t]["c"][i - hl] - 1 for t in ("SPY", "QQQ", "SMH")
                                                                          if m.E[t]["c"][i - hl] == m.E[t]["c"][i - hl])):
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
        gx = cfg.get("giant_gate_exempt")
        is_g = bool(gx) and m.giants(i, gx)[0][j]          # exp64: the gx most-traded members skip the OBV and ADX gates
        if cfg.get("pocket_pivot") and not m.volf()["pp10"][i, j]:
            continue                                            # exp76: pocket pivot in the last 10 sessions
        if cfg.get("vdry_max") and not (m.volf()["vdry"][i, j] <= cfg["vdry_max"]):
            continue                                            # exp76: volume dry-up (quiet base) before buying
        if cfg.get("gap_max") and not (m.lit()["gap90"][i, j] <= cfg["gap_max"]):
            continue                                            # exp75 idea 14: Clenow gap filter
        if cfg.get("volr_max") and m.lit()["volr"][i, j] > cfg["volr_max"]:
            continue                                            # exp75 idea 12: late-stage volume surge (Lee & Swaminathan 2000)
        if cfg.get("overnight_filter") and m.lit()["in63"][i, j] < 0 and m.lit()["on63"][i, j] > cfg["overnight_filter"]:
            continue                                            # exp75 idea 8: gain made overnight, faded intraday
        if cfg.get("obv_min") is not None and not is_g and not (P["OBVT"][i, j] >= cfg["obv_min"]):
            continue
        if cfg.get("rvol_min") and not (P["RVOL"][i, j] >= cfg["rvol_min"]):
            continue
        if cfg.get("adx_max") and not (P["ADX"][i, j] <= cfg["adx_max"]):
            continue                                            # buy only early in a trend (ADX still low)
        if cfg.get("adx_min") and not is_g and not (P["ADX"][i, j] >= cfg["adx_min"] and P["PDI_GT"][i, j]):
            ex = cfg.get("adx_exempt_rs")                       # exp57: the strongest stocks skip the ADX gate
            if not ex:
                continue
            if "_rsp" not in cfg or cfg["_rsp"][0] != i:
                cfg["_rsp"] = (i, pct_rank(np.where(u, P["rs126"][i].astype(float), np.nan)))
            if not (cfg["_rsp"][1][j] >= ex):
                continue
        if cfg.get("hi52_min") and not (P["hi52"][i, j] >= cfg["hi52_min"]):
            continue
        if cfg.get("buy_dip5") and not (i >= 5 and P["Cf"][i, j] <= P["Cf"][i - 5, j]):
            continue                                            # exp68: buy only after the stock dipped over the last 5 sessions
        if cfg.get("buy_qqq_macd_down") and not (m.etf_macd("QQQ", i) < m.etf_macd("QQQ", i - 5)):
            continue                                            # exp68: buy only while QQQ's MACD histogram is falling (market dipping)
        if cfg.get("max_per_sector") and m.sector is not None and m.sector[i, j] >= 0:
            held_s = [k for k in pos if k not in exiting and m.sector[i, k] == m.sector[i, j]] + [o["j"] for o in out if o.get("kind") == "new" and m.sector[i, o["j"]] == m.sector[i, j]]
            if len(held_s) >= cfg["max_per_sector"]:
                continue                                        # exp68: at most this many holdings per industry
        if cfg.get("corr_max"):
            hold = [k for k in pos if k not in exiting] + [o["j"] for o in out if o.get("kind") == "new"]
            if hold and i >= 63:
                rr = np.nan_to_num(P["ret1"][i - 62:i + 1][:, [j] + hold].astype(float))
                cc = np.corrcoef(rr.T)[0, 1:]
                if np.nanmax(cc) > cfg["corr_max"]:
                    continue                                    # exp68: skip a name that moves almost in step with a holding
        if cfg.get("rs_rising") and not (i >= 5 and P["rs126"][i, j] > P["rs126"][i - 5, j]):
            continue                                            # exp67: 6-month strength vs SPY rising over the last week
        if cfg.get("dist200_min") is not None and not (P["dist200"][i, j] >= cfg["dist200_min"]):
            continue                                            # exp67: at least this far above the 200-day average
        if cfg.get("minervini"):                                 # stage-2 trend template (Minervini 2013), RS left to the rank
            c_, s50, s150, s200 = P["Cf"][i, j], P["SMA50"][i, j], P["SMA150"][i, j], P["SMA200"][i, j]
            s200p = P["SMA200"][i - cfg.get("minv_rise", 21), j] if i >= 21 else np.nan
            if not (c_ > s150 and c_ > s200 and s150 > s200 and s200 > s200p and c_ > s50
                    and (cfg["minervini"] == "core" or (s50 > s150 and c_ >= cfg.get("minv_lo", 1.3) * P["LO52"][i, j]
                                                        and c_ >= cfg.get("minv_hi", 0.75) * P["HI52C"][i, j]))):
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
