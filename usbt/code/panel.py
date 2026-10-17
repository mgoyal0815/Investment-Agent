"""Build the price panel and every indicator both systems need.

    python panel.py <data_dir>      -> <data_dir>/panel.pkl

Session index = SPY's dates (framework 3.1). Stock frames are bounded-forward-filled for
indicators (3.4) while raw NaN is kept for fills and delisting (19.16). Indicator
definitions follow the live sleeve_select.py exactly (simple ATR, rv30 not annualised,
EMA20 adjust=False, 20-session MEDIAN dollar volume). India indicators follow
v5-india/code/backtest.py (126-session median turnover, 252-session volatility).
"""
import json, os, sys
import numpy as np
import pandas as pd

D = sys.argv[1]
ETFS = ["SPY", "QQQ", "SMH", "TLT", "GLD", "UUP", "DBC", "SHY", "HYG", "BIL", "XLK", "IEF", "MTUM"]
KEYS = os.environ.get("V5_KEYS", "/home/user/v5-forward-test/keys")     # clone of mgoyal0815/v5-forward-test
# sector / industry ETFs the stock picker may hold like a stock (engine switch etf_in_universe)
SECTOR_ETFS = ["SMH", "SOXX", "XLK", "IGV", "XLE", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC"]
# broad index ETFs that may also compete for slots (exp48); written to a separate panel so older results stay reproducible
if os.environ.get("IDX_COLS"):
    SECTOR_ETFS = SECTOR_ETFS + ["QQQ", "SPY", "DIA", "IWM"]


def load(sym):
    return pd.read_parquet(f"{D}/raw/{sym}.parquet").set_index("date").sort_index()


def preflight():
    """REBUILD.md: refuse to build from an incomplete download (a panel without the sector ETFs or the Nasdaq-100
    names silently changes every result)."""
    miss = [s for s in ETFS + SECTOR_ETFS if not os.path.exists(f"{D}/raw/{s}.parquet")]
    if os.environ.get("NDX"):
        t = [l.strip() for l in open(f"{D}/ndx_tickers.txt") if l.strip()]
        miss += [s for s in t if not (os.path.exists(f"{D}/raw/{s}.parquet") or os.path.exists(f"{D}/raw/{s}.none"))]
    if miss:
        raise SystemExit(f"panel: {len(miss)} symbols never fetched ({', '.join(sorted(set(miss))[:20])} ...): run fetch.py again")


def stamp(path, shape, kept, dropped):
    """Write <panel>.BUILD.json next to the panel: what was built, from which fetch, with which libraries."""
    import datetime as dt, hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    fetch = json.load(open(f"{D}/FETCH.json")) if os.path.exists(f"{D}/FETCH.json") else None
    info = dict(built_utc=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", panel=os.path.basename(path), shape=list(shape),
                stocks_kept=kept, dropped=dropped, sha256=h.hexdigest(), numpy=np.__version__, pandas=pd.__version__, fetch=fetch)
    json.dump(info, open(path + ".BUILD.json", "w"), indent=1)
    print("stamp", path + ".BUILD.json", info["sha256"][:16])


def main():
    preflight()
    spy = load("SPY")
    idx = spy.index[(spy.index >= "1999-01-01")]
    # ---- ETFs (not forward-filled here; the overlay forward-fills prices and reads raw NaN
    # for availability, 3.2)
    etf = {}
    for s in ETFS:
        if os.path.exists(f"{D}/raw/{s}.parquet"):
            etf[s] = load(s).reindex(idx)[["open", "close"]]
    # ---- stocks
    live = set()
    for k in ["us_only_2026-10-12", "us_only_2026-09-07", "us_only_2026-09-30"]:
        live |= {l.strip() for l in open(f"{KEYS}/{k}") if l.strip()}
    sp = pd.read_csv(f"{D}/sp500_ticker_start_end.csv")
    sp["ticker"] = sp["ticker"].str.replace(".", "-", regex=False)
    sp_win = sp.groupby("ticker").agg(start=("start_date", "min"), end=("end_date", lambda x: x.fillna("2099").max()))
    if os.environ.get("NDX"):
        # add point-in-time Nasdaq-100 members (ndx_members.py) as their own membership windows
        h = pd.read_csv(f"{D}/ndx_hist.csv", parse_dates=["date"]).sort_values("date")
        w = {}
        for d, t in zip(h.date, h.tickers):
            for s in t.split(","):
                a, b = w.get(s, (d, d))
                w[s] = (min(a, d), max(b, d))
        nd = pd.DataFrame({s: dict(start=str(a.date()), end=("2099" if b == h.date.max() else str(b.date()))) for s, (a, b) in w.items()}).T
        nd = nd[~nd.index.isin(sp_win.index)]
        sp_win = pd.concat([sp_win, nd])
    syms, frames, dropped = [], {}, {}
    for f in sorted(os.listdir(f"{D}/raw")):
        if not f.endswith(".parquet"):
            continue
        s = f[:-8]
        if s in ETFS and s not in SECTOR_ETFS:
            continue
        if s not in live and s not in sp_win.index and s not in SECTOR_ETFS:
            continue     # only in the superseded pit/mcap lists (live meta: DO NOT USE)
        df = load(s)
        df = df[(df.index >= "1999-01-01")]
        df = df[df.index.isin(idx)]
        if len(df) < 60:
            dropped[s] = "fewer than 60 sessions"
            continue
        if s not in live and s not in SECTOR_ETFS:
            # an S&P-history ticker: FMP's history must overlap the index membership window,
            # otherwise the ticker now belongs to a different company (EMC, SGP, WB ...)
            st, en = sp_win.loc[s, "start"], sp_win.loc[s, "end"]
            if df.index[-1] < pd.Timestamp(st) or df.index[0] > pd.Timestamp(en):
                dropped[s] = f"FMP history {df.index[0].date()}..{df.index[-1].date()} misses S&P window {st}..{en}"
                continue
        c = df["close"]
        if (c <= 0).any():
            df = df[df["close"] > 0]
            c = df["close"]
        # audit (15 Oct 2026): this gate reads the whole future price path. AUDIT_KEEP lists S&P members it removed
        # whose data inside their index window is clean, so they can be put back for the survivorship check.
        # Standard since the audit: S&P members with clean data inside their index window are kept even if their
        # later price collapsed (Signature Bank, Rite Aid, Tupperware); the gate only screens the rest. Sovereign
        # Bancorp (SOV) stays out: FMP's SOV prices are fractions of a cent (bad data, not a collapse).
        KEEP = {"SBNY", "RAD", "TUP"} | set(os.environ.get("AUDIT_KEEP", "").split(","))
        if c.max() / c.iloc[-1] > 1000 and s not in KEEP:
            dropped[s] = "split-span gate (live universe_meta rule)"
            continue
        frames[s] = df
        syms.append(s)
    print(len(syms), "stocks kept,", len(dropped), "dropped", flush=True)
    json.dump(dropped, open(f"{D}/dropped_audit.json" if os.environ.get("AUDIT_KEEP") else f"{D}/dropped_ndx.json" if os.environ.get("NDX") else (f"{D}/dropped_idx.json" if os.environ.get("IDX_COLS") else f"{D}/dropped.json"), "w"), indent=0)

    def wide(col):
        return pd.DataFrame({s: frames[s][col] for s in syms}).reindex(idx)

    O, H, L, C, V = (wide(c) for c in ("open", "high", "low", "close", "volume"))
    raw_C, raw_O = C.copy(), O.copy()

    def bounded_ffill(d):
        v = d.notna()
        c = v.cumsum()
        return d.ffill().mask(c.eq(0) | (c.eq(v.sum(), axis=0) & ~v))

    Cf, Hf, Lf, Vf = (bounded_ffill(x) for x in (C, H, L, V))
    Vf = Vf.where(raw_C.notna(), 0.0).where(Cf.notna())     # a halted day trades nothing

    # ---- split factor: raw traded price = adjusted price x product of later split ratios
    splits = json.load(open(f"{D}/splits.json")) if os.path.exists(f"{D}/splits.json") else {}
    SF = pd.DataFrame(1.0, index=idx, columns=syms)
    for s in syms:
        for dte, r in splits.get(s, []):
            if r and r > 0 and r != 1:
                SF.loc[SF.index < pd.Timestamp(dte), s] *= r

    # ---- v5 indicators (sleeve_select.py)
    SMA200 = Cf.rolling(200, min_periods=200).mean()
    SMA50 = Cf.rolling(50, min_periods=50).mean()
    EMA20 = Cf.ewm(span=20, adjust=False, min_periods=20).mean()
    RMAX20 = Cf.shift(1).rolling(20, min_periods=20).max()
    pc = Cf.shift(1)
    TR = pd.DataFrame(np.maximum.reduce([(Hf - Lf).abs().values, (Hf - pc).abs().values, (Lf - pc).abs().values]),
                      index=idx, columns=syms)
    ATR = TR.rolling(14, min_periods=14).mean()
    RV30 = Cf.pct_change(fill_method=None).rolling(30, min_periods=30).std()
    DV = Cf * Vf
    ADV20 = DV.rolling(20, min_periods=10).median()
    cont = ((Cf > SMA50).astype(int) + (SMA50 > SMA50.shift(20)).astype(int) + (Cf > Cf.shift(63)).astype(int)) >= 2
    rec = (Cf > EMA20) & (EMA20 > EMA20.shift(10)) & ((Cf > SMA50) | (Cf > RMAX20)) & ~cont
    spyc = etf["SPY"]["close"].ffill()
    rs126 = (Cf / Cf.shift(126) - 1).sub(spyc / spyc.shift(126) - 1, axis=0)
    dist200 = (Cf - SMA200) / SMA200
    # ---- India indicators (backtest.py)
    traded = raw_C.notna()
    medv = DV.where(traded).rolling(126, min_periods=100).median()
    hist = traded.rolling(252, min_periods=1).sum()
    vol = Cf.pct_change(fill_method=None).rolling(252, min_periods=240).std() * np.sqrt(252)
    m6 = (Cf / Cf.shift(126) - 1) / vol
    m12 = (Cf / Cf.shift(252) - 1) / vol
    ret1 = Cf.pct_change(fill_method=None)

    # residual momentum (Blitz, Huij & Martens): 12-1 month sum of returns net of SPY beta, over the
    # residual volatility; and nearness to the 52-week high (George & Hwang)
    mret = spyc.pct_change(fill_method=None)
    mv = mret.rolling(252, min_periods=200).var()
    cov = ret1.mul(mret, axis=0).rolling(252, min_periods=200).mean() - ret1.rolling(252, min_periods=200).mean().mul(mret.rolling(252, min_periods=200).mean(), axis=0)
    beta = cov.div(mv, axis=0).shift(1)
    resid = ret1 - beta.mul(mret, axis=0)
    resmom = resid.shift(21).rolling(231, min_periods=180).sum() / resid.rolling(252, min_periods=200).std()
    hi52 = Cf / Cf.rolling(252, min_periods=200).max()
    # how closely each name has moved with SMH and with QQQ over the last 126 sessions (daily
    # returns, correlation), for the "lean into the leading index" tilt. Prices only, so delisted
    # names are covered.
    corr = {}
    for k in ("SMH", "QQQ"):
        er = etf[k]["close"].ffill().pct_change(fill_method=None)
        corr[k] = ret1.rolling(126, min_periods=100).corr(er)
    # trend-strength and turn indicators (owner, 15 Oct 2026): ADX(14, Wilder), MACD histogram (12/26/9) as a
    # share of price, and a 55-session breakout (close at a new 55-session high)
    up, dn = Hf.diff(), -Lf.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    wild = lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    atr_w = wild(TR)
    pdi, ndi = 100 * wild(pdm) / atr_w, 100 * wild(ndm) / atr_w
    ADX = wild(100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan))
    macd = Cf.ewm(span=12, adjust=False, min_periods=26).mean() - Cf.ewm(span=26, adjust=False, min_periods=26).mean()
    MACDH = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / Cf
    BO55 = Cf >= Cf.shift(1).rolling(55, min_periods=55).max()
    dlt = Cf.diff()
    g_, l_ = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(), (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    RSI = 100 - 100 / (1 + g_ / l_.replace(0, np.nan))
    EMA50 = Cf.ewm(span=50, adjust=False, min_periods=50).mean()
    vix = pd.read_csv(f"{D}/vix.csv", parse_dates=["date"]).set_index("date")["close"].reindex(idx).ffill() if os.path.exists(f"{D}/vix.csv") else None
    f32 = lambda x: x.to_numpy(dtype=np.float32)
    P = dict(dates=idx, syms=np.array(syms), etf={k: (v["open"].to_numpy(), v["close"].to_numpy()) for k, v in etf.items()},
             O=f32(raw_O), C=f32(raw_C), Cf=f32(Cf), SF=f32(SF),
             SMA200=f32(SMA200), SMA50=f32(SMA50), EMA20=f32(EMA20), ATR=f32(ATR), RV30=f32(RV30),
             ADV20=f32(ADV20), cont=cont.to_numpy(), rec=rec.to_numpy(), rs126=f32(rs126), dist200=f32(dist200),
             medv=f32(medv), hist=f32(hist), m6=f32(m6), m12=f32(m12), ret1=f32(ret1),
             in_live=np.array([s in live for s in syms]), is_etf=np.array([s in SECTOR_ETFS for s in syms]),
             corr_SMH=f32(corr["SMH"]), corr_QQQ=f32(corr["QQQ"]), resmom=f32(resmom), hi52=f32(hi52),
             ADX=f32(ADX), MACDH=f32(MACDH), BO55=BO55.to_numpy(), PDI_GT=(pdi > ndi).to_numpy(),
             RSI=f32(RSI), EMA50=f32(EMA50), VIX=(vix.to_numpy() if vix is not None else None))
    if os.environ.get("AUDIT_KEEP"):
        pd.to_pickle(P, f"{D}/panel_audit.pkl"); return
    out = f"{D}/panel_ndx.pkl" if os.environ.get("NDX") else (f"{D}/panel_idx.pkl" if os.environ.get("IDX_COLS") else f"{D}/panel.pkl")
    pd.to_pickle(P, out)
    print("panel", C.shape, idx[0].date(), idx[-1].date())
    stamp(out, C.shape, len(syms), len(dropped))


if __name__ == "__main__":
    main()
