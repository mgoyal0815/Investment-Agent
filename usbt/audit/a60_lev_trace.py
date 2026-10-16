"""Where does the 2x-parking worst fall come from? Calendar [-1, 13] (worst -57.5% vs R -47.7%): every third monthly
start, find the start with the deepest fall for R and for R + 2x parking above 200-day; then the peak/trough dates and
what the account held at the peak (stocks / 1x index / 2x index)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, numpy as np, pandas as pd, engine as X, yardstick as Y
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"


def job(a):
    cfg, s = a
    c = X.run(Y.MM, mat(cfg), start=str(s.date()), end=Y.STD_END)["curve"]
    dd = c / c.cummax() - 1
    t = dd.idxmin(); p = c[:t].idxmax()
    return dict(start=s, dd=float(dd.min()), peak=p, trough=t)


if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    R = dict(json.load(open("../system_R.json"))["cfg"], stagger=[-1, int(sys.argv[1]) if len(sys.argv) > 1 else 13])
    L = dict(R, lev_park=200, lev_after=126)
    for lab, c in (("R", R), ("R + 2x", L)):
        T = pd.DataFrame(pool.map(job, [(c, s) for s in Y.starts(step=3)]))
        w = T.sort_values("dd").head(5)
        print(f"\n{lab}: worst falls"); print(w.to_string(index=False))
    s = pd.DataFrame(pool.map(job, [(L, s) for s in Y.starts(step=3)])).sort_values("dd").iloc[0]
    Y._init(PATH); m = Y.MM
    for lab, c in (("R", R), ("R + 2x", L)):
        r = X.run(m, mat(c), start=str(pd.Timestamp(s.start).date()), end=Y.STD_END, record=True); d = r["daily"]; d.index = pd.to_datetime(d.index)
        pk, tr = pd.Timestamp(s.peak), pd.Timestamp(s.trough)
        seg = d.loc[pk:tr]
        print(f"\n{lab}, start {pd.Timestamp(s.start).date()}, {pk.date()} -> {tr.date()}: fall {100*(seg.nav.iloc[-1]/seg.nav.iloc[0]-1):.1f}%,"
              f" avg in stocks {100*(seg.stocks/seg.nav).mean():.0f}%, avg parked {100*(seg.parked/seg.nav).mean():.0f}%, paused {100*seg.paused.mean():.0f}% of days")
        t = r["trades"]; t["date"] = pd.to_datetime(t.date)
        tp = t[(t.date >= pk - pd.Timedelta(days=60)) & (t.date <= tr)]
        print("  ETF/park trades:", ", ".join(f"{x.date.date()} {x.sym} {x.side}" for x in tp.itertuples() if x.sym in m.E)[:400])
