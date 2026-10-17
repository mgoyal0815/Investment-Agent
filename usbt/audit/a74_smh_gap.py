"""Why R1 trails SMH in strong chip years (owner, 17 Oct 2026). Fresh $10k each calendar year (as a73), daily book:
share of the account in stocks, parked in an ETF, in cash/SHY; days paused; share of stock money in semiconductors
(SMH's 2025 holdings list as a proxy); what was parked."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, numpy as np, pandas as pd, yardstick as Y, engine as X
from exp24_timing import mat
PATH = "../data/panel_ndx.pkl"
R1 = json.load(open("../system_R1.json"))["cfg"]
SEMI = set("NVDA TSM AVGO AMD ASML QCOM TXN INTC MU AMAT LRCX KLAC ADI MRVL NXPI MCHP ON MPWR SWKS QRVO TER ENTG SNPS CDNS ARM SMCI".split())


def one(y):
    m = Y.MM
    s = m.dates[m.dates.searchsorted(pd.Timestamp(f"{y}-01-01"))]
    e = m.dates[min(m.dates.searchsorted(pd.Timestamp(f"{y}-12-31"), side="right"), len(m.dates)) - 1]
    r = X.run(m, mat(R1), start=str(s.date()), end=str(e.date()), record=True)
    d = r["daily"]; t = r["trades"]
    nav = d["nav"]
    buys = t[t.side == "BUY"]
    semi_buy = buys[buys.sym.isin(SEMI)].value.sum() / max(buys.value.sum(), 1)
    return dict(year=y, r1=float(nav.iloc[-1] / 10000 - 1), smh=float(m.E["SMH"]["c"][m.dates.get_loc(e)] / m.E["SMH"]["c"][m.dates.get_loc(s)] - 1),
                in_stocks=float((d["stocks"] / nav).mean()), parked=float((d["parked"] / nav).mean()),
                cash_shy=float(((d["cash"] + d.get("shy", 0)) / nav).mean()), days_paused=float(d["paused"].astype(float).mean()),
                first_q_stocks=float((d["stocks"] / nav).iloc[:63].mean()), semis_share_of_buys=float(semi_buy),
                stocks_bought=int(buys.sym.nunique()), sells_rank=int(((t.side == "SELL") & t.why.astype(str).str.contains("rank")).sum()),
                sells_lock=int(((t.side == "SELL") & t.why.astype(str).str.contains("lock")).sum()))


if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    T = pd.DataFrame(pool.map(one, range(2016, 2026), chunksize=1)).set_index("year")
    T.to_csv("../audit/a74_smh_gap.csv")
    pd.set_option("display.width", 200)
    print(T.round(2).to_string())
