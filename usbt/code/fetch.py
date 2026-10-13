"""Download dividend-adjusted daily OHLCV from FMP for the candidate universe and the ETFs.

    python fetch.py <data_dir>

Candidates = every name the live v5 store has ever used as a universe (active and delisted)
UNION every ticker that was ever in the S&P 500 since 1996 (fja05680/sp500). Two calls per
symbol (FMP returns at most 5,000 rows): 1999-2012 and 2013-today. Resumable: a symbol whose
parquet exists is skipped. A symbol FMP has no data for is recorded in missing.txt.
"""
import json, os, subprocess, sys, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

D = sys.argv[1]
KEY = os.environ["FMP_API_KEY"]
BASE = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
KEYS = "/home/user/v5-forward-test/keys"
ETFS = ["SPY", "QQQ", "SMH", "TLT", "GLD", "UUP", "DBC", "SHY", "HYG", "BIL", "XLK", "IEF", "MTUM"]
WINDOWS = [("1999-01-01", "2012-12-31"), ("2013-01-01", "2026-10-12")]


class Pacer:
    def __init__(self, per_min):
        self.per_min, self.lock, self.log = per_min, threading.Lock(), deque()

    def take(self):
        while True:
            with self.lock:
                now = time.time()
                while self.log and now - self.log[0] >= 60:
                    self.log.popleft()
                if len(self.log) < self.per_min:
                    self.log.append(now)
                    return
                wait = 60 - (now - self.log[0]) + 0.05
            time.sleep(wait)


P = Pacer(int(os.environ.get("FMP_PER_MIN", "230")))


def one(sym, frm, to):
    url = f"{BASE}?symbol={sym}&from={frm}&to={to}&limit=5000&apikey={KEY}"
    for a in range(6):
        P.take()
        r = subprocess.run(["curl", "-sS", "--max-time", "90", url], capture_output=True, text=True)
        try:
            j = json.loads(r.stdout)
        except Exception:
            j = None
        if isinstance(j, list):
            return j
        if isinstance(j, dict) and "Limit" in str(j):
            time.sleep(20 * (a + 1))
            continue
        time.sleep(3 * (a + 1))
    raise RuntimeError(f"{sym} {frm}: no answer")


def get(sym):
    out = f"{D}/raw/{sym}.parquet"
    if os.path.exists(out) or os.path.exists(f"{D}/raw/{sym}.none"):
        return
    rows = []
    for frm, to in WINDOWS:
        rows += one(sym, frm, to)
    if not rows:
        open(f"{D}/raw/{sym}.none", "w").close()
        return
    df = pd.DataFrame(rows).rename(columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low",
                                            "adjClose": "close"})
    df = df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date").to_parquet(out, index=False)


def candidates():
    s = set()
    for k in ["us_only_2026-10-12", "us_only_2026-09-07", "us_only_2026-09-30",
              "backtest_universe_pit_2026-09-06", "backtest_universe_mcap8bn_2026-09-06"]:
        p = f"{KEYS}/{k}"
        if os.path.exists(p):
            s |= {l.strip() for l in open(p) if l.strip()}
    sp = pd.read_csv(f"{D}/sp500_ticker_start_end.csv")
    s |= {t.replace(".", "-") for t in sp["ticker"]}
    return sorted(s)


if __name__ == "__main__":
    os.makedirs(f"{D}/raw", exist_ok=True)
    syms = ETFS + [c for c in candidates() if c not in ETFS]
    if "--reverse" in sys.argv:
        syms = syms[::-1]
    print(len(syms), "symbols", flush=True)
    done = [0]

    def job(s):
        try:
            get(s)
        except Exception as e:
            print("FAIL", s, e, flush=True)
        done[0] += 1
        if done[0] % 100 == 0:
            print(done[0], flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(job, syms))
    print("done")
