"""Split history per symbol from FMP (stable/splits), so whole-share counts and per-share
commission can be computed on the price a buyer actually paid rather than on today's
split-adjusted price (AAPL in 2000 is $0.9 adjusted and ~$100 as traded).

    python fetch_splits.py <data_dir>      -> <data_dir>/splits.json  {sym: [[date, ratio], ...]}
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

D = sys.argv[1]
KEY = os.environ["FMP_API_KEY"]
out = f"{D}/splits.json"
have = json.load(open(out)) if os.path.exists(out) else {}
syms = sorted(f[:-8] for f in os.listdir(f"{D}/raw") if f.endswith(".parquet"))
todo = [s for s in syms if s not in have]
print(len(todo), "to fetch", flush=True)


def one(s):
    url = f"https://financialmodelingprep.com/stable/splits?symbol={s}&apikey={KEY}"
    for a in range(5):
        time.sleep(1.5)    # 8 threads: at most ~200 calls a minute
        r = subprocess.run(["curl", "-sS", "--max-time", "60", url], capture_output=True, text=True)
        try:
            j = json.loads(r.stdout)
        except Exception:
            j = None
        if isinstance(j, list):
            return s, [[x["date"], x["numerator"] / x["denominator"]] for x in j
                       if x.get("numerator") and x.get("denominator")]
        time.sleep(10 * (a + 1))
    return s, None


with ThreadPoolExecutor(8) as ex:
    for k, (s, v) in enumerate(ex.map(one, todo)):
        if v is not None:
            have[s] = v
        if k % 200 == 0:
            json.dump(have, open(out, "w"))
            print(k, flush=True)
json.dump(have, open(out, "w"))
print("done", len(have))
