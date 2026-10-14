"""Point-in-time Nasdaq-100 membership 2004-2026 from shardul0701/NQ100-Survivorship-bias-data-2004-2026
(2015+ from jmccarrell/n100tickers; 2004-2014 reconstructed backward from the 2015 anchor).
Writes ../data/ndx_hist.csv: date, tickers (comma list), one row per change date."""
import glob, sys
import yaml
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/user/shardul0701/nq100-survivorship-bias-data-2004-2026/src/nasdaq_100_ticker_history"
rows = []
for f in sorted(glob.glob(f"{SRC}/n100-ticker-changes-*.yaml")):
    y = yaml.load(open(f), Loader=yaml.BaseLoader)   # BaseLoader: tickers like ON stay strings
    cur = set(y["tickers_on_Jan_1"])
    rows.append((pd.Timestamp(f"{y['year']}-01-01"), sorted(cur)))
    for d, ch in sorted((y.get("changes") or {}).items()):
        cur = (cur - set(ch.get("difference") or [])) | set(ch.get("union") or [])
        rows.append((pd.Timestamp(d), sorted(cur)))
H = pd.DataFrame([(d, ",".join(t).replace(".", "-")) for d, t in rows], columns=["date", "tickers"])
H.to_csv("../data/ndx_hist.csv", index=False)
allt = sorted({t for _, ts in rows for t in ts})
print(len(H), "snapshots", H.date.min().date(), H.date.max().date(), len(allt), "distinct tickers; sizes", min(len(t) for _, t in rows), max(len(t) for _, t in rows))
open("../data/ndx_tickers.txt", "w").write("\n".join(t.replace(".", "-") for t in allt))
