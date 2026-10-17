"""Stamp a panel built before the 17 Oct 2026 guards (REBUILD.md) with a BUILD.json after the fact, so results scored
on it carry its fingerprint. Same fields as panel.stamp(); the build-time library versions and fetch record are not
known, so they are marked as such. The SHA-256 is what matters: it identifies the exact file a result came from.

    python3 stamp_existing.py ../data/panel_ndx.pkl [dropped_ndx.json]"""
import datetime as dt, hashlib, json, os, sys
import pandas as pd

path = sys.argv[1]
drop = sys.argv[2] if len(sys.argv) > 2 else None
if os.path.exists(path + ".BUILD.json"):
    raise SystemExit(f"{path}.BUILD.json exists: not overwriting")
h = hashlib.sha256()
with open(path, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
P = pd.read_pickle(path)
shape = list(P["Cf"].shape) if "Cf" in P else [len(P["dates"]), len(P["syms"])]
dropped = len(json.load(open(os.path.join(os.path.dirname(path), drop)))) if drop else None
info = dict(built_utc=dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc).isoformat(timespec="seconds"),
            panel=os.path.basename(path), shape=shape, stocks_kept=shape[1], dropped=dropped, sha256=h.hexdigest(),
            numpy="unknown (built before stamping)", pandas="unknown (built before stamping)", fetch=None,
            first_date=str(pd.Timestamp(P["dates"][0]).date()), last_date=str(pd.Timestamp(P["dates"][-1]).date()),
            note="stamped after the fact on " + dt.date.today().isoformat() + " by stamp_existing.py; built_utc is the file time")
json.dump(info, open(path + ".BUILD.json", "w"), indent=1)
print("stamp", path + ".BUILD.json", info["sha256"][:16], shape)
