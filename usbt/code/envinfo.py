"""The scoring environment, checked (REBUILD.md, 17 Oct 2026).

The engine ranks with np.argsort, which does not keep tied scores in a fixed order, and percentile scores tie often.
numpy 1.26 and numpy 2.x break those ties differently: System R 2000-2025 ends at $2.34M on numpy 1.26.4 / pandas
2.2.2 and at $2.87M on numpy 2.5.3 / pandas 3.0.5 (the versions every result in REPORT.md was scored with). Until the
engine broke ties explicitly, scoring on any other versions was refused unless ALLOW_ANY_ENV=1.

17 Oct 2026: the engine now breaks ties explicitly (engine.rank_desc), and decisions are identical on numpy 1.26.4 /
pandas 2.2.2 and 2.5.3 / 3.0.5 (test_determinism.py; account values differ by about a cent of float rounding). The
check stays so that stored results keep reproducing to the cent; ALLOW_ANY_ENV=1 is now safe for decisions.

    import envinfo; envinfo.check()          # at import of any scoring entry point
    envinfo.stamp(panel_path)                # dict to store with a result: versions + panel fingerprint
"""
import json, os, sys
import numpy as np
import pandas as pd

EXPECTED = {"numpy": "2.5.3", "pandas": "3.0.5"}


def versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}


def check():
    v = versions()
    bad = {k: (v[k], want) for k, want in EXPECTED.items() if v[k] != want}
    if bad and not os.environ.get("ALLOW_ANY_ENV"):
        raise SystemExit("envinfo: scoring environment differs from the one the results were scored with: "
                         + ", ".join(f"{k} {have} (expected {want})" for k, (have, want) in bad.items())
                         + ". Tied ranks resolve differently, so results will not reproduce. Use the container's python3 "
                           "(3.13) or set ALLOW_ANY_ENV=1 knowingly. See REBUILD.md.")
    return v


def stamp(panel_path):
    out = dict(versions())
    b = panel_path + ".BUILD.json"
    if os.path.exists(b):
        j = json.load(open(b))
        out.update(panel=j.get("panel"), panel_sha256=j.get("sha256"), panel_shape=j.get("shape"),
                   fetched_utc=(j.get("fetch") or {}).get("fetched_utc"))
    else:
        out.update(panel=os.path.basename(panel_path), panel_sha256=None, note="no BUILD.json: panel built before 17 Oct 2026")
    return out
