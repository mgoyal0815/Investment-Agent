"""Calendar-robust yardstick (owner, 16 Oct 2026). A system's score is its average over 5 review calendars, not one.
The month-end review stays; every other review day in the system's `stagger` is shifted by -6, -3, 0, +3, +6
sessions (System O's 11th-session review becomes the 5th, 8th, 11th, 14th and 17th). Each calendar is scored from
every `step`-th monthly start 2000-2024 with runs stopped at 31 Dec 2025 (2026 not counted).
Reported: the mean over calendars (the headline), the worst calendar, and the spread."""
import numpy as np, pandas as pd
import yardstick as Y

SHIFTS = (-6, -3, 0, 3, 6)


def calendars(cfg):
    st = cfg.get("stagger") or [-1]
    if all(d < 0 for d in st):                       # month-end only: move the single review instead
        return [dict(cfg, stagger=[-1])] + [dict(cfg, stagger=[d]) for d in (4, 9, 14, 17)][:len(SHIFTS) - 1]
    return [dict(cfg, stagger=[d if d < 0 else int(np.clip(d + s, 0, 19)) for d in st]) for s in SHIFTS]


def rscore(pool, cfg, step=3, day=1):
    rows = []
    for c in calendars(cfg):
        T = Y.evaluate(pool, c, Y.starts(step=step, day=day)); s = Y.summary(T)
        e = {era: Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])["yrs_beat_SPY_QQQ"]
             for era, (a, b) in {"00-11": (2000, 2011), "12-19": (2012, 2019), "20-24": (2020, 2024)}.items()}
        ec = {era: Y.summary(T[(T.start.dt.year >= a) & (T.start.dt.year <= b)])["cal_years_beat"]
              for era, (a, b) in {"00-11": (2000, 2011), "12-19": (2012, 2019), "20-24": (2020, 2024)}.items()}
        rows.append(dict(stagger=str(c["stagger"]), beat=s["yrs_beat_SPY_QQQ"], cal=s["cal_years_beat"], smh3=s["r3_beat_SMH"], ddok=s["dd_ok"], **{f"cal {k}": v for k, v in ec.items()},
                         cagr=s["med_cagr"], worst=s["worst_dd"], **{f"beat {k}": v for k, v in e.items()}))
    R = pd.DataFrame(rows)
    head = dict(cal=R.cal.mean(), cal_min=R.cal.min(), cal_max=R.cal.max(), beat=R.beat.mean(), beat_min=R.beat.min(), beat_max=R.beat.max(), smh3=R.smh3.mean(), ddok=R.ddok.mean(),
                ddok_min=R.ddok.min(), cagr=R.cagr.mean(), worst=R.worst.min(),
                **{c: R[c].mean() for c in R.columns if c.startswith("beat ") or c.startswith("cal ")})
    return head, R


# ---------------------------------------------------------------- tie luck (owner, 17 Oct 2026)
# A system's ranks tie often (percentile scores). engine.rank_desc breaks ties by symbol (alphabetical) by default;
# cfg["tie_seed"] = s breaks them by a fixed random order of the symbols instead. A path's luck in those ties moved
# System R 2000-2025 between $1.8M and $3.3M. tscore() scores a system over TIE_SEEDS x the 5 review calendars and
# reports, per measure, the median over seeds (each seed's value is its average over the 5 calendars), the seed range
# (the tie-luck spread) and the alphabetical tie-break (the deterministic system that would run live).
TIE_SEEDS = tuple(range(1, 9))
MEASURES = ("cal", "beat", "smh3", "ddok", "cagr", "worst")


def tscore(pool, cfg, seeds=TIE_SEEDS, step=3, day=1):
    rows, grid = [], []
    for sd in (None,) + tuple(seeds):
        h, Rr = rscore(pool, dict(cfg, tie_seed=sd), step=step, day=day)
        rows.append(dict(tie_seed="alpha" if sd is None else sd, **h))
        Rr.insert(0, "tie_seed", "alpha" if sd is None else sd); grid.append(Rr)
    S = pd.DataFrame(rows)
    sd_ = S[S.tie_seed != "alpha"]
    head = {"alpha " + k: float(S[k].iloc[0]) for k in MEASURES}
    for k in MEASURES:
        head[f"{k} med"] = float(sd_[k].median()); head[f"{k} lo"] = float(sd_[k].min()); head[f"{k} hi"] = float(sd_[k].max())
        head[f"{k} spread"] = head[f"{k} hi"] - head[f"{k} lo"]
    head["worst all"] = float(min(S["worst"]))                       # worst single path over every seed and calendar
    return head, S, pd.concat(grid, ignore_index=True)


def provenance(seeds=TIE_SEEDS, **kw):
    """What to store with every result (owner, 17 Oct 2026): seeds, library versions, panel BUILD.json fingerprint."""
    import datetime as dt
    return dict(scored_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), tie_seeds=list(seeds),
                tie_default="alphabetical symbol (engine.rank_desc)", calendar_shifts=list(SHIFTS), env=Y.ENV, **kw)
