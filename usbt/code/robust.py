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
        rows.append(dict(stagger=str(c["stagger"]), beat=s["yrs_beat_SPY_QQQ"], smh3=s["r3_beat_SMH"], ddok=s["dd_ok"],
                         cagr=s["med_cagr"], worst=s["worst_dd"], **{f"beat {k}": v for k, v in e.items()}))
    R = pd.DataFrame(rows)
    head = dict(beat=R.beat.mean(), beat_min=R.beat.min(), beat_max=R.beat.max(), smh3=R.smh3.mean(), ddok=R.ddok.mean(),
                ddok_min=R.ddok.min(), cagr=R.cagr.mean(), worst=R.worst.min(),
                **{c: R[c].mean() for c in R.columns if c.startswith("beat ")})
    return head, R
