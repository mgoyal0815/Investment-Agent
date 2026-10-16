"""Boost Q (owner, 16 Oct 2026): ideas rejected earlier against the inflated one-calendar score, re-tested against
Q's honest 70.2% with the calendar-robust yardstick (the giants slot and the large-cap margin both passed this way)."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [("Rank: 35% 6-month strength / 65% distance above 200-day", dict(Q, robust_w=0.35)),
         ("Rank: 65% 6-month strength / 35% distance above 200-day", dict(Q, robust_w=0.65)),
         ("Sector ETFs compete for slots like stocks", dict(Q, etf_in_universe=True)),
         ("Giants get +5 percentile in the rank", dict(Q, giant_bonus=[20, 0.05])),
         ("Lean rank toward most-traded names (size_w 0.2)", dict(Q, size_w=0.2)),
         ("Top 5% strongest skip the ADX gate", dict(Q, adx_exempt_rs=0.95)),
         ("No rank sells while SPY is 20%+ below its high", dict(Q, crash_no_rank=0.2)),
         ("New buys get 21 sessions unless down 10%", dict(Q, rank_min_hold=21, rank_grace_loss=0.10)),
         ("Pause holds its state at least 10 sessions", dict(Q, pause_min_days=10)),
         ("Restart also on McClellan oscillator", dict(Q, unpause_on=Q["unpause_on"] + ["mcclellan"])),
         ("Index hurdle needs a 5-point lead", dict(Q, index_hurdle_margin=0.05)),
         ("At most 3 holdings per industry", dict(Q, max_per_sector=3)),
         ("Also pause when our book is 10% below its 63-day high", dict(Q, book_pause={"dd": 0.10, "mode": "or"})),
         ("Also pause when our book is 15% below its 63-day high", dict(Q, book_pause={"dd": 0.15, "mode": "or"})),
         ("Giants slot keeps buying during pauses", dict(Q, pause_giants=True)),
         ("Giants slot from top 10 instead of top 20", dict(Q, giant_top=10))]
    heads = []
    for lab, c in G:
        try:
            h, R = RB.rscore(pool, c)
        except Exception as e:
            print({"test": lab, "error": str(e)[:120]}, flush=True); continue
        R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a51_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a51_q_boost.csv", index=False)
