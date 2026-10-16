"""Ideas from the literature review (owner, 16 Oct 2026), on System Q, calendar-robust yardstick, Q = 70.2%.
1 momentum-stress pause vote (Heidari 2015; Barroso & Santa-Clara 2015); 2 mega-cap switch: 2-3 giant slots when SPY
beats the equal-weight universe by 3%+ over 63 days and QQQ is above a rising 50-day; 4 state-dependent pause speed
(Goulding, Harvey & Mazzoleni 2023); 6 dispersion cap at 6 stocks (Stivers & Sun 2010); 8 overnight-vs-intraday filter
(Lou, Polk & Skouras 2019); 12 late-stage volume filter (Lee & Swaminathan 2000); 14 Clenow gap filter."""
import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, robust as RB
PATH = "../data/panel_ndx.pkl"
if __name__ == "__main__":
    Y._init(PATH); Y.MM.lit()                                   # build the cached features once before forking
    pool = mp.get_context("fork").Pool(4, initializer=Y._init, initargs=(PATH,))
    Q = json.load(open("../system_Q.json"))["cfg"]
    G = [("1 Momentum-stress pause vote", dict(Q, pause_rules=("breadth", "hyg", "large", "momstress"))),
         ("2 Mega-cap switch: 3 giant slots", dict(Q, giant_dyn=[3, 0.03])),
         ("2b Mega-cap switch: 2 giant slots", dict(Q, giant_dyn=[2, 0.03])),
         ("4 State-dependent pause speed", dict(Q, state_pause=True)),
         ("6 Dispersion cap at 6 stocks", dict(Q, disp_cap=6)),
         ("8 Skip gains made overnight and faded intraday", dict(Q, overnight_filter=0.2)),
         ("12 Skip late-stage volume surges (50d > 2x 250d)", dict(Q, volr_max=2.0)),
         ("14 Clenow gap filter: no 15%+ day in 90 sessions", dict(Q, gap_max=0.15))]
    heads = []
    for lab, c in G:
        h, R = RB.rscore(pool, c); R.insert(0, "test", lab); heads.append(dict(test=lab, **h))
        R.to_csv("../audit/a53_detail.csv", mode="a", header=len(heads) == 1, index=False)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in heads[-1].items()}, flush=True)
        pd.DataFrame(heads).to_csv("../audit/a53_literature.csv", index=False)
