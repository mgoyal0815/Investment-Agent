import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_H.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System H",E)]
G+=[(f"A. Pause holds each on/off state at least {n} sessions",dict(E,pause_min_days=n)) for n in (5,10,20,40)]
G+=[(f"B. No rank sell in first {h} sessions unless down {int(g*100)}%+",dict(E,rank_min_hold=h,rank_grace_loss=g)) for h in (21,42,63) for g in (0.15,0.25)]
G+=[(f"C. No rank sells while SPY is {int(x*100)}%+ below its 1-year high",dict(E,crash_no_rank=x)) for x in (0.10,0.15,0.20)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/fixesH_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
