import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_K.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System K",E)]
G+=[(f"A. No new stock buys with SPY within {h*100:g}% of its 1-year high",dict(E,skip_buys_near_high=h)) for h in (0.005,0.01,0.02)]
G+=[("B. Buy only if 6m strength vs SPY rose over the last week",dict(E,rs_rising=True))]
G+=[(f"C. Buy only {int(x*100)}%+ above the 200-day average",dict(E,dist200_min=x)) for x in (0.10,0.20)]
G+=[(f"D. Skip the most volatile {int(round((1-v)*100))}% of candidates",dict(E,vol_max_pct=v)) for v in (0.8,0.9)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/entry_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
