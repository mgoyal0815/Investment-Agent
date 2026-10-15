import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_J.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System J",E)]
G+=[("Paused: park in gold instead of Treasuries",dict(E,pause_park="GLD")),("Paused: park in gold while gold is above its 200d",dict(E,pause_park="GLD_trend"))]
G+=[(f"Paused: up to {k} slots for the {t} most-traded stocks above a rising 200d",dict(E,pause_giants=k,giant_top=t)) for k in (2,4,6) for t in (10,20)]
G+=[("Paused: gold (uptrend) + 4 giant slots of top 20",dict(E,pause_park="GLD_trend",pause_giants=4,giant_top=20))]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/refuge_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
