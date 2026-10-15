import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_I.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System I",E)]
G+=[(f"Paused: buy top-ranked stocks with RSI <= {r}, up to {k} per review",dict(E,pause_dip_rsi=r,pause_dip_slots=k)) for r,k in ((30,2),(35,2),(40,2),(50,2),(35,1),(35,5))]
G+=[(f"Paused: hold the leading index {h} sessions after SPY RSI <= {r}",dict(E,pause_dip_index_rsi=r,dip_hold=h)) for r in (25,30,35) for h in (21,42)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/dip_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
