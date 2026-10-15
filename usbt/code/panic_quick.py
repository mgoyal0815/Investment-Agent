import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_J.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System J",E)]
G+=[(f"Dip rule triggered by QQQ RSI <= {r}",dict(E,dip_on=["QQQ"],pause_dip_index_rsi=r)) for r in (22,25,30)]
G+=[(f"Dip rule triggered by SPY or QQQ RSI <= {r}",dict(E,dip_on=["SPY","QQQ"],pause_dip_index_rsi=r)) for r in (22,25)]
G+=[(f"Panic ends the pause: {t.upper()} RSI <= {r}",dict(E,unpause_on=list(E["unpause_on"])+[f"{t}rsi{r}"])) for t,r in (("spy",20),("spy",25),("spy",30),("qqq",25))]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/panic_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
