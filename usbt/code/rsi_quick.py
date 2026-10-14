import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel_idx.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_E.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System E",E)]+[(f"sell half if RSI>{v}",dict(E,rsi_take=v)) for v in (95,90,85,80,75)]+[(f"sell all if RSI>{v}",dict(E,rsi_take=v,rsi_take_frac=1.0)) for v in (95,90,85,80)]+[(f"no rank sale if RSI<{v}",dict(E,rsi_sell_floor=v)) for v in (25,30,35,40,45,50)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/rsi_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
