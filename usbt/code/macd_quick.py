import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel_idx.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_E.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System E",E)]+[(f"buy only if MACD>={v}",dict(E,macd_buy_min=v)) for v in (-0.01,-0.005,0.0,0.0025,0.005,0.01)]+[(f"buy only if MACD<={v}",dict(E,macd_buy_max=v)) for v in (-0.005,0.0,0.0025,0.005,0.01)]+[(f"hold while MACD>{v}",dict(E,macd_hold=v)) for v in (-0.0025,0.0,0.0025,0.005)]+[(f"sell if in profit and MACD<{v}",dict(E,macd_exit=v)) for v in (-0.01,-0.005,-0.0025,0.0)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/macd_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
