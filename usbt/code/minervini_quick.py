import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_F.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System F",E),("Template core (price/50/150/200 stacked, 200 rising)",dict(E,minervini="core")),
   ("Core, 200-day rising over 3 months",dict(E,minervini="core",minv_rise=63))]
G+=[(f"Full template, >= {lo}x 52w low, within {round((1-hi)*100)}% of 52w high",dict(E,minervini="full",minv_lo=lo,minv_hi=hi)) for lo in (1.25,1.3,1.5,2.0) for hi in (0.75,0.85,0.95)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/minervini_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
