import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_F.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System F",E)]+[(f"buy if up/down volume >= {v}",dict(E,udvr_min=v)) for v in (0.8,0.9,1.0,1.1,1.2,1.3,1.5)]+[(f"buy if money flow >= {v}",dict(E,cmf_min=v)) for v in (-0.1,0.0,0.05,0.1,0.15)]+[(f"buy if OBV trend >= {v}",dict(E,obv_min=v)) for v in (-2.0,0.0,2.0,4.0)]+[(f"buy if relative volume >= {v}",dict(E,rvol_min=v)) for v in (0.8,1.0,1.2)]+[(f"hold while up/down volume > {v}",dict(E,udvr_hold=v)) for v in (1.0,1.2,1.4)]+[(f"sell winner on distribution (up/down < {v})",dict(E,udvr_exit=v)) for v in (0.6,0.7,0.8,0.9)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/volume_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
