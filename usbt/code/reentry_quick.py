import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_F.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System F",E)]+[(f"Coppock, unpause {d} days",dict(E,unpause_on=["coppock"],unpause_days=d)) for d in (21,42,63,126,252)]+[(f"McClellan depth {v}, {d} days",dict(E,unpause_on=["mcclellan"],mcc_depth=v,unpause_days=d)) for v in (100,150,200) for d in (21,42)]+[(f"NH-NL depth {v}, {d} days",dict(E,unpause_on=["nhnl"],nhnl_depth=v,unpause_days=d)) for v in (0.1,0.2,0.3) for d in (21,42)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/reentry_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
