import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_H.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System H",E)]
G+=[(f"Index hurdle: stock's {lb}-day return must beat best of SPY/QQQ/SMH by {int(mg*100)} pts",dict(E,index_hurdle=True,index_hurdle_lb=lb,index_hurdle_margin=mg))
    for lb,mg in ((126,-0.05),(126,0.0),(126,0.05),(126,0.10),(63,0.0),(63,0.05),(252,0.0))]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/hurdle_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
