import sys; sys.path.insert(0, "/home/user/Investment-Agent/usbt/code")
import json, multiprocessing as mp, pandas as pd, yardstick as Y, exp50_bayes_all as B
PATH="../data/panel.pkl"; Y._init(PATH)
pool=mp.get_context("fork").Pool(4,initializer=Y._init,initargs=(PATH,))
E=json.load(open("../system_G.json"))["cfg"]
PER={"choose 2012-16":(Y.starts("2012-01-01","2016-12-31",3),"2019-12-31"),"check 2000-08":(Y.starts("2000-01-01","2008-12-31",4),"2011-12-30"),"check 2020-25":(Y.starts("2020-01-01","2025-10-31",3),None)}
G=[("System G",E)]
G+=[(f"Park {int(f*100)}% in leading index during pauses while it is above its {n}d",dict(E,park_frac=f,park_frac_sma=n)) for n in (200,100) for f in (0.25,0.5,0.75)]
G+=[(f"Giants path: {s} slots for the {t} most-traded above a rising 200d",dict(E,giant_slots=s,giant_top=t)) for s in (2,4) for t in (10,20,50)]
rows=[]
for lab,c in G:
    r=B.evaluate(pool,c,PER); rows.append(dict(rule=lab,**{k:v for k,v in r.items() if k.endswith(("|beat","|ddok","|cagr"))})); print(lab,flush=True)
d=pd.DataFrame(rows); d.to_csv("../results/fixes2_quick.csv",index=False)
pd.set_option("display.width",250); print(d.round(3).to_string(index=False))
